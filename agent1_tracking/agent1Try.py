# agent1_tracking/agent1.py
# Agent 1: Full production pipeline — SPLIT DETECTION + FACE + POSE
#
# ── WHAT CHANGED IN THIS VERSION ──────────────────────────────────────────────
# 1. TRACKER: switched from ByteTrack to BoT-SORT with with_reid=True (see
#    custom_tracker.yaml). ByteTrack has NO appearance signal — it matches
#    tracks purely by motion/IOU, which is exactly why two people crossing
#    paths or standing close can get their identities swapped mid-track:
#    nothing is checking "does this still look like the same person" at the
#    actual matching step. BoT-SORT+ReID checks appearance THERE, which is
#    the structural fix; the face-gallery swap-correction below remains as
#    a safety net for whatever slips through.
# 2. FACE-GALLERY RE-ENTRY (restored + hardened): a brand-new raw track is
#    now checked against every known person's face gallery BEFORE body
#    embedding/HSV/position — a face is far more stable across a multi-
#    minute absence than body appearance (lighting/pose/clothing drift).
#    Hardened with two safeguards since this is a PERMANENT one-shot
#    decision: requires several face samples averaged together (not one
#    frame), and requires the best candidate to beat the second-best by a
#    clear margin — an ambiguous close call falls through to body-based
#    matching instead of guessing and locking in a wrong identity forever.
# 3. POSE ESTIMATION (new): a lightweight pose model (yolov8n-pose) runs on
#    each confirmed person's crop, throttled like the face/phone passes.
#    Classifies SLEEPING vs AWAKE from nose-vs-shoulder keypoint height —
#    when the head keypoint is at or below shoulder height, that's a strong,
#    simple signal of a slumped/sleeping posture. Logged to the DB as
#    event_type='pose_detected' (reusing the existing generic schema, no
#    migration needed) for Agent 2 to build a sleeping-violation rule on.
#
# Position-based Re-ID, the motion-validity gate, zone-anchored home-zone
# voting, and the offline Pass 4 identity reconciliation are all preserved.

import os
import sys
import cv2
import json
import sqlite3
import numpy as np
from ultralytics import YOLO
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH as DEFAULT_VIDEO_PATH, MODEL_NAME, DATABASE_PATH

FACE_MODEL_PATH  = os.path.join(ROOT_DIR, "face_detection_model.pt")
PHONE_MODEL_PATH = os.path.join(ROOT_DIR, "best_phone.pt")
POSE_MODEL_NAME  = "yolov8n-pose.pt"   # ultralytics auto-downloads this like
                                        # MODEL_NAME if not already cached

# ── DEVICE SETUP ───────────────────────────────────────────────────────────────
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF   = DEVICE.startswith("cuda")
if DEVICE.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU detected: {gpu_name} — running on {DEVICE} (FP16={HALF})")
else:
    print("[WARN] No GPU detected — running on CPU. This will be significantly "
          "slower than real-time; consider running on the GPU machine.")

# ── ZONE DATA ──────────────────────────────────────────────────────────────────
ZONES_JSON_PATH = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")

def _load_zones_norm():
    if not os.path.exists(ZONES_JSON_PATH):
        return []
    try:
        with open(ZONES_JSON_PATH) as f:
            return json.load(f).get("zones", [])
    except Exception:
        return []

_ZONES_NORM = _load_zones_norm()

def _zone_for_point(cx, cy, zones_norm, fw, fh):
    for zi, zone in enumerate(zones_norm):
        poly = np.array([[p[0] * fw, p[1] * fh] for p in zone], dtype=np.float32)
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            return zi
    return None

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

print("\n=== AGENT 1 STARTING: DETECTION + FACE + POSE (BOT-SORT REID) ===")
print(f"[INFO] Video source: {VIDEO_PATH}"
      f"{'  (overridden via argv)' if len(sys.argv) > 1 else '  (from config.py default)'}")

# ── DATABASE ──────────────────────────────────────────────────────────────────
conn   = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("DELETE FROM events")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
cursor.execute("DROP TABLE IF EXISTS identity_merges")
conn.commit()
print(f"[INFO] Database: {DATABASE_PATH}  — cleared, starting fresh.")

# ── VIDEO ─────────────────────────────────────────────────────────────────────
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit()

fps          = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video: {total_frames} frames @ {fps:.2f} FPS  "
      f"({frame_width}x{frame_height})")

_ZONES_PX = [
    [(int(p[0] * frame_width), int(p[1] * frame_height)) for p in zone]
    for zone in _ZONES_NORM
]
if _ZONES_PX:
    print(f"[INFO] {len(_ZONES_PX)} calibrated zone(s) loaded — will be drawn on every frame.")
else:
    print("[INFO] No zones.json found (or it's empty) — no zone overlay will be drawn.")

# ── OUTPUT VIDEO ──────────────────────────────────────────────────────────────
output_dir  = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "agent1_output8.mp4")
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

LIVE_FRAME_PATH  = os.path.join(ROOT_DIR, "live_frame.jpg")
LIVE_FRAME_EVERY = 3

face_crops_dir = os.path.join(ROOT_DIR, "face_crops")
os.makedirs(face_crops_dir, exist_ok=True)

# ── LOAD DETECTION MODELS ──────────────────────────────────────────────────────
print(f"[INFO] Loading person model        : {MODEL_NAME}")
model_main = YOLO(MODEL_NAME)

face_model = None
if os.path.exists(FACE_MODEL_PATH):
    print(f"[INFO] Loading face model          : {FACE_MODEL_PATH}")
    face_model = YOLO(FACE_MODEL_PATH)
else:
    print(f"[WARN] Face model NOT found at {FACE_MODEL_PATH} — face detection disabled.")

phone_model = None
if os.path.exists(PHONE_MODEL_PATH):
    print(f"[INFO] Loading custom phone model  : {PHONE_MODEL_PATH}")
    phone_model = YOLO(PHONE_MODEL_PATH)
else:
    print(f"[WARN] Phone model NOT found at {PHONE_MODEL_PATH} — falling back to COCO class 67 on main model.")

pose_model = None
try:
    print(f"[INFO] Loading pose model          : {POSE_MODEL_NAME}")
    pose_model = YOLO(POSE_MODEL_NAME)
except Exception as e:
    print(f"[WARN] Could not load pose model ({e}) — sleeping-pose detection disabled.")

# ── DEEP APPEARANCE EMBEDDING MODEL ────────────────────────────────────────────
EMBEDDING_REID_ENABLED = True
EMBED_MODEL      = None
EMBED_TRANSFORM  = None

if EMBEDDING_REID_ENABLED:
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet18_Weights
        import torchvision.transforms as T

        _backbone = tvm.resnet18(weights=ResNet18_Weights.DEFAULT)
        _backbone.fc = torch.nn.Identity()
        _backbone.eval().to(DEVICE)
        if HALF:
            _backbone.half()
        EMBED_MODEL = _backbone

        EMBED_TRANSFORM = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[INFO] Deep appearance embedding model (ResNet18) loaded.")
    except Exception as e:
        print(f"[WARN] Could not load embedding model ({e}). "
              f"Falling back to HSV histogram appearance matching only.")
        EMBEDDING_REID_ENABLED = False

def compute_embedding(frame, box):
    if not EMBEDDING_REID_ENABLED or EMBED_MODEL is None:
        return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    h = y2 - y1
    ty1, ty2 = y1 + int(h * 0.25), y1 + int(h * 0.85)
    if ty2 <= ty1:
        ty1, ty2 = y1, y2
    crop = frame[ty1:ty2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = EMBED_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
    if HALF:
        tensor = tensor.half()
    with torch.no_grad():
        feat = EMBED_MODEL(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).float().cpu().numpy()

def compute_face_embedding(frame, box):
    if not EMBEDDING_REID_ENABLED or EMBED_MODEL is None:
        return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = EMBED_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
    if HALF:
        tensor = tensor.half()
    with torch.no_grad():
        feat = EMBED_MODEL(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).float().cpu().numpy()

def cosine_sim(a, b):
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))

def compute_appearance_hist(frame, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    h = y2 - y1
    ty1, ty2 = y1 + int(h * 0.25), y1 + int(h * 0.85)
    crop = frame[ty1:ty2, x1:x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# ── POSE / SLEEPING CLASSIFICATION ────────────────────────────────────────────
POSE_CHECK_EVERY_N_FRAMES = 5
POSE_CROP_IMGSZ           = 320
POSE_CONF                 = 0.30
POSE_KP_CONF_FLOOR        = 0.30
SLEEP_HEAD_DROP_MARGIN_PX = 10   # tolerance around the shoulder line

# COCO 17-keypoint order (ultralytics pose models)
KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER = 0, 5, 6

def classify_sleep_pose(kp_xy, kp_conf):
    """
    kp_xy: (17,2) array of keypoint coords (crop-local — fine, since we only
    need the RELATIVE nose-vs-shoulder height, not absolute position).
    kp_conf: (17,) array of per-keypoint confidence.

    Heuristic: in a normal seated/standing posture the head sits clearly
    ABOVE the shoulder line. If the nose keypoint has dropped to or below
    shoulder height, that's a strong, simple signal of a slumped/sleeping
    posture — deliberately simple and explainable rather than a black-box
    pose classifier, given this is a supporting signal for a rule engine,
    not the sole source of truth.

    Returns ('SLEEPING'|'AWAKE', avg_confidence), or (None, 0.0) if the
    needed keypoints aren't confidently visible (e.g. heavily occluded) —
    caller should skip logging rather than guess.
    """
    nose_conf = float(kp_conf[KP_NOSE])
    ls_conf   = float(kp_conf[KP_L_SHOULDER])
    rs_conf   = float(kp_conf[KP_R_SHOULDER])

    if nose_conf < POSE_KP_CONF_FLOOR or (ls_conf < POSE_KP_CONF_FLOOR and rs_conf < POSE_KP_CONF_FLOOR):
        return None, 0.0

    shoulder_ys = []
    if ls_conf >= POSE_KP_CONF_FLOOR:
        shoulder_ys.append(float(kp_xy[KP_L_SHOULDER][1]))
    if rs_conf >= POSE_KP_CONF_FLOOR:
        shoulder_ys.append(float(kp_xy[KP_R_SHOULDER][1]))
    mid_shoulder_y = float(np.mean(shoulder_ys))
    nose_y = float(kp_xy[KP_NOSE][1])

    used_confs = [c for c in (nose_conf, ls_conf, rs_conf) if c >= POSE_KP_CONF_FLOOR]
    avg_conf = float(np.mean(used_confs))

    if nose_y >= mid_shoulder_y - SLEEP_HEAD_DROP_MARGIN_PX:
        return "SLEEPING", avg_conf
    return "AWAKE", avg_conf

# ── TRACKING STATE ────────────────────────────────────────────────────────────
raw_id_counters  = {}
id_registry      = {}
next_clean_id    = 1

last_seen_frame  = {}
position_history = {}
last_known_pos   = {}
last_known_emb   = {}
last_known_hist  = {}

id_emb_sum    = {}
id_emb_count  = {}
id_hist_sum   = {}
id_hist_count = {}
id_frame_set  = {}
id_zone_votes = {}
id_first_seen = {}

HOME_ZONE_WARMUP_SECONDS = 20
HOME_ZONE_WARMUP_FRAMES  = int(fps * HOME_ZONE_WARMUP_SECONDS) if fps else 600

GRACE_PERIOD_FRAMES   = 45
MIN_FRAMES_TO_CONFIRM = 10
REID_DISTANCE_THRESH  = 250

MOTION_VALIDITY_ENABLED   = True
MIN_MOVEMENT_SPREAD       = 3
MOTION_CHECK_GRACE_FRAMES = 150
MOTION_CHECK_HARD_CAP     = 450
STATIC_OBJECT_MIN_AVG_CONF = 0.55
rejected_raw_ids = set()
raw_conf_history  = {}

DEDUP_CENTER_DIST_THRESH = 40
DEDUP_SIZE_RATIO_THRESH  = 0.35

PERSISTENT_DUPLICATE_FRAMES_THRESHOLD = 15
_dup_pair_counts = {}

EMBEDDING_MATCH_FLOOR = 0.72
APPEARANCE_SEARCH_RADIUS = 650
APPEARANCE_MATCH_FLOOR   = 0.55

ONLINE_ZONE_BOOST_ENABLED   = True
ZONE_ONLINE_EMBEDDING_FLOOR = 0.55
ZONE_ONLINE_MIN_VOTES       = 30

# ── FACE-ANCHORED SWAP CORRECTION (live, between simultaneously active people) ─
FACE_SWAP_CHECK_ENABLED       = True
FACE_EMB_MIN_CONF_FOR_CHECK   = 0.65
FACE_EMB_OWN_MATCH_FLOOR      = 0.45
FACE_EMB_SWAP_CONFIRM_FLOOR   = 0.55
SWAP_COOLDOWN_FRAMES          = 90

FACE_SWAP_PERSISTENCE_THRESHOLD = 4
_face_swap_evidence = {}

FACE_GALLERY_MAX_SIZE          = 5
FACE_GALLERY_DIVERSITY_MAX_SIM = 0.85
last_known_face_gallery = {}

# ── FACE-GALLERY RE-ENTRY MATCH (for long absences) ───────────────────────────
# Checked FIRST at graduation, before body embedding/HSV/position — a face
# is far more stable across a multi-minute absence than body appearance.
# Hardened against a single bad-angle read deciding a PERMANENT identity:
# requires several averaged samples, and a clear margin over the runner-up.
FACE_REENTRY_CHECK_ENABLED = True
FACE_REENTRY_MATCH_FLOOR   = 0.45
FACE_REENTRY_MIN_SAMPLES   = 3
FACE_REENTRY_MARGIN        = 0.10
raw_face_embeddings = {}   # raw_id -> [face embeddings collected during probation]

def _face_gallery_best_sim(cid, face_emb):
    gallery = last_known_face_gallery.get(cid)
    if not gallery:
        return None
    return max(cosine_sim(face_emb, g) for g in gallery)

def _face_gallery_add(cid, face_emb):
    gallery = last_known_face_gallery.setdefault(cid, [])
    if len(gallery) >= FACE_GALLERY_MAX_SIZE:
        return
    if gallery and max(cosine_sim(face_emb, g) for g in gallery) >= FACE_GALLERY_DIVERSITY_MAX_SIM:
        return
    gallery.append(face_emb)

_swap_cooldown = {}
_face_check_stats = {'unlinked': 0, 'low_conf': 0, 'checked': 0, 'own_mismatch': 0}

# ── DETECTION THRESHOLDS ───────────────────────────────────────────────────────
PERSON_CONF = 0.20
PERSON_IOU  = 0.15

PHONE_CONF  = 0.60
PHONE_IOU   = 0.30
PHONE_COCO_FALLBACK_CONF = 0.40
PHONE_COCO_FALLBACK_IOU  = 0.30

FACE_CONF   = 0.60
FACE_IOU    = 0.40

MERGE_EMBEDDING_THRESHOLD      = 0.62
ZONE_MERGE_EMBEDDING_THRESHOLD = 0.42
MERGE_APPEARANCE_THRESHOLD      = 0.65
ZONE_MERGE_APPEARANCE_THRESHOLD = 0.40

ROI_PADDING_RATIO           = 0.15
PHONE_CROP_IMGSZ            = 640
FACE_CROP_IMGSZ             = 320
FACE_DETECT_EVERY_N_FRAMES  = 1

best_face_conf = {}
best_face_crop = {}

frame_count = 0

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def expand_box(box, pad_ratio, fw, fh):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return (max(0, x1 - px), max(0, y1 - py), min(fw - 1, x2 + px), min(fh - 1, y2 + py))

def _boxes_are_duplicate(boxA, boxB,
                          center_thresh=DEDUP_CENTER_DIST_THRESH,
                          size_ratio_thresh=DEDUP_SIZE_RATIO_THRESH):
    axc, ayc = get_center(boxA)
    bxc, byc = get_center(boxB)
    if np.sqrt((axc - bxc) ** 2 + (ayc - byc) ** 2) > center_thresh:
        return False
    aw, ah = boxA[2] - boxA[0], boxA[3] - boxA[1]
    bw, bh = boxB[2] - boxB[0], boxB[3] - boxB[1]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    w_ratio = abs(aw - bw) / max(aw, bw)
    h_ratio = abs(ah - bh) / max(ah, bh)
    return w_ratio <= size_ratio_thresh and h_ratio <= size_ratio_thresh

def _current_home_zone(cid, min_votes=ZONE_ONLINE_MIN_VOTES):
    votes = id_zone_votes.get(cid)
    if not votes:
        return None
    zone, count = max(votes.items(), key=lambda kv: kv[1])
    return zone if count >= min_votes else None

def try_capture_probation_face(frame, raw_id, box):
    """Called every frame during probation to opportunistically collect
    face evidence — several samples across the window, not one snapshot,
    since the re-entry decision below is a permanent, one-shot call."""
    if face_model is None:
        return
    ex1, ey1, ex2, ey2 = expand_box(box, 0.15, frame_width, frame_height)
    crop = frame[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return
    try:
        res = face_model.predict(crop, conf=FACE_CONF, iou=FACE_IOU, imgsz=FACE_CROP_IMGSZ,
                                  device=DEVICE, half=HALF, verbose=False)
    except Exception:
        return
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return
    fconfs = r0.boxes.conf.tolist()
    best_i = max(range(len(fconfs)), key=lambda i: fconfs[i])
    if fconfs[best_i] < FACE_CONF:
        return
    fboxes = r0.boxes.xyxy.int().tolist()
    fx1, fy1, fx2, fy2 = fboxes[best_i]
    face_box_full = (fx1 + ex1, fy1 + ey1, fx2 + ex1, fy2 + ey1)
    emb = compute_face_embedding(frame, face_box_full)
    if emb is not None:
        raw_face_embeddings.setdefault(raw_id, []).append(emb)

def find_face_match_for_reentry(face_embeddings, exclude_cids=None):
    """See module docstring. Returns matching clean_id, or None (falls
    back to body-based matching)."""
    if not FACE_REENTRY_CHECK_ENABLED or not last_known_face_gallery or not face_embeddings:
        return None
    if len(face_embeddings) < FACE_REENTRY_MIN_SAMPLES:
        return None

    exclude_cids = exclude_cids or set()

    avg = np.mean(face_embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return None
    query_emb = avg / norm

    scores = []
    for cid in last_known_face_gallery:
        if cid in exclude_cids:
            continue
        s = _face_gallery_best_sim(cid, query_emb)
        if s is not None:
            scores.append((s, cid))
    if not scores:
        return None

    scores.sort(reverse=True)
    best_score, best_cid = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    if best_score >= FACE_REENTRY_MATCH_FLOOR and margin >= FACE_REENTRY_MARGIN:
        print(f"  [FACE RE-ID] Matched via face gallery (best={best_score:.2f}, "
              f"2nd-best={second_score:.2f}, margin={margin:.2f}, "
              f"{len(face_embeddings)} sample(s) averaged) — checked before body appearance.")
        return best_cid
    elif best_score >= FACE_REENTRY_MATCH_FLOOR:
        print(f"  [FACE RE-ID AMBIGUOUS] Best candidate Person {best_cid} scored "
              f"{best_score:.2f} but 2nd-best was {second_score:.2f} (margin={margin:.2f}, "
              f"need {FACE_REENTRY_MARGIN:.2f}) — too close to call, falling back to "
              f"body-based matching instead of guessing.")

    return None

def find_matching_clean_id(avg_cx, avg_cy, frame, box, exclude_cids=None):
    exclude_cids = exclude_cids or set()

    best_cid, best_dist = None, float("inf")
    for cid, (lx, ly) in last_known_pos.items():
        if cid in exclude_cids:
            continue
        dist = np.sqrt((avg_cx - lx) ** 2 + (avg_cy - ly) ** 2)
        if dist < best_dist:
            best_dist, best_cid = dist, cid
    if best_cid is not None and best_dist <= REID_DISTANCE_THRESH:
        return best_cid

    cand_emb = compute_embedding(frame, box) if EMBEDDING_REID_ENABLED else None

    if ONLINE_ZONE_BOOST_ENABLED and cand_emb is not None:
        cur_zone = _zone_for_point(avg_cx, avg_cy, _ZONES_NORM, frame_width, frame_height)
        if cur_zone is not None:
            best_zone_cid, best_zone_score = None, -1.0
            for cid, emb in last_known_emb.items():
                if cid in exclude_cids:
                    continue
                if _current_home_zone(cid) != cur_zone:
                    continue
                score = cosine_sim(cand_emb, emb)
                if score >= ZONE_ONLINE_EMBEDDING_FLOOR and score > best_zone_score:
                    best_zone_score, best_zone_cid = score, cid
            if best_zone_cid is not None:
                print(f"  [ZONE+EMBEDDING RE-ID] Matched via home-zone + appearance "
                      f"(similarity={best_zone_score:.2f}, zone={cur_zone})")
                return best_zone_cid

    if cand_emb is not None:
        best_emb_cid, best_emb_score = None, -1.0
        for cid, emb in last_known_emb.items():
            if cid in exclude_cids:
                continue
            score = cosine_sim(cand_emb, emb)
            if score >= EMBEDDING_MATCH_FLOOR and score > best_emb_score:
                best_emb_score, best_emb_cid = score, cid
        if best_emb_cid is not None:
            print(f"  [EMBEDDING RE-ID] Matched via deep appearance "
                  f"(similarity={best_emb_score:.2f})")
            return best_emb_cid

    cand_hist = compute_appearance_hist(frame, box)
    if cand_hist is not None:
        best_h_cid, best_h_score = None, -1.0
        for cid, (lx, ly) in last_known_pos.items():
            if cid in exclude_cids:
                continue
            dist = np.sqrt((avg_cx - lx) ** 2 + (avg_cy - ly) ** 2)
            if dist > APPEARANCE_SEARCH_RADIUS:
                continue
            hist = last_known_hist.get(cid)
            if hist is None:
                continue
            score = cv2.compareHist(cand_hist, hist, cv2.HISTCMP_CORREL)
            if score >= APPEARANCE_MATCH_FLOOR and score > best_h_score:
                best_h_score, best_h_cid = score, cid
        if best_h_cid is not None:
            print(f"  [HSV RE-ID] Matched via clothing histogram fallback "
                  f"(correlation={best_h_score:.2f})")
            return best_h_cid

    return None

def face_inside_person(face_box, person_boxes_map):
    fx1, fy1, fx2, fy2 = face_box
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    for cid, (px1, py1, px2, py2) in person_boxes_map.items():
        if px1 <= fcx <= px2 and py1 <= fcy <= py2:
            return cid
    best_cid, best_dist = None, float("inf")
    for cid, (px1, py1, px2, py2) in person_boxes_map.items():
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        dist = np.sqrt((fcx - pcx) ** 2 + (fcy - pcy) ** 2)
        if dist < best_dist:
            best_dist, best_cid = dist, cid
    return best_cid if best_dist < 300 else None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count += 1
    current_ts   = frame_count / fps
    active_people = set()

    res_people = model_main.track(
        frame,
        persist=True,
        tracker="agent1_tracking/custom_tracker.yaml",
        classes=[0],
        conf=PERSON_CONF,
        iou=PERSON_IOU,
        imgsz=1280,
        device=DEVICE,
        half=HALF,
        verbose=False
    )

    person_frame    = res_people[0]
    confirmed_boxes = {}

    if person_frame.boxes.id is not None:
        raw_ids     = person_frame.boxes.id.int().tolist()
        boxes       = person_frame.boxes.xyxy.int().tolist()
        confidences = person_frame.boxes.conf.tolist()

        order = sorted(range(len(raw_ids)), key=lambda i: confidences[i], reverse=True)
        kept_idx = []
        for i in order:
            is_dup = False
            for j in kept_idx:
                if not _boxes_are_duplicate(boxes[i], boxes[j]):
                    continue

                dup_raw_id, surv_raw_id = raw_ids[i], raw_ids[j]
                dup_cid  = id_registry.get(dup_raw_id)
                surv_cid = id_registry.get(surv_raw_id)

                if dup_cid is not None and surv_cid is not None and dup_cid != surv_cid:
                    pair_key = frozenset((dup_cid, surv_cid))
                    _dup_pair_counts[pair_key] = _dup_pair_counts.get(pair_key, 0) + 1
                    occurrences = _dup_pair_counts[pair_key]

                    if occurrences < PERSISTENT_DUPLICATE_FRAMES_THRESHOLD:
                        print(f"[WARN] raw_id={dup_raw_id} (Person {dup_cid}) and "
                              f"raw_id={surv_raw_id} (Person {surv_cid}) produced "
                              f"near-identical boxes this frame but are ALREADY "
                              f"different confirmed people ({occurrences}/"
                              f"{PERSISTENT_DUPLICATE_FRAMES_THRESHOLD} occurrences "
                              f"so far) — leaving both identities untouched for now.")
                        continue

                    keep, drop = (dup_cid, surv_cid) if dup_cid < surv_cid else (surv_cid, dup_cid)
                    print(f"[LIVE MERGE] Person {drop} and Person {keep} have produced "
                          f"near-identical boxes in {occurrences} separate frames — "
                          f"this is almost certainly ONE physical person double-tracked "
                          f"as two IDs, not two people. Merging Person {drop} into "
                          f"Person {keep} now, retroactively fixing already-logged DB rows.")

                    cursor.execute("UPDATE events SET person_id = ? WHERE person_id = ?", (keep, drop))
                    conn.commit()

                    for rid, cid in list(id_registry.items()):
                        if cid == drop:
                            id_registry[rid] = keep

                    if drop in last_known_pos:
                        last_known_pos[keep] = last_known_pos.pop(drop)
                    if drop in last_known_emb and keep not in last_known_emb:
                        last_known_emb[keep] = last_known_emb.pop(drop)
                    else:
                        last_known_emb.pop(drop, None)
                    if drop in last_known_hist and keep not in last_known_hist:
                        last_known_hist[keep] = last_known_hist.pop(drop)
                    else:
                        last_known_hist.pop(drop, None)
                    if drop in last_known_face_gallery:
                        keep_gallery = last_known_face_gallery.setdefault(keep, [])
                        for g in last_known_face_gallery.pop(drop):
                            if len(keep_gallery) >= FACE_GALLERY_MAX_SIZE:
                                break
                            if not keep_gallery or max(cosine_sim(g, k) for k in keep_gallery) < FACE_GALLERY_DIVERSITY_MAX_SIM:
                                keep_gallery.append(g)

                    if drop in id_emb_sum:
                        id_emb_sum[keep] = id_emb_sum.get(keep, 0) + id_emb_sum.pop(drop)
                        id_emb_count[keep] = id_emb_count.get(keep, 0) + id_emb_count.pop(drop, 0)
                    if drop in id_hist_sum:
                        id_hist_sum[keep] = id_hist_sum.get(keep, 0) + id_hist_sum.pop(drop)
                        id_hist_count[keep] = id_hist_count.get(keep, 0) + id_hist_count.pop(drop, 0)
                    if drop in id_frame_set:
                        id_frame_set.setdefault(keep, set()).update(id_frame_set.pop(drop))
                    if drop in id_zone_votes:
                        keep_votes = id_zone_votes.setdefault(keep, {})
                        for z, c in id_zone_votes.pop(drop).items():
                            keep_votes[z] = keep_votes.get(z, 0) + c
                    if drop in id_first_seen:
                        keep_first = id_first_seen.get(keep, id_first_seen[drop])
                        id_first_seen[keep] = min(keep_first, id_first_seen.pop(drop))
                    best_face_conf.pop(drop, None)
                    best_face_crop.pop(drop, None)
                    _dup_pair_counts.pop(pair_key, None)

                    last_seen_frame[dup_raw_id] = frame_count
                    is_dup = True
                    break

                last_seen_frame[dup_raw_id] = frame_count
                if surv_cid is not None and dup_cid is None:
                    id_registry[dup_raw_id] = surv_cid
                    print(f"  [DEDUP] raw_id={dup_raw_id} overlaps confirmed "
                          f"raw_id={surv_raw_id} — aliased to Person {surv_cid}.")
                elif dup_cid is not None and surv_cid is None:
                    id_registry[surv_raw_id] = dup_cid
                    print(f"  [DEDUP] raw_id={surv_raw_id} overlaps confirmed "
                          f"raw_id={dup_raw_id} — aliased to Person {dup_cid}.")
                is_dup = True
                break
            if not is_dup:
                kept_idx.append(i)
        kept_idx = sorted(kept_idx)
        raw_ids     = [raw_ids[i] for i in kept_idx]
        boxes       = [boxes[i] for i in kept_idx]
        confidences = [confidences[i] for i in kept_idx]

        currently_active_cids = {
            id_registry[rid] for rid in raw_ids if rid in id_registry
        }

        for raw_id, box, _conf_this_frame in zip(raw_ids, boxes, confidences):
            cx, cy = get_center(box)

            if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            else:
                raw_id_counters[raw_id] = 1
                position_history[raw_id]  = []
                raw_conf_history[raw_id]  = []

            last_seen_frame[raw_id] = frame_count

            if raw_id not in id_registry:
                position_history.setdefault(raw_id, []).append((cx, cy))
                raw_conf_history.setdefault(raw_id, []).append(_conf_this_frame)
                try_capture_probation_face(frame, raw_id, box)

            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM \
                    and raw_id not in id_registry and raw_id not in rejected_raw_ids:
                positions = position_history.get(raw_id, [(cx, cy)])
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                spread = (max(xs) - min(xs)) + (max(ys) - min(ys))

                if MOTION_VALIDITY_ENABLED and spread < MIN_MOVEMENT_SPREAD:
                    conf_hist = raw_conf_history.get(raw_id, [])
                    avg_conf  = float(np.mean(conf_hist)) if conf_hist else 0.0

                    if avg_conf >= STATIC_OBJECT_MIN_AVG_CONF:
                        print(f"[CAUTION] raw_id={raw_id} graduating after "
                              f"{raw_id_counters[raw_id]} frames with near-zero movement "
                              f"(spread={spread}px) but high avg confidence "
                              f"({avg_conf:.2f}) — treating as a real, still person.")
                    elif raw_id_counters[raw_id] < MOTION_CHECK_HARD_CAP:
                        continue
                    else:
                        rejected_raw_ids.add(raw_id)
                        raw_face_embeddings.pop(raw_id, None)
                        print(f"[REJECTED] raw_id={raw_id} discarded after "
                              f"{raw_id_counters[raw_id]} frames — near-zero movement "
                              f"(spread={spread}px) and low avg confidence "
                              f"({avg_conf:.2f}). Likely a static object, not a person.")
                        continue

                avg_cx, avg_cy = int(np.mean(xs)), int(np.mean(ys))

                matched = find_face_match_for_reentry(
                    raw_face_embeddings.get(raw_id, []), exclude_cids=currently_active_cids
                )
                if matched is None:
                    matched = find_matching_clean_id(
                        avg_cx, avg_cy, frame, box, exclude_cids=currently_active_cids
                    )

                if matched is not None:
                    id_registry[raw_id] = matched
                    currently_active_cids.add(matched)
                    print(f"[RE-ENTRY] Person {matched} returned! (raw_id={raw_id})")
                else:
                    id_registry[raw_id] = next_clean_id
                    currently_active_cids.add(next_clean_id)
                    next_clean_id += 1
                    print(f"[NEW PERSON] Person ID {id_registry[raw_id]} confirmed.")
                raw_face_embeddings.pop(raw_id, None)

        for raw_id, box in zip(raw_ids, boxes):
            if raw_id not in id_registry:
                continue
            clean_id = id_registry[raw_id]
            last_known_pos[clean_id] = get_center(box)

            emb = compute_embedding(frame, box)
            if emb is not None:
                last_known_emb[clean_id] = emb
                if clean_id not in id_emb_sum:
                    id_emb_sum[clean_id] = emb.copy()
                else:
                    id_emb_sum[clean_id] += emb
                id_emb_count[clean_id] = id_emb_count.get(clean_id, 0) + 1

            hist = compute_appearance_hist(frame, box)
            if hist is not None:
                last_known_hist[clean_id] = hist
                if clean_id not in id_hist_sum:
                    id_hist_sum[clean_id] = hist.copy()
                else:
                    id_hist_sum[clean_id] += hist
                id_hist_count[clean_id] = id_hist_count.get(clean_id, 0) + 1

        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id not in id_registry:
                continue
            clean_id = id_registry[raw_id]
            active_people.add(clean_id)
            x1, y1, x2, y2 = box
            confirmed_boxes[clean_id] = (x1, y1, x2, y2)

            id_frame_set.setdefault(clean_id, set()).add(frame_count)

            if _ZONES_NORM:
                if clean_id not in id_first_seen:
                    id_first_seen[clean_id] = frame_count
                if (frame_count - id_first_seen[clean_id]) <= HOME_ZONE_WARMUP_FRAMES:
                    pcx, pcy = get_center(box)
                    zone_idx = _zone_for_point(pcx, pcy, _ZONES_NORM, frame_width, frame_height)
                    if zone_idx is not None:
                        votes = id_zone_votes.setdefault(clean_id, {})
                        votes[zone_idx] = votes.get(zone_idx, 0) + 1

            cursor.execute("""
                INSERT INTO events
                    (timestamp, person_id, event_type, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (round(current_ts, 3), clean_id, "detected",
                  round(conf, 4), x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{clean_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    phone_dets = []

    if confirmed_boxes and (phone_model is not None or True):
        crop_imgs, crop_meta = [], []
        for cid, box in confirmed_boxes.items():
            ex1, ey1, ex2, ey2 = expand_box(box, ROI_PADDING_RATIO, frame_width, frame_height)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            crop_imgs.append(crop)
            crop_meta.append((cid, ex1, ey1))

        if crop_imgs:
            if phone_model is not None:
                results = phone_model.predict(
                    crop_imgs, conf=PHONE_CONF, iou=PHONE_IOU, imgsz=PHONE_CROP_IMGSZ,
                    device=DEVICE, half=HALF, verbose=False
                )
            else:
                results = model_main.predict(
                    crop_imgs, classes=[67], conf=PHONE_COCO_FALLBACK_CONF,
                    iou=PHONE_COCO_FALLBACK_IOU,
                    imgsz=PHONE_CROP_IMGSZ, device=DEVICE, half=HALF, verbose=False
                )
            for (cid, ox, oy), res in zip(crop_meta, results):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                for pbox, pconf in zip(res.boxes.xyxy.int().tolist(), res.boxes.conf.tolist()):
                    px1, py1, px2, py2 = pbox
                    phone_dets.append((px1 + ox, py1 + oy, px2 + ox, py2 + oy, pconf, cid))
    elif not confirmed_boxes:
        if phone_model is not None:
            res_full = phone_model.predict(frame, conf=PHONE_CONF, iou=PHONE_IOU, imgsz=1280,
                                            device=DEVICE, half=HALF, verbose=False)
        else:
            res_full = model_main.predict(frame, classes=[67], conf=PHONE_COCO_FALLBACK_CONF,
                                           iou=PHONE_COCO_FALLBACK_IOU,
                                           imgsz=1280, device=DEVICE, half=HALF, verbose=False)
        rf = res_full[0]
        if rf.boxes is not None and len(rf.boxes):
            for pbox, pconf in zip(rf.boxes.xyxy.int().tolist(), rf.boxes.conf.tolist()):
                x1, y1, x2, y2 = pbox
                phone_dets.append((x1, y1, x2, y2, pconf, None))

    for x1, y1, x2, y2, conf, owner_id in phone_dets:
        cursor.execute("""
            INSERT INTO events
                (timestamp, person_id, event_type, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (round(current_ts, 3), owner_id, "phone_detected",
              round(conf, 4), x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        plabel = f"PHONE{f' P{owner_id}' if owner_id else ''} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 200), -1)
        cv2.putText(frame, plabel, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # ── POSE PASS (sleeping detection) ────────────────────────────────────────
    run_pose_pass = (pose_model is not None) and confirmed_boxes \
        and (frame_count % POSE_CHECK_EVERY_N_FRAMES == 0)

    if run_pose_pass:
        crop_imgs, crop_meta = [], []
        for cid, box in confirmed_boxes.items():
            ex1, ey1, ex2, ey2 = expand_box(box, ROI_PADDING_RATIO, frame_width, frame_height)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            crop_imgs.append(crop)
            crop_meta.append((cid, box))

        if crop_imgs:
            pose_results = pose_model.predict(
                crop_imgs, conf=POSE_CONF, imgsz=POSE_CROP_IMGSZ,
                device=DEVICE, half=HALF, verbose=False
            )
            for (cid, obox), res in zip(crop_meta, pose_results):
                if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
                    continue
                kp_xy = res.keypoints.xy[0].cpu().numpy()
                kp_conf = (res.keypoints.conf[0].cpu().numpy()
                           if res.keypoints.conf is not None else np.ones(len(kp_xy)))
                label, pconf = classify_sleep_pose(kp_xy, kp_conf)
                if label is None:
                    continue

                x1, y1, x2, y2 = obox
                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2, vlm_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (round(current_ts, 3), cid, "pose_detected", round(float(pconf), 4),
                      x1, y1, x2, y2, label))

                if label == "SLEEPING":
                    cv2.putText(frame, "SLEEPING?", (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    run_face_pass = (face_model is not None) and (frame_count % FACE_DETECT_EVERY_N_FRAMES == 0)

    if run_face_pass:
        res_faces = face_model.predict(
            frame, conf=FACE_CONF, iou=FACE_IOU, imgsz=960,
            device=DEVICE, half=HALF, verbose=False
        )
        face_frame = res_faces[0]

        if face_frame.boxes is not None and len(face_frame.boxes):
            face_boxes = face_frame.boxes.xyxy.int().tolist()
            face_confs = face_frame.boxes.conf.tolist()

            for fbox, fconf in zip(face_boxes, face_confs):
                fx1, fy1, fx2, fy2 = fbox
                fx1 = max(0, fx1); fy1 = max(0, fy1)
                fx2 = min(frame_width - 1, fx2); fy2 = min(frame_height - 1, fy2)

                cid = face_inside_person(fbox, confirmed_boxes)

                if cid is None:
                    _face_check_stats['unlinked'] += 1
                elif fconf < FACE_EMB_MIN_CONF_FOR_CHECK:
                    _face_check_stats['low_conf'] += 1

                if FACE_SWAP_CHECK_ENABLED and cid is not None and fconf >= FACE_EMB_MIN_CONF_FOR_CHECK:
                    _face_check_stats['checked'] += 1
                    face_emb = compute_face_embedding(frame, (fx1, fy1, fx2, fy2))
                    if face_emb is not None:
                        own_sim = _face_gallery_best_sim(cid, face_emb)

                        if own_sim is None or own_sim >= FACE_EMB_OWN_MATCH_FLOOR:
                            _face_gallery_add(cid, face_emb)
                        else:
                            _face_check_stats['own_mismatch'] += 1
                            best_other_cid, best_other_sim = None, -1.0
                            for other_cid in confirmed_boxes.keys():
                                if other_cid == cid:
                                    continue
                                s = _face_gallery_best_sim(other_cid, face_emb)
                                if s is not None and s > best_other_sim:
                                    best_other_sim, best_other_cid = s, other_cid

                            if best_other_cid is not None and best_other_sim >= FACE_EMB_SWAP_CONFIRM_FLOOR \
                                    and best_other_sim > own_sim:
                                pair_key = frozenset((cid, best_other_cid))
                                _face_swap_evidence[pair_key] = _face_swap_evidence.get(pair_key, 0) + 1
                                evidence_count = _face_swap_evidence[pair_key]
                                on_cooldown = frame_count - _swap_cooldown.get(pair_key, -10**9) < SWAP_COOLDOWN_FRAMES

                                if evidence_count < FACE_SWAP_PERSISTENCE_THRESHOLD or on_cooldown:
                                    status = "on cooldown" if on_cooldown else "not yet enough"
                                    print(f"  [FACE MISMATCH] Person {cid}'s face matched Person "
                                          f"{best_other_cid}'s gallery better than its own "
                                          f"(own={own_sim:.2f}, other={best_other_sim:.2f}) — "
                                          f"evidence {evidence_count}/{FACE_SWAP_PERSISTENCE_THRESHOLD} "
                                          f"({status}, not correcting yet).")
                                else:
                                    raw_id_for_cid = next(
                                        (rid for rid, c in id_registry.items() if c == cid and rid in raw_ids), None
                                    )
                                    raw_id_for_other = next(
                                        (rid for rid, c in id_registry.items() if c == best_other_cid and rid in raw_ids), None
                                    )
                                    if raw_id_for_cid is not None and raw_id_for_other is not None:
                                        id_registry[raw_id_for_cid]   = best_other_cid
                                        id_registry[raw_id_for_other] = cid
                                        last_known_face_gallery[cid], last_known_face_gallery[best_other_cid] = \
                                            last_known_face_gallery.get(best_other_cid, []), last_known_face_gallery.get(cid, [])
                                        _face_gallery_add(best_other_cid, face_emb)
                                        _swap_cooldown[pair_key] = frame_count
                                        _face_swap_evidence.pop(pair_key, None)
                                        print(f"[SWAP CORRECTED] Face evidence shows Person {cid} and "
                                              f"Person {best_other_cid} were swapped by the tracker "
                                              f"(confirmed across {evidence_count} separate frames; "
                                              f"own-gallery similarity={own_sim:.2f}, matched-other "
                                              f"similarity={best_other_sim:.2f}) — corrected the raw-track "
                                              f"mapping going forward.")
                                        cid = best_other_cid

                crop_path = None
                if cid is not None and fconf > best_face_conf.get(cid, 0.0):
                    best_face_conf[cid] = fconf
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        crop_path = os.path.join(face_crops_dir, f"person_{cid}_face.jpg")
                        cv2.imwrite(crop_path, face_crop)
                        best_face_crop[cid] = crop_path

                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (round(current_ts, 3), cid, "face_detected",
                      round(fconf, 4), fx1, fy1, fx2, fy2, crop_path))

                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 100, 0), 2)
                flabel = f"FACE{f' P{cid}' if cid is not None else ''} {fconf:.2f}"
                (tw, th), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (fx1, fy1 - th - 8), (fx1 + tw + 6, fy1), (200, 80, 0), -1)
                cv2.putText(frame, flabel, (fx1 + 3, fy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    ZONE_COLOR_BGR = (241, 102, 99)
    ZONE_LINE_THICKNESS = 1
    for zi, zone_px in enumerate(_ZONES_PX):
        pts = np.array(zone_px, dtype=np.int32)
        cv2.polylines(frame, [pts], True, ZONE_COLOR_BGR, ZONE_LINE_THICKNESS, lineType=cv2.LINE_AA)
        zx = int(np.mean(pts[:, 0])) - 30
        zy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, f"Zone {zi}", (zx, zy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR_BGR, 1, lineType=cv2.LINE_AA)

    if frame_count % 30 == 0:
        conn.commit()
        print(f" -> Frame {frame_count:>5}/{total_frames} | "
              f"Active people: {sorted(active_people)} | "
              f"Total confirmed: {sorted(set(id_registry.values()))}")
        print(f"    [FACE CHECK STATS since start] unlinked={_face_check_stats['unlinked']} "
              f"low_conf={_face_check_stats['low_conf']} checked={_face_check_stats['checked']} "
              f"own_mismatch={_face_check_stats['own_mismatch']} "
              f"gallery_sizes={ {cid: len(g) for cid, g in last_known_face_gallery.items()} }")

    print(f"PROGRESS:{frame_count}:{total_frames}", flush=True)

    if frame_count % LIVE_FRAME_EVERY == 0:
        cv2.imwrite(LIVE_FRAME_PATH, frame)

    video_writer.write(frame)

def _average_embedding(cid):
    if id_emb_count.get(cid, 0) == 0:
        return None
    avg = id_emb_sum[cid] / id_emb_count[cid]
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else None

def _average_hist(cid):
    if id_hist_count.get(cid, 0) == 0:
        return None
    return id_hist_sum[cid] / id_hist_count[cid]

def _home_zone(cid):
    votes = id_zone_votes.get(cid)
    if not votes:
        return None
    return max(votes, key=votes.get)

def _appearance_score(id_a, id_b):
    emb_a, emb_b = _average_embedding(id_a), _average_embedding(id_b)
    if emb_a is not None and emb_b is not None:
        return float(np.dot(emb_a, emb_b)), True

    hist_a, hist_b = _average_hist(id_a), _average_hist(id_b)
    if hist_a is not None and hist_b is not None:
        score = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32),
                                 cv2.HISTCMP_CORREL)
        return float(score), False

    return None, False

all_confirmed_ids = sorted(set(id_registry.values()))
parent = {cid: cid for cid in all_confirmed_ids}

def _find_root(cid):
    while parent[cid] != cid:
        cid = parent[cid]
    return cid

merge_log = []
for i, id_a in enumerate(all_confirmed_ids):
    for id_b in all_confirmed_ids[i + 1:]:
        if _find_root(id_a) == _find_root(id_b):
            continue

        if id_frame_set.get(id_a, set()) & id_frame_set.get(id_b, set()):
            continue

        score, used_embedding = _appearance_score(id_a, id_b)
        if score is None:
            continue

        home_a, home_b = _home_zone(id_a), _home_zone(id_b)
        same_zone = home_a is not None and home_a == home_b

        if used_embedding:
            threshold = ZONE_MERGE_EMBEDDING_THRESHOLD if same_zone else MERGE_EMBEDDING_THRESHOLD
        else:
            threshold = ZONE_MERGE_APPEARANCE_THRESHOLD if same_zone else MERGE_APPEARANCE_THRESHOLD

        if score >= threshold:
            root_a, root_b = _find_root(id_a), _find_root(id_b)
            keep, drop = (root_a, root_b) if root_a < root_b else (root_b, root_a)
            parent[drop] = keep
            merge_log.append((drop, keep, score, used_embedding, same_zone,
                               home_a if same_zone else None))

if merge_log:
    print("\n[IDENTITY RECONCILIATION] Merging fragmented tracks:")
    for drop, keep, score, used_embedding, same_zone, zone_idx in merge_log:
        sig = "embedding" if used_embedding else "HSV histogram"
        zone_note = f", same home zone={zone_idx}" if same_zone else ""
        print(f"  Person {drop} -> Person {keep}  "
              f"({sig} similarity={score:.2f}, never co-present{zone_note})")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS identity_merges (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            dropped_track_id     INTEGER,
            canonical_person_id  INTEGER,
            appearance_score     REAL,
            used_embedding       INTEGER,
            same_home_zone       INTEGER,
            matching_zone_index  INTEGER,
            merged_at_timestamp  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for drop, keep, score, used_embedding, same_zone, zone_idx in merge_log:
        cursor.execute("""
            INSERT INTO identity_merges
                (dropped_track_id, canonical_person_id, appearance_score,
                 used_embedding, same_home_zone, matching_zone_index)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (drop, keep, round(float(score), 4), int(used_embedding),
              int(same_zone), zone_idx))

    for cid in all_confirmed_ids:
        root = _find_root(cid)
        if root != cid:
            cursor.execute("UPDATE events SET person_id = ? WHERE person_id = ?", (root, cid))
    conn.commit()
    final_count = len(set(_find_root(cid) for cid in all_confirmed_ids))
    print(f"[INFO] {len(all_confirmed_ids)} confirmed track(s) reconciled to "
          f"{final_count} unique people. Merge history logged in 'identity_merges' table.\n")
else:
    print(f"\n[INFO] Identity reconciliation found no mergeable tracks "
          f"({len(all_confirmed_ids)} confirmed track(s) stand as-is).\n")

conn.commit()
conn.close()
video_capture.release()
video_writer.release()

_final_people_count = len(set(_find_root(cid) for cid in all_confirmed_ids)) if all_confirmed_ids else 0

print("\n########################################")
print("  AGENT 1 COMPLETE: DATABASE POPULATED  ")
print("########################################")
print(f"Total frames processed : {frame_count}")
print(f"Confirmed tracks (pre-merge) : {next_clean_id - 1}")
print(f"Total unique people (post-reconciliation) : {_final_people_count}")
print(f"Database location      : {DATABASE_PATH}")
print(f"Output video           : {output_path}")
print(f"Face crops saved to    : {face_crops_dir}")
print()

if best_face_crop:
    print("Best face crops saved:")
    for cid, path in sorted(best_face_crop.items()):
        print(f"  Person {cid} -> {path}")
else:
    print("[INFO] No face crops saved (model not found or no faces detected).")