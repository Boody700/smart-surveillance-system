# agent1_tracking/agent1.py
# Agent 1: Full production pipeline — SPLIT DETECTION + FACE DETECTION
#
# THREE separate inference passes per frame:
#   1. PERSON pass   — class 0 only, higher conf, full tracking
#   2. PHONE  pass   — class 67 only, lower conf, dedicated focus
#   3. FACE   pass   — face_detection_model.pt, no tracking, just detection + crop
#
# Position-based Re-ID is preserved from the original.
# Face detections are linked to the nearest confirmed person by bbox overlap.

import os
import sys
import cv2
import json
import sqlite3
import numpy as np
from ultralytics import YOLO
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH as DEFAULT_VIDEO_PATH, MODEL_NAME, DATABASE_PATH

FACE_MODEL_PATH  = os.path.join(ROOT_DIR, "face_detection_model.pt")
PHONE_MODEL_PATH = os.path.join(ROOT_DIR, "best_phone.pt")

# ── ZONE DATA (for zone-anchored identity reconciliation) ─────────────────────
# Loaded here too (not just in Agent 2) so the offline reconciliation pass can
# use "did this person come back to the SAME physical desk" as independent
# evidence on top of appearance — for a fixed-workstation setup, that's a much
# stronger signal than clothing similarity alone.
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

# Video path can be overridden from the command line (e.g. by app.py passing
# the uploaded file's path). Falls back to config.py's default otherwise.
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

print("\n=== AGENT 1 STARTING: SPLIT DETECTION + FACE ===")
print(f"[INFO] Video source: {VIDEO_PATH}"
      f"{'  (overridden via argv)' if len(sys.argv) > 1 else '  (from config.py default)'}")

# ── DATABASE ──────────────────────────────────────────────────────────────────
conn   = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("DELETE FROM events")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
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

# Zone polygons in pixel space, precomputed once — drawn on every frame below
# (live preview + saved output video), so you can see calibrated zones during
# processing, not just after the fact.
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
output_path = os.path.join(output_dir, "agent1_output3.mp4")
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ── LIVE PREVIEW FOR THE UI ───────────────────────────────────────────────────
# Same contract as before: Streamlit polls this JPEG + reads PROGRESS lines
# from stdout to drive a progress bar and a live "as it processes" preview.
LIVE_FRAME_PATH  = os.path.join(ROOT_DIR, "live_frame.jpg")
LIVE_FRAME_EVERY = 3  # write a preview frame every N frames

# ── FACE CROPS DIR ────────────────────────────────────────────────────────────
face_crops_dir = os.path.join(ROOT_DIR, "face_crops")
os.makedirs(face_crops_dir, exist_ok=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
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

# ── TRACKING STATE ────────────────────────────────────────────────────────────
raw_id_counters  = {}   # raw_id  -> frame count survived
id_registry      = {}   # raw_id  -> clean sequential ID
next_clean_id    = 1

last_seen_frame  = {}   # raw_id  -> last frame it appeared
position_history = {}   # raw_id  -> [(cx,cy), ...] during probation
last_known_pos   = {}   # clean_id -> (cx, cy)  — used for Re-ID + face linking
last_known_hist  = {}   # clean_id -> HSV clothing histogram — appearance Re-ID fallback

# Whole-track accumulators for the offline identity-reconciliation pass
# (run once, after tracking finishes — see bottom of file).
id_hist_sum   = {}   # clean_id -> running sum of HSV histograms across the full track
id_hist_count = {}   # clean_id -> number of histograms summed
id_frame_set  = {}   # clean_id -> set of frame_counts where this ID was seen
id_zone_votes = {}   # clean_id -> {zone_index: frame_count} — for home-zone detection
id_first_seen = {}   # clean_id -> frame_count when first confirmed — home-zone
                       # votes only count within a warmup window after this,
                       # so a LATER unauthorized-zone violation can never
                       # corrupt where someone actually started/belongs.
HOME_ZONE_WARMUP_SECONDS = 20  # matches Agent 2's WARMUP_SECONDS concept
HOME_ZONE_WARMUP_FRAMES  = int(fps * HOME_ZONE_WARMUP_SECONDS) if fps else 600

GRACE_PERIOD_FRAMES   = 45
MIN_FRAMES_TO_CONFIRM = 50
REID_DISTANCE_THRESH  = 250   # px — raise if Re-ID misses, lower if it merges people

# Motion-validity gate — filters out static false-positive "people" (furniture,
# shadows, decor) that would otherwise graduate into a real, phantom ID.
MOTION_VALIDITY_ENABLED   = True
MIN_MOVEMENT_SPREAD       = 3     # px total (x-range + y-range) over probation
MOTION_CHECK_GRACE_FRAMES = 150   # safety valve — graduate anyway past this many
                                   # frames even without enough spread

# ── APPEARANCE RE-ID (SECONDARY, TOGGLEABLE) ──────────────────────────────────
# Rollback flag: set False to fall back to the exact original position-only
# Re-ID logic. When True, this ONLY kicks in as a fallback after the tight
# position match below has already failed to find anyone — it never
# overrides or changes a match the original position-only logic would make.
# Pure OpenCV HSV histogram of the torso region — no new dependencies,
# no deep model, no facial data.
APPEARANCE_REID_ENABLED  = True
APPEARANCE_SEARCH_RADIUS = 650   # px — wider net; appearance disambiguates candidates
APPEARANCE_MATCH_FLOOR   = 0.55  # min HSV histogram correlation to accept a match
HIST_BINS                = 32

# Per-person face crop — save one good crop per person (highest conf face seen)
best_face_conf  = {}    # clean_id -> best conf seen so far
best_face_crop  = {}    # clean_id -> path to saved crop

frame_count = 0

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def find_matching_clean_id(avg_cx, avg_cy, appearance_hist=None):
    # --- ORIGINAL LOGIC — completely unchanged, always tried first ---
    best_cid  = None
    best_dist = float("inf")
    for cid, (lx, ly) in last_known_pos.items():
        dist = np.sqrt((avg_cx - lx) ** 2 + (avg_cy - ly) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_cid  = cid
    if best_dist <= REID_DISTANCE_THRESH:
        return best_cid   # tight position match — exact original behavior

    # --- NEW: appearance-gated fallback, only reached if the above found nothing ---
    if APPEARANCE_REID_ENABLED and appearance_hist is not None:
        best_app_cid, best_app_score = None, -1.0
        for cid, (lx, ly) in last_known_pos.items():
            dist = np.sqrt((avg_cx - lx) ** 2 + (avg_cy - ly) ** 2)
            if dist > APPEARANCE_SEARCH_RADIUS:
                continue
            hist = last_known_hist.get(cid)
            if hist is None:
                continue
            score = cv2.compareHist(appearance_hist, hist, cv2.HISTCMP_CORREL)
            if score >= APPEARANCE_MATCH_FLOOR and score > best_app_score:
                best_app_score = score
                best_app_cid   = cid
        if best_app_cid is not None:
            print(f"  [APPEARANCE RE-ID] Matched via clothing histogram "
                  f"(correlation={best_app_score:.2f}, position-only would have missed this)")
            return best_app_cid

    return None

def compute_appearance_hist(frame, box):
    """
    HSV color histogram of the torso region of a person crop.
    Skips the top ~25% (head — avoids relying on face) and bottom ~15%
    (feet/floor bleed), keeping the clothing-dominated middle band.
    Pure OpenCV — no new dependency, no deep model, no facial data.
    """
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
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def iou(boxA, boxB):
    """Intersection over union — used to match face bbox to person bbox."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / float(areaA + areaB - inter)

def face_inside_person(face_box, person_boxes_map):
    """
    Given a face bbox, find which confirmed person it belongs to.
    Strategy: face centre must lie inside the person bbox,
    OR fall back to closest person centre.
    Returns clean_id or None.
    """
    fx1, fy1, fx2, fy2 = face_box
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

    # Check containment first (face centre inside person box)
    for cid, (px1, py1, px2, py2) in person_boxes_map.items():
        if px1 <= fcx <= px2 and py1 <= fcy <= py2:
            return cid

    # Fallback: closest person centre within 300px
    best_cid  = None
    best_dist = float("inf")
    for cid, (px1, py1, px2, py2) in person_boxes_map.items():
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        dist = np.sqrt((fcx - pcx) ** 2 + (fcy - pcy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_cid  = cid

    return best_cid if best_dist < 300 else None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count    += 1
    current_ts      = frame_count / fps
    active_people   = set()

    # ── PASS 1: PERSON TRACKING ──────────────────────────────────────────────
    res_people = model_main.track(
        frame,
        persist=True,
        tracker="agent1_tracking/custom_tracker2.yaml",
        classes=[0],            # people only
        conf=0.25,              # slightly higher — fewer false positives
        iou=0.30,
        imgsz=1280,             # bumped from 960 — people at desk-distance/partly
                                 # occluded need the extra resolution; this pass
                                 # doesn't do tracking-match dedup so the speed
                                 # cost is worth it. Drop back to 960 if CPU-bound.
        verbose=False
    )

    person_frame    = res_people[0]
    confirmed_boxes = {}    # clean_id -> (x1,y1,x2,y2) for face linking this frame

    if person_frame.boxes.id is not None:
        raw_ids     = person_frame.boxes.id.int().tolist()
        boxes       = person_frame.boxes.xyxy.int().tolist()
        confidences = person_frame.boxes.conf.tolist()

        # PHASE 1a: probation + graduation
        for raw_id, box in zip(raw_ids, boxes):
            cx, cy = get_center(box)

            if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            else:
                raw_id_counters[raw_id] = 1
                position_history[raw_id] = []

            last_seen_frame[raw_id] = frame_count

            if raw_id not in id_registry:
                position_history.setdefault(raw_id, []).append((cx, cy))

            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                positions = position_history.get(raw_id, [(cx, cy)])

                # ── MOTION VALIDITY GATE ──────────────────────────────────────
                # Rollback flag: set False to graduate purely on frame-count,
                # exactly as before. When True, a track must show at least
                # MIN_MOVEMENT_SPREAD px of total position spread across its
                # whole probation window before being trusted as a real
                # person. Real people — even sitting still — show a few
                # pixels of natural detector jitter frame to frame; a box
                # frozen tighter than that for 50+ straight frames is a much
                # stronger signal of a static misdetection (furniture, a
                # shadow, decor) than of a live person.
                #
                # Safety valve: if a track still hasn't shown that much
                # spread after MOTION_CHECK_GRACE_FRAMES, graduate it anyway
                # (flagged) — so a genuinely near-motionless real person is
                # never silently ignored forever, just double-checked.
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                spread = (max(xs) - min(xs)) + (max(ys) - min(ys))

                if MOTION_VALIDITY_ENABLED and spread < MIN_MOVEMENT_SPREAD \
                        and raw_id_counters[raw_id] < MOTION_CHECK_GRACE_FRAMES:
                    continue  # not enough movement evidence yet — keep watching

                if MOTION_VALIDITY_ENABLED and spread < MIN_MOVEMENT_SPREAD:
                    print(f"[CAUTION] raw_id={raw_id} graduating after "
                          f"{raw_id_counters[raw_id]} frames with near-zero movement "
                          f"(spread={spread}px) — verify this is a real person and "
                          f"not a static false detection.")

                avg_cx    = int(np.mean(xs))
                avg_cy    = int(np.mean(ys))
                grad_hist = compute_appearance_hist(frame, box)
                matched   = find_matching_clean_id(avg_cx, avg_cy, grad_hist)
                if matched is not None:
                    id_registry[raw_id] = matched
                    print(f"[RE-ENTRY] Person {matched} returned! (raw_id={raw_id})")
                else:
                    id_registry[raw_id] = next_clean_id
                    next_clean_id += 1
                    print(f"[NEW PERSON] Person ID {id_registry[raw_id]} confirmed.")

        # PHASE 1b: update positions + appearance
        for raw_id, box in zip(raw_ids, boxes):
            if raw_id in id_registry:
                clean_id = id_registry[raw_id]
                last_known_pos[clean_id] = get_center(box)
                hist = compute_appearance_hist(frame, box)
                if hist is not None:
                    last_known_hist[clean_id] = hist

        # PHASE 1c: write to DB + draw
        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id not in id_registry:
                continue

            clean_id = id_registry[raw_id]
            active_people.add(clean_id)
            x1, y1, x2, y2 = box
            confirmed_boxes[clean_id] = (x1, y1, x2, y2)

            # Accumulate whole-track appearance + presence data for the
            # offline identity-reconciliation pass at the end of the run.
            track_hist = compute_appearance_hist(frame, box)
            if track_hist is not None:
                if clean_id not in id_hist_sum:
                    id_hist_sum[clean_id] = track_hist.copy()
                else:
                    id_hist_sum[clean_id] += track_hist
                id_hist_count[clean_id] = id_hist_count.get(clean_id, 0) + 1
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

            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{clean_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── PASS 2: PHONE DETECTION ───────────────────────────────────────────────
    # Uses your custom-trained single-class phone model if available,
    # otherwise falls back to COCO class 67 on the main model.
    if phone_model is not None:
        res_phones = phone_model.predict(
            frame,
            conf=0.40,           # custom model — start moderate, tune after testing
            iou=0.30,
            imgsz=1280,           # phones are small objects — resolution is the
                                  # single biggest lever here, worth the extra
                                  # cost since this pass has no tracking overhead
            verbose=False
        )
    else:
        res_phones = model_main.predict(
            frame,
            classes=[67],
            conf=0.40,
            iou=0.30,
            imgsz=1280,
            verbose=False
        )

    phone_frame = res_phones[0]

    if phone_frame.boxes is not None and len(phone_frame.boxes):
        ph_boxes = phone_frame.boxes.xyxy.int().tolist()
        ph_confs = phone_frame.boxes.conf.tolist()

        for box, conf in zip(ph_boxes, ph_confs):
            x1, y1, x2, y2 = box

            # Try to assign phone to nearest confirmed person
            owner_id = None
            if confirmed_boxes:
                fcx, fcy = (x1 + x2) / 2, (y1 + y2) / 2
                best_dist = float("inf")
                for cid, (px1, py1, px2, py2) in confirmed_boxes.items():
                    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                    dist = np.sqrt((fcx - pcx) ** 2 + (fcy - pcy) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        owner_id  = cid
                if best_dist > 400:   # too far — don't assign
                    owner_id = None

            cursor.execute("""
                INSERT INTO events
                    (timestamp, person_id, event_type, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (round(current_ts, 3), owner_id, "phone_detected",
                  round(conf, 4), x1, y1, x2, y2))

            # Draw phone box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            plabel = f"PHONE{f' P{owner_id}' if owner_id else ''} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 200, 200), -1)
            cv2.putText(frame, plabel, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # ── PASS 3: FACE DETECTION ────────────────────────────────────────────────
    if face_model is not None:
        res_faces = face_model.predict(
            frame,
            conf=0.50,          # faces — keep reasonable threshold
            iou=0.40,
            imgsz=960,
            verbose=False
        )

        face_frame = res_faces[0]

        if face_frame.boxes is not None and len(face_frame.boxes):
            face_boxes = face_frame.boxes.xyxy.int().tolist()
            face_confs = face_frame.boxes.conf.tolist()

            for fbox, fconf in zip(face_boxes, face_confs):
                fx1, fy1, fx2, fy2 = fbox

                # Clamp to frame bounds
                fx1 = max(0, fx1); fy1 = max(0, fy1)
                fx2 = min(frame_width - 1, fx2)
                fy2 = min(frame_height - 1, fy2)

                # Link face to a confirmed person
                owner_id = face_inside_person(fbox, confirmed_boxes)

                # Save best face crop per person (highest conf wins)
                crop_path = None
                if owner_id is not None:
                    if fconf > best_face_conf.get(owner_id, 0.0):
                        best_face_conf[owner_id] = fconf
                        crop = frame[fy1:fy2, fx1:fx2]
                        if crop.size > 0:
                            crop_path = os.path.join(
                                face_crops_dir,
                                f"person_{owner_id}_face.jpg"
                            )
                            cv2.imwrite(crop_path, crop)
                            best_face_crop[owner_id] = crop_path

                # Log face detection
                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (round(current_ts, 3), owner_id, "face_detected",
                      round(fconf, 4), fx1, fy1, fx2, fy2, crop_path))

                # Draw face box
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 100, 0), 2)
                flabel = f"FACE{f' P{owner_id}' if owner_id else ''} {fconf:.2f}"
                (tw, th), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (fx1, fy1 - th - 8), (fx1 + tw + 6, fy1), (200, 80, 0), -1)
                cv2.putText(frame, flabel, (fx1 + 3, fy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── DRAW CALIBRATED ZONES ─────────────────────────────────────────────────
    # Drawn last, on top of the person/phone/face boxes, so you can see zone
    # boundaries during processing itself — not just after the fact in Agent 2.
    for zi, zone_px in enumerate(_ZONES_PX):
        pts = np.array(zone_px, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (99, 102, 241), 2)  # indigo — matches the app's zone color
        zx = int(np.mean(pts[:, 0])) - 30
        zy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, f"Zone {zi}", (zx, zy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99, 102, 241), 2)

    # ── PROGRESS LOG ─────────────────────────────────────────────────────────
    if frame_count % 30 == 0:
        conn.commit()
        print(f" -> Frame {frame_count:>5}/{total_frames} | "
              f"Active people: {sorted(active_people)} | "
              f"Total confirmed: {sorted(set(id_registry.values()))}")

    # Machine-readable progress line for the Streamlit UI (every frame, cheap)
    print(f"PROGRESS:{frame_count}:{total_frames}", flush=True)

    # Live annotated-frame snapshot for the UI's "watch it process" preview
    if frame_count % LIVE_FRAME_EVERY == 0:
        cv2.imwrite(LIVE_FRAME_PATH, frame)

    video_writer.write(frame)

# ── PASS 4 (OFFLINE): IDENTITY RECONCILIATION ─────────────────────────────────
# Online tracking (above) necessarily over-counts when someone re-enters in a
# way the live position/appearance checks can't catch — it has to commit to a
# decision immediately, using only past data. Now that the whole video has
# been processed, we can do something an online tracker structurally can't:
# compare EVERY pair of confirmed IDs across their ENTIRE track and ask
# whether they were actually the same person all along.
#
# A merge only happens if BOTH hold:
#   1. HARD CONSTRAINT — the two IDs were never seen in the same frame.
#      If they ever co-occurred, they are provably different people, no
#      matter how similar they look. This is what keeps the merge honest —
#      it can't collapse two people who were genuinely both present.
#   2. Their FULL-TRACK averaged appearance histograms correlate above
#      MERGE_APPEARANCE_THRESHOLD (stricter than the online fallback, since
#      this decision is harder to reverse and rewrites logged history).
#
# This does not cap or assume a known number of people — it will correctly
# leave distinct people unmerged, and will correctly merge fragments of the
# same person however many tracks they were split into.
MERGE_APPEARANCE_THRESHOLD      = 0.65   # default bar — appearance alone
ZONE_MERGE_APPEARANCE_THRESHOLD = 0.40   # looser bar — used ONLY when both
                                          # candidates share a home zone.
                                          # "Same clothing AND same physical
                                          # desk" needs far less appearance
                                          # certainty than clothing alone —
                                          # this is what makes a person
                                          # returning to their own desk, after
                                          # any absence length, reliably stick
                                          # to one ID even if their appearance
                                          # shifted (lighting, pose, partial
                                          # occlusion) enough to miss the
                                          # stricter default bar.

def _average_hist(cid):
    if id_hist_count.get(cid, 0) == 0:
        return None
    return id_hist_sum[cid] / id_hist_count[cid]

def _home_zone(cid):
    """Zone this ID spent the most time in, across its whole track. None if
    it was never seen inside any calibrated zone."""
    votes = id_zone_votes.get(cid)
    if not votes:
        return None
    return max(votes, key=votes.get)

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
            continue  # already merged via a chain

        # Hard constraint — never co-present
        if id_frame_set.get(id_a, set()) & id_frame_set.get(id_b, set()):
            continue

        hist_a, hist_b = _average_hist(id_a), _average_hist(id_b)
        if hist_a is None or hist_b is None:
            continue

        score = cv2.compareHist(
            hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL
        )

        home_a, home_b = _home_zone(id_a), _home_zone(id_b)
        same_zone = home_a is not None and home_a == home_b
        threshold = ZONE_MERGE_APPEARANCE_THRESHOLD if same_zone else MERGE_APPEARANCE_THRESHOLD

        if score >= threshold:
            root_a, root_b = _find_root(id_a), _find_root(id_b)
            keep, drop = (root_a, root_b) if root_a < root_b else (root_b, root_a)
            parent[drop] = keep
            merge_log.append((drop, keep, score, same_zone, home_a if same_zone else None))

if merge_log:
    print("\n[IDENTITY RECONCILIATION] Merging fragmented tracks:")
    for drop, keep, score, same_zone, zone_idx in merge_log:
        zone_note = f", same home zone={zone_idx}" if same_zone else ""
        print(f"  Person {drop} -> Person {keep}  "
              f"(full-track appearance correlation={score:.2f}, never co-present{zone_note})")

    # Durable audit trail — records WHAT got merged into WHAT and WHY, so the
    # correction is provable rather than a silent overwrite. Query this table
    # any time to show exactly which raw tracks were reconciled.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS identity_merges (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            dropped_track_id     INTEGER,
            canonical_person_id  INTEGER,
            appearance_score     REAL,
            same_home_zone       INTEGER,
            matching_zone_index  INTEGER,
            merged_at_timestamp  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for drop, keep, score, same_zone, zone_idx in merge_log:
        cursor.execute("""
            INSERT INTO identity_merges
                (dropped_track_id, canonical_person_id, appearance_score,
                 same_home_zone, matching_zone_index)
            VALUES (?, ?, ?, ?, ?)
        """, (drop, keep, round(float(score), 4), int(same_zone), zone_idx))

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

# ── CLEANUP ───────────────────────────────────────────────────────────────────
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

# Print face crop summary
if best_face_crop:
    print("Best face crops saved:")
    for cid, path in sorted(best_face_crop.items()):
        print(f"  Person {cid} -> {path}")
else:
    print("[INFO] No face crops saved (model not found or no faces detected).")