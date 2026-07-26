# config.py - Shared settings for all agents
import os

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH    = os.path.join(ROOT_DIR, "videos", "Chaos.mp4")
DATABASE_PATH = os.path.join(ROOT_DIR, "database", "Test.db")
CROPS_DIR     = os.path.join(ROOT_DIR, "crops")

FACE_MODEL_PATH  = os.path.join(ROOT_DIR, "face_detection_model.pt")
PHONE_MODEL_PATH = os.path.join(ROOT_DIR, "best_phone_nano.pt")

# ── Person Detector ────────────────────────────────────────────────────────────
MODEL_NAME = "yolo26n.pt"
# NOTE: YOLO26 (released Jan 2026) is a genuinely different, NMS-free
# architecture, not just a bigger/newer nano checkpoint - verified this is a
# real Ultralytics release, not a typo of yolo12n. Worth a quick empirical
# check for your writeup: since YOLO26 drops NMS internally, the `iou=`
# argument passed to model.track()/predict() may now be a no-op. Log box
# counts with a couple of different iou= values and confirm before citing it
# as a tuned parameter.

# --- Detection confidence (YOLO person detector) ---
# Minimum confidence for YOLO to keep a "person" box at all, BEFORE it even
# reaches BoT-SORT/probation/re-id. Kept low (0.15) deliberately so atypical
# poses (slumped/asleep, back turned, partial occlusion) aren't dropped before
# they ever get a box - MIN_FRAMES_TO_CONFIRM (probation) is what filters out
# the resulting stray false positives, not this threshold.
DETECTION_CONF_THRESHOLD = 0.10

# ── Persistent Re-ID: OSNet -> 512-D embedding -> FAISS -> Persistent Person ID
REID_MODEL_NAME       = "osnet_ain_x1_0"
EMBEDDING_DIM         = 512
REID_MATCH_THRESHOLD  = 0.55   # cosine similarity needed to accept a gallery match
MIN_CROP_SIZE         = 20     # ignore crops smaller than this (px) before embedding

# CRITICAL: point this at a REAL re-id-trained OSNet checkpoint on YOUR
# machine. Left empty on purpose - the path in the version you got from your
# friend was hardcoded to their own PC and will not exist on yours. If this
# stays "", PersonEmbedder falls back to generic ImageNet weights (NOT
# trained to distinguish people) and prints a loud warning + burns an orange
# banner into the output video so you can't miss it. Download the checkpoint
# from https://huggingface.co/kaiyangzhou/osnet and set the path here before
# trusting any Re-ID accuracy numbers for the writeup.
REID_WEIGHTS_PATH = "C:\\Users\\Admin\\Documents\\GitHub\\smart-surveillance-system\\weights\\osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"

# Hybrid match scoring weights (visual is the floor; spatial/temporal only
# ever add confidence on top, never subtract) - must sum to 1.0.
HYBRID_MATCH_WEIGHTS = {
    "visual": 0.6,
    "spatial": 0.25,
    "temporal": 0.15,
}

TEMPORAL_MEMORY_LENGTH   = 30   # recent (frame, bbox, embedding) samples kept per person
RECHECK_INTERVAL_FRAMES  = 30   # re-embed an already-matched track every N frames (~1s @ 30fps)
MIN_FRAMES_TO_CONFIRM    = 10  # probation: consecutive hits needed before a track is trusted
MAX_FRAMES_OCCLUDED      = 90   # frames to keep predicting an occluded track's position (~3s @ 30fps)
OCCLUSION_SEARCH_SCALE   = 1.5  # how much to expand the search region around an occluded bbox

# ── Phone Detection (own model, separate pass - not part of person tracking) ──
PHONE_CONF = 0.5
PHONE_IOU  = 0.30
# Only used if best_phone.pt isn't found at PHONE_MODEL_PATH - falls back to
# COCO class 67 ("cell phone") on the main yolo26n person model instead.
PHONE_COCO_FALLBACK_CONF = 0.40
PHONE_COCO_FALLBACK_IOU  = 0.30
PHONE_DETECT_EVERY_N_FRAMES = 1   # phone episodes are brief - check every frame

# Phone-to-face matching thresholds for Agent 2.
PHONE_MIN_PERSON_OVERLAP = 0.3
PHONE_FACE_MIN_SCORE = 0.35
PHONE_FACE_MAX_CENTER_DIST_RATIO = 1.4
PHONE_FACE_RECENCY_SECONDS = 1.0
PHONE_MIN_CONSECUTIVE_SECONDS = 4.0
PHONE_GAP_SECONDS = 1.0

# ── Face Detection (own model, scoped to each tracked person's box) ──────────
FACE_CONF = 0.50
FACE_IOU  = 0.40
FACE_CROP_PAD_RATIO = 0.15   # padding around a person's bbox before running the face model on it
# Throttled rather than every frame - matches the earlier fix for redundant
# face detection eating latency for no accuracy gain.
FACE_DETECT_EVERY_N_FRAMES = 1

# --- Pose estimation / sleeping detection ---
POSE_MODEL_PATH = "yolo26n-pose.pt"   # stock Ultralytics checkpoint, auto-downloads - no ROOT_DIR path needed
POSE_CONF = 0.5
POSE_DETECT_EVERY_N_FRAMES = 1

SLEEP_HEAD_DROP_RATIO = 0.15
SLEEP_FACE_VISIBILITY_THRESHOLD = 0.95   # max confidence across nose/eyes/ears below this = face effectively not visible
SLEEP_FACE_RECENCY_FRAMES = 60           # if the face pass found a real face within this many frames, don't call it sleeping
SLEEP_MIN_CONSECUTIVE_FRAMES = 10        # must hold for this many checked frames in a row before it's trusted
SLEEP_GRACE_FRAMES = 6                   # one ambiguous check doesn't reset an ongoing episode

# Agent 2 uses a duration gate on top of the raw sleeping detections.
SLEEP_MIN_DURATION_SECONDS = 8.0
SLEEP_GAP_SECONDS = 1.0

