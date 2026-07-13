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
import sqlite3
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH as DEFAULT_VIDEO_PATH, MODEL_NAME, DATABASE_PATH

FACE_MODEL_PATH  = os.path.join(ROOT_DIR, "face_detection_model.pt")
PHONE_MODEL_PATH = os.path.join(ROOT_DIR, "Phone_best.pt")

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

# ── OUTPUT VIDEO ──────────────────────────────────────────────────────────────
output_dir  = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "agent1_output.mp4")
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

GRACE_PERIOD_FRAMES   = 45
MIN_FRAMES_TO_CONFIRM = 50
REID_DISTANCE_THRESH  = 250   # px — raise if Re-ID misses, lower if it merges people

# Per-person face crop — save one good crop per person (highest conf face seen)
best_face_conf  = {}    # clean_id -> best conf seen so far
best_face_crop  = {}    # clean_id -> path to saved crop

frame_count = 0

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def find_matching_clean_id(avg_cx, avg_cy):
    best_cid  = None
    best_dist = float("inf")
    for cid, (lx, ly) in last_known_pos.items():
        dist = np.sqrt((avg_cx - lx) ** 2 + (avg_cy - ly) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_cid  = cid
    return best_cid if best_dist <= REID_DISTANCE_THRESH else None

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
        tracker="agent1_tracking/custom_tracker.yaml",
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
                avg_cx    = int(np.mean([p[0] for p in positions]))
                avg_cy    = int(np.mean([p[1] for p in positions]))
                matched   = find_matching_clean_id(avg_cx, avg_cy)
                if matched is not None:
                    id_registry[raw_id] = matched
                    print(f"[RE-ENTRY] Person {matched} returned! (raw_id={raw_id})")
                else:
                    id_registry[raw_id] = next_clean_id
                    next_clean_id += 1
                    print(f"[NEW PERSON] Person ID {id_registry[raw_id]} confirmed.")

        # PHASE 1b: update positions
        for raw_id, box in zip(raw_ids, boxes):
            if raw_id in id_registry:
                last_known_pos[id_registry[raw_id]] = get_center(box)

        # PHASE 1c: write to DB + draw
        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id not in id_registry:
                continue

            clean_id = id_registry[raw_id]
            active_people.add(clean_id)
            x1, y1, x2, y2 = box
            confirmed_boxes[clean_id] = (x1, y1, x2, y2)

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
            conf=0.25,           # custom model — start moderate, tune after testing
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
            conf=0.12,
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
            conf=0.40,          # faces — keep reasonable threshold
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

# ── CLEANUP ───────────────────────────────────────────────────────────────────
conn.commit()
conn.close()
video_capture.release()
video_writer.release()

print("\n########################################")
print("  AGENT 1 COMPLETE: DATABASE POPULATED  ")
print("########################################")
print(f"Total frames processed : {frame_count}")
print(f"Total unique people    : {next_clean_id - 1}")
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