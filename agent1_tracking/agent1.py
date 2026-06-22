# agent1_tracking/agent1.py

# Agent 1: Full production pipeline.
# Tracks people AND phones using YOLO + ByteTrack, writes every detection to sentinel.db.
# EMA smoothing on bounding boxes to reduce jitter.
# ByteTrack handles re-entry via track_buffer.
# Every time this runs, it clears old data and starts a fresh analysis.

import os
import sys
import cv2
import sqlite3
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH, MODEL_NAME, DATABASE_PATH

print("\n=== AGENT 1 STARTING: TRACKING + DATABASE FEED ===")

# --- CONNECT TO THE SHARED DATABASE ---
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("DELETE FROM events")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
conn.commit()
print(f"[INFO] Connected to database: {DATABASE_PATH}")
print(f"[INFO] Previous events cleared. Starting fresh analysis.")

# --- OPEN VIDEO ---
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit()

fps          = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video: {total_frames} frames at {fps:.2f} FPS")
print(f"[INFO] Resolution: {frame_width}x{frame_height}")
print(f"[INFO] Detecting: people (class 0) + phones (class 67)")
print(f"[INFO] -----------------------------------------------")

# --- SET UP OUTPUT VIDEO ---
output_dir   = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path  = os.path.join(output_dir, "Test2.mp4")
fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- TRACKING REGISTRIES ---
raw_id_counters = {}   # probation frame counts
last_seen_frame = {}   # grace period tracking
id_registry     = {}   # raw YOLO ID -> clean sequential ID
next_clean_id   = 1

# EMA smoothed boxes: raw_id -> [x1, y1, x2, y2] as floats
smoothed_boxes  = {}
EMA_ALPHA       = 0.7   # weight for previous box — higher = smoother but slower to update

MIN_FRAMES_TO_CONFIRM = 35
GRACE_PERIOD_FRAMES   = 45
MIN_WRITE_CONF        = 0.25

frame_count = 0
raw_ids     = []

def ema_smooth(raw_id, new_box):
    """
    Apply Exponential Moving Average to bbox coordinates.
    Returns smoothed [x1, y1, x2, y2] as ints.
    First time we see this raw_id, initialize with the raw box.
    """
    new_box_f = [float(v) for v in new_box]
    if raw_id not in smoothed_boxes:
        smoothed_boxes[raw_id] = new_box_f
    else:
        prev = smoothed_boxes[raw_id]
        smoothed_boxes[raw_id] = [
            EMA_ALPHA * prev[i] + (1 - EMA_ALPHA) * new_box_f[i]
            for i in range(4)
        ]
    return [int(v) for v in smoothed_boxes[raw_id]]

# --- MAIN PROCESSING LOOP ---
while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count += 1
    current_timestamp = frame_count / fps

    results = model.track(
        frame,
        persist=True,
        tracker="agent1_tracking/custom_tracker.yaml",
        classes=[0, 67],   # 0 = person, 67 = cell phone
        iou=0.20,
        imgsz=1536,
        conf=0.20,
        verbose=False
    )

    frame_results = results[0]

    # --- PHONE DETECTIONS (no tracking needed, just log presence) ---
    # Phones don't get IDs — we just store their bbox so Agent 2
    # can check overlap with person bboxes to flag phone usage
    if frame_results.boxes is not None:
        all_classes = frame_results.boxes.cls.int().tolist() if frame_results.boxes.cls is not None else []
        all_boxes   = frame_results.boxes.xyxy.int().tolist() if frame_results.boxes.xyxy is not None else []
        all_confs   = frame_results.boxes.conf.tolist() if frame_results.boxes.conf is not None else []

        for cls, box, conf in zip(all_classes, all_boxes, all_confs):
            if cls == 67 and conf >= MIN_WRITE_CONF:
                x1, y1, x2, y2 = box
                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round(current_timestamp, 3),
                    None,           # no person_id for phones
                    "phone_detected",
                    round(conf, 4),
                    x1, y1, x2, y2
                ))
                # Draw phone box in blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
                cv2.putText(frame, "PHONE", (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    # --- PERSON TRACKING ---
    if frame_results.boxes.id is not None:
        # Filter to only person detections (class 0) for tracking
        all_ids     = frame_results.boxes.id.int().tolist()
        all_classes = frame_results.boxes.cls.int().tolist()
        all_boxes_r = frame_results.boxes.xyxy.int().tolist()
        all_confs_r = frame_results.boxes.conf.tolist()

        # Zip and filter to people only
        person_data = [
            (rid, box, conf)
            for rid, cls, box, conf in zip(all_ids, all_classes, all_boxes_r, all_confs_r)
            if cls == 0
        ]

        raw_ids     = [d[0] for d in person_data]
        boxes       = [d[1] for d in person_data]
        confidences = [d[2] for d in person_data]

        # --- PHASE 1: PROBATION COUNTERS WITH GRACE PERIOD ---
        for raw_id in raw_ids:
            if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            else:
                raw_id_counters[raw_id] = 1

            last_seen_frame[raw_id] = frame_count

            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                id_registry[raw_id] = next_clean_id
                next_clean_id += 1
                print(f"\n[NEW PERSON] *** Person {id_registry[raw_id]} confirmed! ***\n")

        # --- PHASE 2: WRITE TO DATABASE + DRAW ON VIDEO ---
        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id in id_registry and conf >= MIN_WRITE_CONF:
                clean_id     = id_registry[raw_id]
                smooth_box   = ema_smooth(raw_id, box)
                x1, y1, x2, y2 = smooth_box

                # Clamp to frame bounds
                x1 = max(0, min(x1, frame_width))
                x2 = max(0, min(x2, frame_width))
                y1 = max(0, min(y1, frame_height))
                y2 = max(0, min(y2, frame_height))

                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round(current_timestamp, 3),
                    clean_id,
                    "detected",
                    round(conf, 4),
                    x1, y1, x2, y2
                ))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"ID:{clean_id}"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - txt_h - 6), (x1 + txt_w + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # --- PROGRESS LOG EVERY 30 FRAMES ---
    if frame_count % 30 == 0:
        conn.commit()

        active_this_frame = sorted(set([
            id_registry[rid] for rid in raw_ids
            if rid in id_registry
        ]))

        progress = (frame_count / total_frames) * 100
        print(f"[{progress:5.1f}%] Frame {frame_count}/{total_frames} | "
              f"On screen now: {active_this_frame} | "
              f"All confirmed: {sorted(set(id_registry.values()))}")

    video_writer.write(frame)

# --- FINAL COMMIT AND CLEANUP ---
conn.commit()
conn.close()
video_capture.release()
video_writer.release()

print("\n########################################")
print("  AGENT 1 COMPLETE: DATABASE POPULATED  ")
print("########################################")
print(f"Total frames processed : {frame_count}")
print(f"Total unique people    : {next_clean_id - 1}")
print(f"People confirmed       : {sorted(set(id_registry.values()))}")
print(f"Database location      : {DATABASE_PATH}")
print(f"Video saved            : {output_path}\n")