# agent1_tracking/agent1.py
# Agent 1: Full production pipeline.
# Tracks people using YOLO + ByteTrack, writes every detection to sentinel.db.
# Single tracking pass with classes=[0,67] keeps person detection stable.
# Separate phone predict pass at lower conf for better phone detection.
# Position-based Re-ID: when ByteTrack loses someone, match by last known position.
# Every time this runs, it clears old data and starts a fresh analysis.

import os
import sys
import cv2
import sqlite3
import numpy as np
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
print(f"[INFO] -----------------------------------------------")

# --- SET UP OUTPUT VIDEO ---
output_dir   = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path  = os.path.join(output_dir, "Tracking_No_Violations.mp4")
fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- TRACKING REGISTRIES ---
raw_id_counters  = {}
id_registry      = {}
next_clean_id    = 1
last_seen_frame  = {}
position_history = {}
last_known_pos   = {}

GRACE_PERIOD_FRAMES     = 45
MIN_FRAMES_TO_CONFIRM   = 45
REID_DISTANCE_THRESHOLD = 250

phones_this_window = 0
frame_count        = 0
raw_ids            = []

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
    if best_dist <= REID_DISTANCE_THRESHOLD:
        return best_cid
    return None

# --- MAIN PROCESSING LOOP ---
while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count += 1
    current_timestamp = frame_count / fps

    # ================================================================
    # PASS 1: TRACKING — people + phones together (keeps left guy stable)
    # ================================================================
    results = model.track(
        frame,
        persist=True,
        tracker="agent1_tracking/custom_tracker.yaml",
        classes=[0, 67],    # both — context helps detect occluded left guy
        conf=0.25,
        iou=0.30,
        imgsz=1536,
        augment=True,
        verbose=False
    )

    # ================================================================
    # PASS 2: PHONE DETECTION — separate predict at lower conf
    # ================================================================
    phone_results = model.predict(
        frame,
        classes=[67],
        conf=0.20,          # lower than tracking pass — catches phones better
        iou=0.30,
        imgsz=1536,
        augment=True,
        verbose=False
    )

    frame_results = results[0]
    phone_frame   = phone_results[0]

    # --- PHONE DETECTIONS (from dedicated phone pass) ---
    # We use a set of bbox tuples to deduplicate vs tracking pass phones
    logged_phone_boxes = set()

    if phone_frame.boxes is not None and len(phone_frame.boxes) > 0:
        for box, conf in zip(phone_frame.boxes.xyxy.int().tolist(),
                             phone_frame.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            box_key = (x1, y1, x2, y2)
            if box_key in logged_phone_boxes:
                continue
            logged_phone_boxes.add(box_key)
            phones_this_window += 1

            cursor.execute("""
                INSERT INTO events
                    (timestamp, person_id, event_type, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                round(current_timestamp, 3),
                None,
                "phone_detected",
                round(conf, 4),
                x1, y1, x2, y2
            ))

            # Yellow box for phones
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            label = f"PHONE {conf:.2f}"
            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - txt_h - 6), (x1 + txt_w + 4, y1), (0, 255, 255), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # --- PERSON TRACKING ---
    active_people_ids = set()

    if frame_results.boxes.id is not None:
        all_ids     = frame_results.boxes.id.int().tolist()
        all_classes = frame_results.boxes.cls.int().tolist()
        all_boxes   = frame_results.boxes.xyxy.int().tolist()
        all_confs   = frame_results.boxes.conf.tolist()

        # Filter to people only for tracking
        person_data = [
            (rid, box, conf)
            for rid, cls, box, conf in zip(all_ids, all_classes, all_boxes, all_confs)
            if cls == 0
        ]

        raw_ids     = [d[0] for d in person_data]
        boxes       = [d[1] for d in person_data]
        confidences = [d[2] for d in person_data]

        # --- PHASE 1: PROBATION + POSITION HISTORY ---
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

            # Graduate from probation
            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                positions = position_history.get(raw_id, [(cx, cy)])
                avg_cx = int(np.mean([p[0] for p in positions]))
                avg_cy = int(np.mean([p[1] for p in positions]))

                matched_cid = find_matching_clean_id(avg_cx, avg_cy)

                if matched_cid is not None:
                    id_registry[raw_id] = matched_cid
                    print(f"[RE-ENTRY] Person {matched_cid} returned! (raw_id={raw_id})")
                else:
                    id_registry[raw_id] = next_clean_id
                    next_clean_id += 1
                    print(f"[NEW PERSON] Person ID {id_registry[raw_id]} confirmed.")

        # --- PHASE 2: UPDATE LAST KNOWN POSITIONS ---
        for raw_id, box in zip(raw_ids, boxes):
            if raw_id in id_registry:
                last_known_pos[id_registry[raw_id]] = get_center(box)

        # --- PHASE 3: WRITE TO DB + DRAW ---
        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id in id_registry:
                clean_id = id_registry[raw_id]
                active_people_ids.add(clean_id)
                x1, y1, x2, y2 = box

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
        phone_status = f"📱 {phones_this_window} phone detections" if phones_this_window > 0 else "no phones"
        print(f" -> Frame {frame_count}/{total_frames} | "
              f"Active: {sorted(active_people_ids)} | "
              f"Total confirmed: {sorted(set(id_registry.values()))} | "
              f"{phone_status}")
        phones_this_window = 0

    video_writer.write(frame)
    last_frame_ids = set(raw_ids) if frame_results.boxes.id is not None else set()

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
print(f"Database location      : {DATABASE_PATH}")
print(f"Video saved            : {output_path}\n")