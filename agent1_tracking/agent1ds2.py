# agent1_tracking/agent1.py
# Agent 1: Full production pipeline.
# Tracks people using YOLO + ByteTrack, writes every detection to sentinel.db.
# This is the data source that feeds the entire system.
# Every time this runs, it clears old data and starts a fresh analysis.

import os
import sys
import cv2
import sqlite3
import time
from ultralytics import YOLO

# Fix paths so we can import config from the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH, MODEL_NAME, DATABASE_PATH

print("\n=== AGENT 1 STARTING: TRACKING + DATABASE FEED ===")

# --- CONNECT TO THE SHARED DATABASE ---
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Every time Agent 1 runs, it means we are processing a fresh video.
# We wipe the events table so old data from a previous run never
# bleeds into the current analysis and confuses Agents 2, 3, and 4.
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

fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video: {total_frames} frames at {fps:.2f} FPS")
print(f"[INFO] Writing detections to database every frame...")

# --- SET UP OUTPUT VIDEO FOR VISUAL VERIFICATION ---
output_dir = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Tracking_No_Violations.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- TRACKING REGISTRIES ---
raw_id_counters  = {}    # How many frames each raw ID has survived
id_registry      = {}    # Maps raw YOLO ID -> clean sequential ID
next_clean_id    = 1     # Our clean people counter

# NEW: Track when we last saw a raw ID to allow a grace period
last_seen_frame  = {}    
GRACE_PERIOD_FRAMES = 5  # Allow a raw ID to disappear for 5 frames without resetting

MIN_FRAMES_TO_CONFIRM = 12
frame_count = 0

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
        classes=[0, 67],   # 0 = people, 67 = cell phones (COCO class)
        conf=0.25,
        iou=0.30,
        imgsz=1536,
        augment=True,      # <--- ADDED: Helps catch faint/occluded people
        verbose=False
    )

    frame_results = results[0]

    # --- TRACK ACTIVE PEOPLE IDs FOR LOGGING ---
    active_people_ids = set()

    if frame_results.boxes.id is not None:
        raw_ids     = frame_results.boxes.id.int().tolist()
        boxes       = frame_results.boxes.xyxy.int().tolist()
        confidences = frame_results.boxes.conf.tolist()
        classes     = frame_results.boxes.cls.int().tolist()

        # --- PHASE 1: UPDATE PROBATION COUNTERS WITH GRACE PERIOD (PEOPLE ONLY) ---
        for raw_id, cls in zip(raw_ids, classes):
            if cls != 0:  # Skip phones for tracking
                continue
                
            # If seen recently within the grace period, increment count. Otherwise, reset.
            if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            else:
                raw_id_counters[raw_id] = 1

            # Update last seen frame
            last_seen_frame[raw_id] = frame_count

            # Graduate from probation
            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                id_registry[raw_id] = next_clean_id
                next_clean_id += 1
                print(f"[NEW PERSON] Person ID {id_registry[raw_id]} confirmed.")
        
        # --- PHASE 2: PROCESS ALL DETECTIONS (PEOPLE + PHONES) ---
        for raw_id, box, conf, cls in zip(raw_ids, boxes, confidences, classes):
            
            # --- CASE 1: PHONE DETECTION (class 67) - YELLOW BBOX ---
            if cls == 67 and conf >= 0.25:
                x1, y1, x2, y2 = box
                
                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round(current_timestamp, 3),
                    None,  # No person_id for phones
                    "phone_detected",
                    round(conf, 4),
                    x1, y1, x2, y2
                ))
                
                # --- YELLOW BOUNDING BOX (same style, yellow color) ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow
                label = "PHONE"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - txt_h - 6), (x1 + txt_w + 4, y1), (0, 255, 255), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # --- CASE 2: PERSON DETECTION (class 0) - GREEN BBOX ---
            elif cls == 0 and raw_id in id_registry:
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

                # --- GREEN BOUNDING BOX ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"ID:{clean_id}"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - txt_h - 6), (x1 + txt_w + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Commit to database every 30 frames to avoid hammering the disk
    if frame_count % 30 == 0:
        conn.commit()
        # --- UPDATED LOG: Shows people currently in frame + total confirmed ---
        print(f" -> Frame {frame_count}/{total_frames} | "
              f"Active: {sorted(active_people_ids)} | "
              f"Total confirmed: {list(id_registry.values())}")

    # Write annotated frame to output video
    video_writer.write(frame)

    # Store current frame's IDs for next iteration's continuity check
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