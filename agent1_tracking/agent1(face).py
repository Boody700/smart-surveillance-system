# agent1_tracking/agent1.py

# Agent 1: Full production pipeline.
# Tracks people using YOLO + ByteTrack, writes every detection to sentinel.db.
# Max 3 IDs. IDs assigned by average position during probation (their settled seat).
# Once confirmed, ID sticks to the person forever regardless of movement.
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
print(f"[INFO] Max people: 3 | Seat-based ID assignment (locked after confirmation)")
print(f"[INFO] -----------------------------------------------")

# --- SET UP OUTPUT VIDEO ---
output_dir   = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path  = os.path.join(output_dir, "First_Detection(face).mp4")
fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- LOAD MODEL ---
model = YOLO(MODEL_NAME)

# --- TRACKING REGISTRIES ---
raw_id_counters   = {}   # probation frame counts
last_seen_frame   = {}   # grace period tracking
position_history  = {}   # raw_id -> list of center_x values during probation

# Maps raw YOLO ID -> clean ID (1, 2, or 3)
id_registry = {}

# Locked seat position for each clean ID — set once at confirmation, never updated
seat_positions = {}   # clean_id -> average center_x at time of confirmation

MAX_PEOPLE            = 3
MIN_FRAMES_TO_CONFIRM = 35
GRACE_PERIOD_FRAMES   = 45
MIN_WRITE_CONF        = 0.25

frame_count = 0
raw_ids     = []

def get_center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) // 2

def assign_clean_id_by_seat(avg_center_x):
    """
    Assign clean ID based on average seated position.
    If under MAX_PEOPLE: assign next available slot.
    If all slots filled (re-entry after long absence): match to closest seat.
    """
    if len(seat_positions) < MAX_PEOPLE:
        # Sort existing seats left to right and insert this person
        # to figure out which slot (1, 2, 3) they belong to
        existing = sorted(seat_positions.items(), key=lambda x: x[1])  # (cid, pos)
        taken_ids = set(seat_positions.keys())

        # All positions including the new one
        all_positions = [(cid, pos) for cid, pos in existing] + [(-1, avg_center_x)]
        all_positions.sort(key=lambda x: x[1])  # sort by x position

        # Find rank of new person among confirmed seats
        rank = next(i for i, (cid, _) in enumerate(all_positions) if cid == -1)

        # Assign the (rank+1)-th available ID
        available = sorted([i for i in [1, 2, 3] if i not in taken_ids])
        if rank < len(available):
            return available[rank]
        else:
            return available[-1]
    else:
        # All 3 confirmed — match re-entry to closest seat
        best_cid  = None
        best_dist = float("inf")
        for cid, pos in seat_positions.items():
            dist = abs(avg_center_x - pos)
            if dist < best_dist:
                best_dist = dist
                best_cid  = cid
        return best_cid

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
        classes=[0],
        iou=0.20,
        imgsz=1536,
        conf=0.20,
        verbose=False
    )

    frame_results = results[0]

    if frame_results.boxes.id is not None:
        raw_ids     = frame_results.boxes.id.int().tolist()
        boxes       = frame_results.boxes.xyxy.int().tolist()
        confidences = frame_results.boxes.conf.tolist()

        # --- PHASE 1: PROBATION COUNTERS + POSITION HISTORY ---
        for raw_id, box in zip(raw_ids, boxes):
            if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            else:
                raw_id_counters[raw_id] = 1
                position_history[raw_id] = []  # reset position history on new track

            last_seen_frame[raw_id] = frame_count

            # Accumulate center_x during probation (only before confirmation)
            if raw_id not in id_registry:
                position_history.setdefault(raw_id, []).append(get_center_x(box))

            # Graduate from probation
            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                # Average position over entire probation = their seat
                avg_x    = int(np.mean(position_history.get(raw_id, [get_center_x(box)])))
                clean_id = assign_clean_id_by_seat(avg_x)

                if clean_id is not None:
                    id_registry[raw_id]      = clean_id
                    seat_positions[clean_id] = avg_x   # locked forever
                    print(f"\n[NEW PERSON] *** Person {clean_id} confirmed | seat at x={avg_x} ***\n")
                else:
                    print(f"[WARN] Could not assign clean ID for raw_id {raw_id}, skipping.")

        # --- PHASE 2: WRITE TO DATABASE + DRAW ON VIDEO ---
        for raw_id, box, conf in zip(raw_ids, boxes, confidences):
            if raw_id in id_registry and conf >= MIN_WRITE_CONF:
                clean_id = id_registry[raw_id]
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

        active_this_frame = sorted(set([
            id_registry[rid] for rid in raw_ids
            if rid in id_registry
        ])) if frame_results.boxes.id is not None else []

        progress = (frame_count / total_frames) * 100
        print(f"[{progress:5.1f}%] Frame {frame_count}/{total_frames} | "
              f"On screen now: {active_this_frame} | "
              f"All confirmed: {sorted(set(id_registry.values()))}")

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
print(f"Total unique people    : {len(set(id_registry.values()))}")
print(f"People confirmed       : {sorted(set(id_registry.values()))}")
print(f"Seat positions         : {seat_positions}")
print(f"Database location      : {DATABASE_PATH}")
print(f"Video saved            : {output_path}\n")