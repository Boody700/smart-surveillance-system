# agent1_tracking/test_run.py
# Agent 1: Production Tracking with Temporal Stabilization & Ghost Filtering

import os
import sys
import cv2
from ultralytics import YOLO

# Fix paths for integration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH, MODEL_NAME

print("\n=== [ADVANCED TEMPORAL STABILIZATION] PURGING GHOST TRACKS ===")

video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"[ERROR] Can't open: {VIDEO_PATH}")
    sys.exit()

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

output_dir = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_production_tracked.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

model = YOLO(MODEL_NAME)
frame_count = 0

# --- ADVANCED STABILIZATION REGISTRY ---
raw_id_counters = {}  # Tracks how many frames a raw ID has lived: {raw_id: frame_count}
id_registry = {}      # Confirmed human mapping: {raw_id: clean_sequential_id}
next_clean_id = 1     # Clean user-facing counter

# HYPERPARAMETER: How many frames must a person exist to be considered "real"
# 15 frames = 0.5 seconds. Split-second chair blips will NEVER hit this threshold.
MIN_FRAMES_TO_CONFIRM = 15 

while True:
    success, frame = video_capture.read()
    if not success:
        break
        
    frame_count += 1
    
    results = model.track(
        frame, 
        persist=True, 
        tracker="agent1_tracking/custom_tracker.yaml", 
        classes=[0], 
        iou=0.20, 
        imgsz=1280,
        augment=True, 
        verbose=False
    )
    
    frame_results = results[0]
    annotated_frame = frame.copy()
    clean_active_ids = []
    
    if frame_results.boxes.id is not None:
        raw_ids = frame_results.boxes.id.int().tolist()
        boxes = frame_results.boxes.xyxy.int().tolist()
        
        # Phase 1: Update frame persistence counters for everything seen
        for raw_id in raw_ids:
            raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            
            # If it passes the probationary period and isn't registered yet, register it!
            if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                id_registry[raw_id] = next_clean_id
                next_clean_id += 1
        
        # Phase 2: Only draw boxes for CONFIRMED tracks
        for raw_id, box in zip(raw_ids, boxes):
            if raw_id in id_registry:
                display_id = id_registry[raw_id]
                clean_active_ids.append(display_id)
                
                # Draw high-quality green display artifacts
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 145, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"Person ID: {display_id}", (x1 + 5, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Console tracking validation
    if frame_count % 30 == 0:
        print(f" -> Frame {frame_count}/{total_frames} | Confirmed Humans Transmitting: {clean_active_ids}")
        
    video_writer.write(annotated_frame)

video_capture.release()
video_writer.release()

print("\n########################################")
print("  PRODUCTION STABILIZATION RE-RENDERED! ")
print("########################################")
print(f"Clean system output saved: {output_path}\n")