import os
import sys
import cv2
import sqlite3
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import VIDEO_PATH, MODEL_NAME, DATABASE_PATH

# --- CONFIG ---
MIN_FRAMES_TO_CONFIRM = 35
REID_DISTANCE_THRESHOLD = 250
DUPLICATE_THRESHOLD = 100 
EMA_ALPHA = 0.7 


# --- INITIALIZATION ---
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("DELETE FROM events"); cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
conn.commit()

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(ROOT_DIR, "output_videos", "Final_Tracking.mp4")
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# --- REGISTRIES ---
id_registry = {}; last_pos = {}; smoothed_boxes = {}; raw_id_counters = {}; confirmed_ids = set(); next_clean_id = 1

def get_center(box): return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def is_duplicate(new_box, existing_boxes):
    new_center = get_center(new_box)
    for ex_box in existing_boxes:
        if np.linalg.norm(np.array(new_center) - np.array(get_center(ex_box))) < DUPLICATE_THRESHOLD: return True
    return False

# GEOMETRIC FILTER: Rejects things that are too wide (chairs/tables) or too thin
def is_valid_person(box):
    width, height = box[2] - box[0], box[3] - box[1]
    aspect_ratio = width / height if height > 0 else 0
    return 0.15 < aspect_ratio < 0.9

print("[INFO] Starting refined pipeline...")

while True:
    success, frame = cap.read()
    if not success: break
    
    results = model.track(frame, persist=True, tracker="agent1_tracking/custom_tracker.yaml", 
                          classes=[0, 67], conf=0.30 , iou=0.45 , imgsz=1536, verbose=False)

    frame_processed_boxes = []

    # 1. HANDLE PHONES (YELLOW)
    if results[0].boxes is not None:
        for cls, box in zip(results[0].boxes.cls.int().tolist(), results[0].boxes.xyxy.int().tolist()):
            if cls == 67:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                cursor.execute("INSERT INTO events (timestamp, event_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?,?,?,?,?,?)",
                               (round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 3), "phone_detected", *box))

    # 2. HANDLE PEOPLE (GREEN + GEOMETRIC FILTER)
    if results[0].boxes.id is not None:
        raw_ids = results[0].boxes.id.int().tolist()
        boxes = results[0].boxes.xyxy.int().tolist()
        
        for raw_id, box in zip(raw_ids, boxes):
            if not is_valid_person(box) or is_duplicate(box, frame_processed_boxes): continue
            
            raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
            if raw_id not in smoothed_boxes: smoothed_boxes[raw_id] = [float(v) for v in box]
            else: smoothed_boxes[raw_id] = [EMA_ALPHA * smoothed_boxes[raw_id][i] + (1 - EMA_ALPHA) * float(box[i]) for i in range(4)]
            
            s_box = [int(v) for v in smoothed_boxes[raw_id]]
            center = get_center(s_box)
            frame_processed_boxes.append(s_box)

            if raw_id_counters[raw_id] == MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                assigned_cid = None
                for cid, pos in last_pos.items():
                    if np.linalg.norm(np.array(center) - np.array(pos)) < REID_DISTANCE_THRESHOLD:
                        assigned_cid = cid; break
                if assigned_cid is None: assigned_cid = next_clean_id; next_clean_id += 1
                id_registry[raw_id] = assigned_cid
                if assigned_cid not in confirmed_ids: confirmed_ids.add(assigned_cid)

            if raw_id in id_registry:
                cid = id_registry[raw_id]
                last_pos[cid] = center
                cv2.rectangle(frame, (s_box[0], s_box[1]), (s_box[2], s_box[3]), (0, 255, 0), 1)
                label = f"ID:{cid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (s_box[0], s_box[1]-th-6), (s_box[0]+tw+4, s_box[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (s_box[0]+2, s_box[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                cursor.execute("INSERT INTO events (timestamp, person_id, event_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?,?,?,?,?,?,?)",
                               (round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 3), cid, "detected", *s_box))

    cv2.imshow("Live Tracking", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) == ord('q'): break
    video_writer.write(frame)

cv2.destroyAllWindows()
conn.commit(); conn.close(); cap.release(); video_writer.release()