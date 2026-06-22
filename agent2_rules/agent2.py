import os
import sys
import sqlite3
import cv2
import numpy as np
import json

# --- PATH & MODULES ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'agent3_vlm'))
import agent3
from config import VIDEO_PATH, DATABASE_PATH

# --- CONFIG ---
WARMUP_SECONDS = 5
AFK_THRESHOLD = 10
CROPS_DIR = "violations"
os.makedirs(CROPS_DIR, exist_ok=True)

def load_zones():
    path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
    with open(path, "r") as f:
        return json.load(f)["zones"]

def main():
    zones_data = load_zones()
    
    # 1. Setup Polygons
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    polys = [np.array([[z[0]*w, z[1]*h], [z[2]*w, z[1]*h], [z[2]*w, z[3]*h], [z[0]*w, z[3]*h]], dtype=np.float32) for z in zones_data]

    # 2. Load Events
    conn = sqlite3.connect(DATABASE_PATH)
    rows = conn.execute("SELECT timestamp, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM events WHERE event_type = 'detected' ORDER BY person_id, timestamp").fetchall()
    conn.close()

    people = {}
    for r in rows: people.setdefault(r[1], []).append(r)

    # 3. Process
    zone_votes, home_zone_map, outside_start, locked = {}, {}, {}, set()

    for pid, detections in people.items():
        for det in detections:
            ts, _, x1, y1, x2, y2 = det
            cx, cy = (x1+x2)/2, (y1+y2)/2
            curr_zone = next((i for i, p in enumerate(polys) if cv2.pointPolygonTest(p, (cx, cy), False) >= 0), None)

            # Warmup Phase
            if ts <= WARMUP_SECONDS:
                zone_votes.setdefault(pid, {})
                if curr_zone is not None: zone_votes[pid][curr_zone] = zone_votes[pid].get(curr_zone, 0) + 1
                continue

            # Assign Home Zone
            if pid not in home_zone_map:
                home_zone_map[pid] = max(zone_votes[pid], key=zone_votes[pid].get) if zone_votes.get(pid) else None
            
            home = home_zone_map.get(pid)
            if home is None: continue

            # Violation Logic
            if curr_zone != home:
                if pid not in outside_start: outside_start[pid] = ts
                if (ts - outside_start[pid]) > AFK_THRESHOLD and pid not in locked:
                    locked.add(pid)
                    
                    cap = cv2.VideoCapture(VIDEO_PATH)
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                    success, frame = cap.read()
                    
                    if success:
                        # 1. DRAW ZONES (Spatial Context Mapper)
                        for i, poly in enumerate(polys):
                            color = (0, 0, 255) if i == home else (255, 0, 0)
                            cv2.polylines(frame, [poly.astype(int)], True, color, 2)
                            cv2.putText(frame, f"Zone {i}", (int(poly[0][0]), int(poly[0][1]-5)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 2. DRAW PERSON
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        
                        p_path = os.path.join(CROPS_DIR, f"v_{pid}_{int(ts)}.jpg")
                        cv2.imwrite(p_path, frame)
                        
                        # 3. GET VLM ANALYSIS
                        v_type = agent3.run_agent3_auditor(p_path, pid, home)
                        
                        # 4. SAVE TO DB
                        conn = sqlite3.connect(DATABASE_PATH)
                        conn.execute("UPDATE events SET vlm_summary = ?, crop_path = ?, event_type = ? WHERE person_id = ? AND timestamp = ?", 
                                     (v_type, p_path, "violation_away", pid, ts))
                        conn.commit()
                        conn.close()
                        print(f"Logged Violation: {v_type} for PID {pid}")
                    cap.release()
            else:
                outside_start.pop(pid, None)
                locked.discard(pid)

if __name__ == "__main__":
    main()