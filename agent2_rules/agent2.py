import os
import sys
import sqlite3
import cv2
import numpy as np
import json

# ---------------- PATH SETUP ----------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import VIDEO_PATH, DATABASE_PATH

# ---------------- SETTINGS ----------------
WARMUP_SECONDS = 5 
AFK_THRESHOLD = 10

# ---------------- LOAD ZONES ----------------
def load_zones():
    json_path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
    if not os.path.exists(json_path):
        print("Please run calibrate_zones.py first")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)["zones"]

# ---------------- MAIN ----------------
def main():
    zones_data = load_zones()

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Convert zones → pixel polygons
    pixel_polygons = []
    for z in zones_data:
        x1, y1, x2, y2 = z
        poly = np.array([
            [x1 * width, y1 * height],
            [x2 * width, y1 * height],
            [x2 * width, y2 * height],
            [x1 * width, y2 * height]
        ], dtype=np.float32)
        pixel_polygons.append(poly)

    # ---------------- DB ----------------
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, timestamp, person_id, confidence,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events
        WHERE event_type = 'detected'
        ORDER BY person_id, timestamp
    """)
    rows = cursor.fetchall()

    # Group by person
    people = {}
    for r in rows:
        people.setdefault(r[2], []).append(r)

    # ---------------- STATE ----------------
    zone_votes = {}
    person_zone = {}
    outside_start = {}
    violation_locked = set()
    violations = 0

    # ---------------- PROCESS ----------------
    for pid, detections in people.items():
        for det in detections:
            _, timestamp, _, conf, x1, y1, x2, y2 = det
            timestamp = float(timestamp)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # -------- FIND CURRENT ZONE --------
            current_zone = None
            for zi, poly in enumerate(pixel_polygons):
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    current_zone = zi
                    break

            # -------- WARMUP (ASSIGN DESK) --------
            if timestamp <= WARMUP_SECONDS:
                if pid not in zone_votes:
                    zone_votes[pid] = {}
                if current_zone is not None:
                    zone_votes[pid][current_zone] = zone_votes[pid].get(current_zone, 0) + 1
                continue

            # Assign home zone once
            if pid not in person_zone:
                if pid in zone_votes and zone_votes[pid]:
                    person_zone[pid] = max(zone_votes[pid], key=zone_votes[pid].get)
                else:
                    person_zone[pid] = None

            home_zone = person_zone.get(pid, None)
            if home_zone is None:
                continue

            inside_home = (current_zone == home_zone)

            # -------- AFK LOGIC --------
            if not inside_home:
                if pid not in outside_start:
                    outside_start[pid] = timestamp
                else:
                    duration = timestamp - outside_start[pid]

                    if duration > AFK_THRESHOLD and pid not in violation_locked:
                        violations += 1
                        violation_locked.add(pid)

                        mins = int(duration // 60)
                        secs = int(duration % 60)

                        print(
                            f"VIOLATION: Person {pid} AFK for "
                            f"{mins}m {secs}s at {round(timestamp, 2)}s"
                        )
            else:
                # Claude's Fix: Pop the key entirely out of the dictionary when inside home
                outside_start.pop(pid, None)
                violation_locked.discard(pid)

    print("\n============================")
    print(f"Total violations: {violations}")
    print("============================")

    conn.close()

if __name__ == "__main__":
    main()