# agent2_rules/agent2.py
# Agent 2: Violation Detector
# Detects: UNAUTHORIZED ZONE, AFK, LEFT FRAME
# Each violation type logged ONCE per person per run.
# Captures TWO frames for AFK/LEFT violations:
#   Frame 1 — when person first left their zone
#   Frame 2 — when person returned (or end of video)

import os
import sys
import sqlite3
import cv2
import numpy as np
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'agent3_vlm'))

import agent3
from config import VIDEO_PATH, DATABASE_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
WARMUP_SECONDS  = 10
AFK_THRESHOLD   = 60
ZONE_THRESHOLD  = 10
CROPS_DIR       = os.path.join(ROOT_DIR, "violations")
os.makedirs(CROPS_DIR, exist_ok=True)

# ── COLORS (BGR) ─────────────────────────────────────────────────────────────
COLOR_HOME_ZONE  = (0,   0,   255)
COLOR_OTHER_ZONE = (255, 100, 0  )
COLOR_PERSON     = (0,   255, 0  )
COLOR_LABEL_BG   = (0,   0,   200)
COLOR_LABEL_TEXT = (255, 255, 255)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_zones():
    path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
    with open(path, "r") as f:
        return json.load(f)["zones"]

def build_polys(zones_data, w, h):
    return [
        np.array([[p[0] * w, p[1] * h] for p in z], dtype=np.float32)
        for z in zones_data
    ]

def get_zone(cx, cy, polys):
    for i, poly in enumerate(polys):
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            return i
    return None

def grab_frame(ts):
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    success, frame = cap.read()
    cap.release()
    return frame if success else None

def annotate_and_save(frame, pid, x1, y1, x2, y2, home_zone, polys, label, duration, suffix=""):
    if frame is None: return None
    
    # 1. Draw Zones (Polygons)
    for i, poly in enumerate(polys):
        color = COLOR_HOME_ZONE if i == home_zone else COLOR_OTHER_ZONE
        thickness = 3 if i == home_zone else 2
        # Use poly as it comes from the pre-calculated polygons list
        cv2.polylines(frame, [poly.astype(int)], True, color, thickness)
        
        # Label the zones
        cx_z = int(np.mean(poly[:, 0]))
        cy_z = int(np.mean(poly[:, 1]))
        cv2.putText(frame, f"Z{i}", (cx_z, cy_z), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2. Draw Person (Green box)
    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 3)
    
    # 3. Label the violation
    text = f"P{pid} {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), COLOR_LABEL_BG, -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    path = os.path.join(CROPS_DIR, f"v_{pid}_{label}_{suffix}.jpg")
    cv2.imwrite(path, frame)
    return path

def log_violation(pid, event_type, ts, duration, crop, vlm, x1, y1, x2, y2, zone_id, zone_name):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        INSERT INTO events
            (timestamp, person_id, event_type, duration_seconds,
             crop_path, vlm_summary, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             zone_id, zone_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (round(ts, 3), pid, event_type, round(duration, 2),
          crop, vlm, int(x1), int(y1), int(x2), int(y2),
          zone_id, zone_name))
    conn.commit()
    conn.close()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== AGENT 2 STARTING: VIOLATION DETECTION ===")

    zones_data = load_zones()

    cap            = cv2.VideoCapture(VIDEO_PATH)
    w              = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h              = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    polys = build_polys(zones_data, w, h)
    print(f"[INFO] {len(polys)} zones | Video: {video_duration:.1f}s")
    print(f"[INFO] AFK: {AFK_THRESHOLD}s | Zone: {ZONE_THRESHOLD}s\n")

    conn = sqlite3.connect(DATABASE_PATH)
    rows = conn.execute("""
        SELECT timestamp, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events WHERE event_type = 'detected'
        ORDER BY person_id, timestamp
    """).fetchall()
    conn.close()

    people = {}
    for r in rows:
        people.setdefault(r[1], []).append(r)

    print(f"[INFO] People: {sorted(people.keys())}\n")

    for pid, detections in people.items():
        print(f"── Person {pid} ({len(detections)} detections) ──")

        # PHASE 1: HOME ZONE
        zone_votes = {}
        for det in detections:
            ts, _, x1, y1, x2, y2 = det
            if ts > WARMUP_SECONDS:
                break
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            z = get_zone(cx, cy, polys)
            if z is not None:
                zone_votes[z] = zone_votes.get(z, 0) + 1

        if not zone_votes:
            print(f"  [WARN] No warmup zone data, skipping.\n")
            continue

        home_zone = max(zone_votes, key=zone_votes.get)
        print(f"  Home zone: Zone {home_zone}")

        # PHASE 2: VIOLATIONS
        outside_start     = None
        outside_start_det = None
        afk_logged        = False
        unauth_logged     = False
        last_det          = detections[-1]
        last_ts           = last_det[0]

        for det in detections:
            ts, _, x1, y1, x2, y2 = det
            if ts <= WARMUP_SECONDS: continue

            cx, cy    = (x1 + x2) / 2, (y1 + y2) / 2
            curr_zone = get_zone(cx, cy, polys)

            # 1. Back home — Reset everything
            if curr_zone == home_zone:
                outside_start = None
                continue

            # 2. Initialize absence
            if outside_start is None:
                outside_start = ts
                outside_start_det = det
            
            time_away = ts - outside_start

            # 3. UNAUTH ZONE: Only trigger if they are currently IN a zone (not walking in empty space)
            # and they have been away from home for more than the threshold
            if curr_zone is not None and not unauth_logged and time_away >= ZONE_THRESHOLD:
                # OPTIONAL: Check if they are actually 'stationary' here if you have velocity data
                print(f"  [!] UNAUTH ZONE {curr_zone} | {time_away:.1f}s | t={ts:.1f}s")
                _, _, ox1, oy1, ox2, oy2 = outside_start_det
                f1 = grab_frame(outside_start)
                crop1 = annotate_and_save(f1, pid, ox1, oy1, ox2, oy2, home_zone, polys, "UNAUTH", time_away, f"{int(outside_start)}")
                vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
                log_violation(pid, "violation_unauth_zone", outside_start, time_away, crop1, vlm, ox1, oy1, ox2, oy2, curr_zone, f"Zone {curr_zone}")
                print(f"  [DB] Logged | VLM: {vlm}")
                unauth_logged = True
            
            # 4. AFK: Trigger if they are in empty space (curr_zone is None)
            elif curr_zone is None and not afk_logged and time_away >= AFK_THRESHOLD:
                print(f"  [!] AFK | {time_away:.1f}s | t={ts:.1f}s")
                _, _, ox1, oy1, ox2, oy2 = outside_start_det
                f1 = grab_frame(outside_start)
                crop1 = annotate_and_save(f1, pid, ox1, oy1, ox2, oy2, home_zone, polys, "AFK", time_away, f"{int(outside_start)}_left")
                f2 = grab_frame(ts)
                crop2 = annotate_and_save(f2, pid, x1, y1, x2, y2, home_zone, polys, "AFK", time_away, f"{int(ts)}_now")
                vlm = agent3.run_agent3_auditor(crop1, pid, home_zone, image_path_2=crop2)
                log_violation(pid, "violation_afk", outside_start, time_away, crop1, vlm, ox1, oy1, ox2, oy2, home_zone, f"Zone {home_zone}")
                print(f"  [DB] Logged | VLM: {vlm}")
                afk_logged = True
        # PHASE 3: LEFT AND NEVER RETURNED
        time_since_last = video_duration - last_ts
        if time_since_last >= AFK_THRESHOLD and not afk_logged:
            _, _, x1, y1, x2, y2 = last_det
            print(f"  [!] LEFT FRAME at t={last_ts:.1f}s | gone {time_since_last:.1f}s")

            f1    = grab_frame(last_ts)
            crop1 = annotate_and_save(f1, pid, x1, y1, x2, y2,
                                      home_zone, polys, "LEFT", time_since_last,
                                      f"{int(last_ts)}_last")

            f2    = grab_frame(video_duration - 5)
            crop2 = annotate_and_save(f2, pid, x1, y1, x2, y2,
                                      home_zone, polys, "LEFT", time_since_last,
                                      f"{int(video_duration)}_end")

            vlm = agent3.run_agent3_auditor(crop1, pid, home_zone, image_path_2=crop2)
            log_violation(pid, "violation_left_frame", last_ts, time_since_last,
                          crop1, vlm, x1, y1, x2, y2, home_zone, f"Zone {home_zone}")
            print(f"  [DB] Logged | VLM: {vlm}")

        print()

    # SUMMARY
    conn = sqlite3.connect(DATABASE_PATH)
    violations = conn.execute("""
        SELECT person_id, event_type, duration_seconds, timestamp
        FROM events WHERE event_type LIKE 'violation_%'
        ORDER BY timestamp
    """).fetchall()
    conn.close()

    print("########################################")
    print("  AGENT 2 COMPLETE: VIOLATIONS LOGGED  ")
    print("########################################")
    print(f"Total violations: {len(violations)}")
    for v in violations:
        print(f"  Person {v[0]} | {v[1]} | {v[2]:.1f}s | t={v[3]:.1f}s")

if __name__ == "__main__":
    main()