# agent2_rules/agent2.py
# Agent 2: Violation Detector
# Detects: UNAUTHORIZED ZONE, AFK, LEFT FRAME
# Look-ahead logic: if an absence will become AFK, skip UNAUTH entirely.
# AFK uses single empty-chair frame from middle of absence.

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
WARMUP_SECONDS  = 20
AFK_THRESHOLD   = 60
ZONE_THRESHOLD  = 20
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
    if frame is None:
        return None

    fh, fw = frame.shape[:2]

    for i, poly in enumerate(polys):
        color     = COLOR_HOME_ZONE if i == home_zone else COLOR_OTHER_ZONE
        thickness = 3 if i == home_zone else 2
        cv2.polylines(frame, [poly.astype(int)], True, color, thickness)
        cx_z = int(np.mean(poly[:, 0]))
        cy_z = int(np.mean(poly[:, 1]))
        zone_label = f"Zone {i}" + (" (YOURS)" if i == home_zone else "")
        cv2.putText(frame, zone_label, (cx_z - 40, cy_z),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if x1 is not None:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_PERSON, 3)

    viol_text = f"{label} | Person {pid}"
    (tw, th), _ = cv2.getTextSize(viol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    lx = 10 if x1 is None else int(x1)
    ly = 40
    cv2.rectangle(frame, (lx, ly), (lx + tw + 6, ly + th + 6), COLOR_LABEL_BG, -1)
    cv2.putText(frame, viol_text, (lx + 3, ly + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_LABEL_TEXT, 2)

    dur_text = f"Absence duration: {int(duration)} seconds ({duration / 60:.1f} minutes)"
    (dw, dh), _ = cv2.getTextSize(dur_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    dx, dy = 10, fh - 15
    cv2.rectangle(frame, (dx - 4, dy - dh - 8), (dx + dw + 4, dy + 4), (0, 0, 0), -1)
    cv2.putText(frame, dur_text, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
          crop, vlm,
          int(x1) if x1 is not None else None,
          int(y1) if y1 is not None else None,
          int(x2) if x2 is not None else None,
          int(y2) if y2 is not None else None,
          zone_id, zone_name))
    conn.commit()
    conn.close()

def becomes_afk(future_dets, outside_start, afk_threshold, polys, home_zone):
    """
    Look ahead in remaining detections to see if this absence will become AFK.
    Returns True if person stays out of home zone long enough to hit AFK threshold.
    Returns False if they return home before that.
    """
    for det in future_dets:
        ts, _, x1, y1, x2, y2 = det
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        curr_zone = get_zone(cx, cy, polys)
        if curr_zone == home_zone:
            return False  # came back — won't be AFK
        if ts - outside_start >= afk_threshold:
            return True   # confirmed will be AFK
    return False  # video ended without AFK threshold hit

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== AGENT 2 STARTING: VIOLATION DETECTION ===")

    conn = sqlite3.connect(DATABASE_PATH)
    deleted = conn.execute("DELETE FROM events WHERE event_type LIKE 'violation_%'").rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"[INFO] Cleared {deleted} previous violation(s) from DB.")

    zones_data = load_zones()

    cap            = cv2.VideoCapture(VIDEO_PATH)
    w              = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h              = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    polys = build_polys(zones_data, w, h)
    print(f"[INFO] {len(polys)} zones | Video duration: {video_duration:.0f} seconds")
    print(f"[INFO] AFK: {AFK_THRESHOLD}s | Zone: {ZONE_THRESHOLD}s | Warmup: {WARMUP_SECONDS}s\n")

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

    print(f"[INFO] People found in DB: {sorted(people.keys())}\n")

    for pid, detections in people.items():
        print(f"── Person {pid} ({len(detections)} detections) ──")

        # PHASE 1: HOME ZONE via warmup voting
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

        for i, det in enumerate(detections):
            ts, _, x1, y1, x2, y2 = det
            if ts <= WARMUP_SECONDS:
                continue

            cx, cy    = (x1 + x2) / 2, (y1 + y2) / 2
            curr_zone = get_zone(cx, cy, polys)

            if curr_zone == home_zone:
                outside_start     = None
                outside_start_det = None
                continue

            if outside_start is None:
                outside_start     = ts
                outside_start_det = det

            time_away = ts - outside_start

            # AFK — check first, takes priority
            if curr_zone is None and time_away >= AFK_THRESHOLD and not afk_logged:
                _, _, ox1, oy1, ox2, oy2 = outside_start_det
                print(f"  [VIOLATION] Person {pid} was absent (AFK) for {time_away:.0f} seconds")

                mid_ts = outside_start + (time_away / 2)
                f1     = grab_frame(mid_ts)
                crop1  = annotate_and_save(f1, pid, None, None, None, None,
                                           home_zone, polys, "AFK", time_away,
                                           f"{int(mid_ts)}_empty_chair")

                vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
                log_violation(pid, "violation_afk", outside_start, time_away,
                              crop1, vlm, ox1, oy1, ox2, oy2, home_zone, f"Zone {home_zone}")
                print(f"  [DB] Logged | VLM says: {vlm}")
                afk_logged    = True
                unauth_logged = True  # block UNAUTH for this absence

            # UNAUTHORIZED ZONE — only if absence won't become AFK
            elif curr_zone is not None and time_away >= ZONE_THRESHOLD and not unauth_logged and not afk_logged:
                # Look ahead — skip UNAUTH if this will become AFK
                future_dets = detections[i:]
                if becomes_afk(future_dets, outside_start, AFK_THRESHOLD, polys, home_zone):
                    continue  # wait — AFK will handle this

                _, _, ox1, oy1, ox2, oy2 = outside_start_det
                print(f"  [VIOLATION] Person {pid} was in unauthorized Zone {curr_zone} for {time_away:.0f} seconds")

                f1    = grab_frame(outside_start)
                crop1 = annotate_and_save(f1, pid, ox1, oy1, ox2, oy2,
                                          home_zone, polys, "UNAUTH", time_away,
                                          f"{int(outside_start)}_start")
                vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
                log_violation(pid, "violation_unauth_zone", outside_start, time_away,
                              crop1, vlm, ox1, oy1, ox2, oy2, curr_zone, f"Zone {curr_zone}")
                print(f"  [DB] Logged | VLM says: {vlm}")
                unauth_logged = True

        # PHASE 3: LEFT AND NEVER RETURNED
        time_since_last = video_duration - last_ts
        if time_since_last >= AFK_THRESHOLD and not afk_logged:
            _, _, x1, y1, x2, y2 = last_det
            print(f"  [VIOLATION] Person {pid} left at {last_ts:.0f} seconds "
                  f"and never returned (gone for {time_since_last:.0f} seconds)")

            f1    = grab_frame(video_duration - 5)
            crop1 = annotate_and_save(f1, pid, None, None, None, None,
                                      home_zone, polys, "LEFT", time_since_last,
                                      f"{int(video_duration)}_empty")

            vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
            log_violation(pid, "violation_left_frame", last_ts, time_since_last,
                          crop1, vlm, x1, y1, x2, y2, home_zone, f"Zone {home_zone}")
            print(f"  [DB] Logged | VLM says: {vlm}")

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
    print(f"Total violations found: {len(violations)}")
    for v in violations:
        print(f"  Person {v[0]} | {v[1]} | Gone for {v[2]:.0f} seconds | At {v[3]:.0f}s in video")

if __name__ == "__main__":
    main()