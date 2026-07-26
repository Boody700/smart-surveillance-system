# agent2_rules/agent2.py
# Agent 2: Violation Detector
# Detects: UNAUTHORIZED ZONE, AFK, LEFT FRAME, PHONE USAGE, SLEEPING
#
# PHONE USAGE and SLEEPING are deterministic - built directly from Agent 1's
# purpose-built phone detector / pose heuristic, no VLM call. Re-asking
# LLaVA "is this a phone/is this person asleep" when a dedicated detector
# already confirmed it would be the same redundant-VLM-work problem this
# project already moved away from once (see AFK/UNAUTH/LEFT below, which DO
# still use Agent 3 - those genuinely benefit from the model's visual/color
# context in a way a phone box or a keypoint heuristic doesn't need).
#
# Look-ahead logic: if an absence will become AFK, skip UNAUTH entirely.
# AFK uses single empty-chair frame from middle of absence.

import os
import sys
import sqlite3
import cv2
import numpy as np
import json
import bisect

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'agent3_vlm'))

import agent3
from config import VIDEO_PATH, DATABASE_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
WARMUP_SECONDS  = 20
AFK_THRESHOLD   = 60
ZONE_THRESHOLD  = 10   # was 20 - "being in another zone for more than 10 seconds"
CROPS_DIR       = os.path.join(ROOT_DIR, "violations")
os.makedirs(CROPS_DIR, exist_ok=True)

# Phone usage: how sustained the phone-near-face signal needs to be before
# it's a violation rather than a brief glance/reach.
PHONE_USAGE_MIN_SECONDS   = 20  # was 5
PHONE_EPISODE_GAP_SECONDS = 3

# "Usage" means the phone is actually up near the person's face (texting,
# scrolling, on a call) - not just anywhere inside their body bbox, which
# would also fire for a phone sitting in their lap or on the desk in front
# of them while they're not touching it. Distance is normalized by the
# face box's own height rather than a fixed pixel count, so it scales
# sensibly whether someone's close to the camera or far away.
PHONE_FACE_PROXIMITY_RATIO = 2.5   # phone-to-face center distance, in face-box-heights
PHONE_FACE_MAX_TIME_GAP    = 2.0   # seconds - how stale a face reading can be and still be trusted


# Sleeping: Agent 1 already requires SLEEP_MIN_CONSECUTIVE_FRAMES before it
# ever writes the first sleeping_detected row, so this is mostly about
# grouping the (already-confirmed) rows into one episode and setting a
# floor on how long the whole thing lasted before it's worth a report entry.
SLEEP_MIN_SECONDS          = 10
SLEEP_EPISODE_GAP_SECONDS  = 5

# A room-wide violation, distinct from any individual person's AFK: a
# stretch where NOBODY is detected anywhere in frame at all, not just one
# person missing from their own zone. Defaults to the same threshold as
# AFK for a single person - tune independently if you want a different bar
# for "everyone's gone" vs "one person's gone".
EMPTY_ROOM_MIN_SECONDS = 60

def format_timestamp(seconds):
    """Format a point in time as M:SS (e.g. 192.4 -> '3:12'). Durations
    (how LONG something lasted) are left as seconds/minutes elsewhere -
    this is specifically for WHEN something happened in the video."""
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"

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

def load_faces_by_person():
    """person_id -> sorted list of (ts, x1, y1, x2, y2) from Agent 1's
    face_detected rows (already person-attributed there, unlike
    phone_detected)."""
    conn = sqlite3.connect(DATABASE_PATH)
    rows = conn.execute("""
        SELECT timestamp, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events WHERE event_type = 'face_detected'
        ORDER BY person_id, timestamp
    """).fetchall()
    conn.close()
    by_person = {}
    for ts, pid, x1, y1, x2, y2 in rows:
        by_person.setdefault(pid, []).append((ts, x1, y1, x2, y2))
    return by_person

def nearest_face_bbox(faces_for_person, target_ts, max_gap_seconds=PHONE_FACE_MAX_TIME_GAP):
    """faces_for_person: sorted list of (ts,x1,y1,x2,y2) for ONE person.
    Returns bbox closest in time to target_ts, or None if too stale/empty."""
    if not faces_for_person:
        return None
    timestamps = [f[0] for f in faces_for_person]
    idx = bisect.bisect_left(timestamps, target_ts)
    candidates = []
    if idx < len(faces_for_person):
        candidates.append(faces_for_person[idx])
    if idx > 0:
        candidates.append(faces_for_person[idx - 1])
    best = min(candidates, key=lambda f: abs(f[0] - target_ts))
    if abs(best[0] - target_ts) > max_gap_seconds:
        return None
    return best[1], best[2], best[3], best[4]

def phone_face_distance_ratio(phone_box, face_box):
    """Center-to-center distance between the phone and a face box,
    normalized by the face box's own height (so it scales with how close
    the person is to the camera instead of using a fixed pixel threshold)."""
    px1, py1, px2, py2 = phone_box
    fx1, fy1, fx2, fy2 = face_box
    phone_cx, phone_cy = (px1 + px2) / 2, (py1 + py2) / 2
    face_cx, face_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    face_h = max(1, fy2 - fy1)
    dist = ((phone_cx - face_cx) ** 2 + (phone_cy - face_cy) ** 2) ** 0.5
    return dist / face_h

def nearest_detection_bbox(detections, target_ts, max_gap_seconds=1.0):
    """detections: sorted list of (ts, pid, x1,y1,x2,y2) for ONE person.
    Returns the bbox (x1,y1,x2,y2) of whichever entry is closest in time to
    target_ts, or None if the closest one is still too far away (person
    wasn't actually tracked near that moment) or the list is empty."""
    if not detections:
        return None
    timestamps = [d[0] for d in detections]
    idx = bisect.bisect_left(timestamps, target_ts)
    candidates = []
    if idx < len(detections):
        candidates.append(detections[idx])
    if idx > 0:
        candidates.append(detections[idx - 1])
    best = min(candidates, key=lambda d: abs(d[0] - target_ts))
    if abs(best[0] - target_ts) > max_gap_seconds:
        return None
    return best[2], best[3], best[4], best[5]

def group_into_episodes(timestamps_sorted, gap_tolerance):
    """Group a sorted list of timestamps into (start, end) episodes where
    consecutive timestamps are no more than gap_tolerance seconds apart."""
    if not timestamps_sorted:
        return []
    episodes = []
    ep_start = ep_end = timestamps_sorted[0]
    for ts in timestamps_sorted[1:]:
        if ts - ep_end <= gap_tolerance:
            ep_end = ts
        else:
            episodes.append((ep_start, ep_end))
            ep_start = ep_end = ts
    episodes.append((ep_start, ep_end))
    return episodes

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

    viol_text = f"{label}" if pid is None else f"{label} | Person {pid}"
    (tw, th), _ = cv2.getTextSize(viol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    lx = 10 if x1 is None else int(x1)
    ly = 40
    cv2.rectangle(frame, (lx, ly), (lx + tw + 6, ly + th + 6), COLOR_LABEL_BG, -1)
    cv2.putText(frame, viol_text, (lx + 3, ly + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_LABEL_TEXT, 2)

    dur_text = f"Duration: {int(duration)} seconds ({duration / 60:.1f} minutes)"
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

def process_phone_usage(people, polys):
    """Reads Agent 1's raw phone_detected rows (person_id is always NULL
    there) and attributes each one to whichever person's FACE it's closest
    to (within PHONE_FACE_PROXIMITY_RATIO face-heights and a recent-enough
    face reading) - not just whichever person's whole body bbox it happens
    to fall inside, since that would also count a phone sitting untouched
    on the desk in front of someone. Groups attributed timestamps per
    person into episodes and logs any long enough to count as real usage."""
    conn = sqlite3.connect(DATABASE_PATH)
    phone_rows = conn.execute("""
        SELECT timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events WHERE event_type = 'phone_detected'
        ORDER BY timestamp
    """).fetchall()
    conn.close()

    if not phone_rows:
        print("[INFO] No phone_detected events found - skipping phone usage check.\n")
        return

    faces_by_person = load_faces_by_person()
    if not faces_by_person:
        print("[INFO] No face_detected events found - can't confirm phone-near-face "
              "proximity for anyone, skipping phone usage check.\n")
        return

    attributed = {}  # person_id -> list of timestamps attributed to them
    for ts, px1, py1, px2, py2 in phone_rows:
        best_pid, best_ratio = None, None
        for pid, faces in faces_by_person.items():
            face_box = nearest_face_bbox(faces, ts)
            if face_box is None:
                continue
            ratio = phone_face_distance_ratio((px1, py1, px2, py2), face_box)
            if ratio <= PHONE_FACE_PROXIMITY_RATIO:
                if best_ratio is None or ratio < best_ratio:
                    best_ratio, best_pid = ratio, pid
        if best_pid is not None:
            attributed.setdefault(best_pid, []).append(ts)

    found_any = False
    for pid, timestamps in attributed.items():
        timestamps.sort()
        for ep_start, ep_end in group_into_episodes(timestamps, PHONE_EPISODE_GAP_SECONDS):
            duration = ep_end - ep_start
            if duration < PHONE_USAGE_MIN_SECONDS:
                continue
            found_any = True
            mid_ts = (ep_start + ep_end) / 2
            bbox = nearest_detection_bbox(people.get(pid, []), mid_ts)
            x1, y1, x2, y2 = bbox if bbox else (None, None, None, None)

            print(f"  [VIOLATION] Person {pid} used their phone for {duration:.0f} seconds")
            frame_img = grab_frame(mid_ts)
            crop = annotate_and_save(frame_img, pid, x1, y1, x2, y2, None, polys,
                                      "PHONE_USAGE", duration, f"{int(mid_ts)}_phone")
            # No VLM call - the phone detector already confirmed this
            # deterministically, LLaVA re-classifying "is this a phone"
            # would add latency without adding information.
            log_violation(pid, "violation_phone_usage", ep_start, duration,
                          crop, None, x1, y1, x2, y2, None, None)
            print(f"  [DB] Logged (deterministic - phone detector, no VLM call)")

    if not found_any:
        print("[INFO] No phone usage episode reached the minimum duration.\n")

def process_sleeping(polys):
    """Reads Agent 1's sleeping_detected rows (already person-attributed,
    unlike phone_detected), groups them into episodes per person, and logs
    any episode meeting the minimum duration."""
    conn = sqlite3.connect(DATABASE_PATH)
    sleep_rows = conn.execute("""
        SELECT timestamp, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events WHERE event_type = 'sleeping_detected'
        ORDER BY person_id, timestamp
    """).fetchall()
    conn.close()

    if not sleep_rows:
        print("[INFO] No sleeping_detected events found - skipping sleeping check.\n")
        return

    by_person = {}
    for ts, pid, x1, y1, x2, y2 in sleep_rows:
        by_person.setdefault(pid, []).append((ts, x1, y1, x2, y2))

    found_any = False
    for pid, rows in by_person.items():
        timestamps = [r[0] for r in rows]
        for ep_start, ep_end in group_into_episodes(timestamps, SLEEP_EPISODE_GAP_SECONDS):
            duration = ep_end - ep_start
            if duration < SLEEP_MIN_SECONDS:
                continue
            found_any = True
            mid_ts = (ep_start + ep_end) / 2
            closest = min(rows, key=lambda r: abs(r[0] - mid_ts))
            _, x1, y1, x2, y2 = closest

            print(f"  [VIOLATION] Person {pid} was sleeping for {duration:.0f} seconds")
            frame_img = grab_frame(mid_ts)
            crop = annotate_and_save(frame_img, pid, x1, y1, x2, y2, None, polys,
                                      "SLEEPING", duration, f"{int(mid_ts)}_sleep")
            # No VLM call - Agent 1's pose heuristic (head_drop_ratio +
            # face-visibility) already confirmed this deterministically.
            log_violation(pid, "violation_sleeping", ep_start, duration,
                          crop, None, x1, y1, x2, y2, None, None)
            print(f"  [DB] Logged (deterministic - pose heuristic, no VLM call)")

    if not found_any:
        print("[INFO] No sleeping episode reached the minimum duration.\n")

def process_empty_room(polys, video_duration):
    """A room-wide violation, distinct from any individual person's AFK: a
    stretch of time where NOBODY was detected in frame at all - built from
    the union of every person's 'detected' timestamps globally, not any
    one person's zone logic. Checks both gaps BETWEEN sightings and a
    trailing gap if the room is still empty when the video ends."""
    conn = sqlite3.connect(DATABASE_PATH)
    rows = conn.execute("""
        SELECT DISTINCT timestamp FROM events WHERE event_type = 'detected'
        ORDER BY timestamp
    """).fetchall()
    conn.close()

    timestamps = [r[0] for r in rows]
    if not timestamps:
        print("[INFO] No detections at all in this video - skipping empty-room check.\n")
        return

    found_any = False

    # Gaps BETWEEN two real sightings (room went empty, then someone came back)
    for prev_ts, next_ts in zip(timestamps, timestamps[1:]):
        gap = next_ts - prev_ts
        if gap >= EMPTY_ROOM_MIN_SECONDS:
            found_any = True
            mid_ts = prev_ts + gap / 2
            print(f"  [VIOLATION] Room was completely empty for {gap:.0f} seconds "
                  f"(from {format_timestamp(prev_ts)} to {format_timestamp(next_ts)})")
            frame_img = grab_frame(mid_ts)
            crop = annotate_and_save(frame_img, None, None, None, None, None,
                                      None, polys, "EMPTY_ROOM", gap, f"{int(mid_ts)}_empty_room")
            log_violation(None, "violation_empty_room", prev_ts, gap,
                          crop, None, None, None, None, None, None, None)
            print(f"  [DB] Logged (deterministic - no VLM call)")

    # Trailing gap: everyone left and the video ended before anyone returned
    last_ts = timestamps[-1]
    tail_gap = video_duration - last_ts
    if tail_gap >= EMPTY_ROOM_MIN_SECONDS:
        found_any = True
        mid_ts = min(last_ts + tail_gap / 2, video_duration - 1)
        print(f"  [VIOLATION] Room was empty for the final {tail_gap:.0f} seconds of the video "
              f"(from {format_timestamp(last_ts)} onward)")
        frame_img = grab_frame(mid_ts)
        crop = annotate_and_save(frame_img, None, None, None, None, None,
                                  None, polys, "EMPTY_ROOM", tail_gap, f"{int(mid_ts)}_empty_room_end")
        log_violation(None, "violation_empty_room", last_ts, tail_gap,
                      crop, None, None, None, None, None, None, None)
        print(f"  [DB] Logged (deterministic - no VLM call)")

    if not found_any:
        print("[INFO] No empty-room episode reached the minimum duration.\n")

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
    print(f"[INFO] AFK: {AFK_THRESHOLD}s | Zone: {ZONE_THRESHOLD}s | Warmup: {WARMUP_SECONDS}s")
    print(f"[INFO] Phone usage: {PHONE_USAGE_MIN_SECONDS}s min | Sleeping: {SLEEP_MIN_SECONDS}s min\n")

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

    # ── ZONE / AFK / LEFT-FRAME (unchanged logic, still routes through Agent 3) ──
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
            print(f"  [VIOLATION] Person {pid} left at {format_timestamp(last_ts)} "
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

    # ── PHONE USAGE (new, deterministic, no VLM) ──────────────────────────────
    print("── Phone usage check ──")
    process_phone_usage(people, polys)

    # ── SLEEPING (new, deterministic, no VLM) ─────────────────────────────────
    print("── Sleeping check ──")
    process_sleeping(polys)

    # ── EMPTY ROOM (new, deterministic, no VLM, room-wide not per-person) ─────
    print("── Empty room check ──")
    process_empty_room(polys, video_duration)

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
        print(f"  Person {v[0]} | {v[1]} | Duration {v[2]:.0f} seconds | At {format_timestamp(v[3])} in video")

if __name__ == "__main__":
    main()