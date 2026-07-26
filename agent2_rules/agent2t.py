# agent2_rules/agent2.py
# Agent 2: Violation Detector
#
# Detects:
#   - violation_afk
#   - violation_loitering
#   - violation_unauth_zone
#   - violation_left_frame
#   - violation_phone_usage
#   - violation_sleeping
#
# Main fix:
#   AFK is based on the away-from-home episode duration, not on the current
#   zone label alone. This prevents the "left desk -> later loitered elsewhere
#   -> got classified as loitering instead of AFK" problem.
#
# Phone usage is sustained phone-to-face proximity.
# Sleeping is sustained sleeping detections from Agent 1.

import os
import sys
import sqlite3
import cv2
import numpy as np
import json
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'agent3_vlm'))

import agent3
from config import (
    VIDEO_PATH,
    DATABASE_PATH,
    PHONE_MIN_PERSON_OVERLAP,
    PHONE_FACE_MIN_SCORE,
    PHONE_FACE_MAX_CENTER_DIST_RATIO,
    PHONE_FACE_RECENCY_SECONDS,
    PHONE_MIN_CONSECUTIVE_SECONDS,
    PHONE_GAP_SECONDS,
    SLEEP_MIN_DURATION_SECONDS,
    SLEEP_GAP_SECONDS,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
WARMUP_SECONDS = 20
AFK_THRESHOLD = 60
ZONE_THRESHOLD = 20
LOITERING_THRESHOLD = ZONE_THRESHOLD

CROPS_DIR = os.path.join(ROOT_DIR, "violations")
os.makedirs(CROPS_DIR, exist_ok=True)

# ── COLORS (BGR) ─────────────────────────────────────────────────────────────
COLOR_HOME_ZONE = (0, 0, 255)
COLOR_OTHER_ZONE = (255, 100, 0)
COLOR_PERSON = (0, 255, 0)
COLOR_FACE = (255, 180, 0)
COLOR_PHONE = (0, 255, 255)
COLOR_LABEL_BG = (0, 0, 200)
COLOR_LABEL_TEXT = (255, 255, 255)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def load_zones():
    path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
    with open(path, "r", encoding="utf-8") as f:
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


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def overlap_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return float(inter_area / inner_area)


def draw_box(frame, box, color, label=None, thickness=2, font_scale=0.45):
    if frame is None or box is None:
        return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + 6), color, -1)
        cv2.putText(frame, label, (x1 + 2, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)


def annotate_and_save(frame, pid, home_zone, polys, label, duration,
                      suffix="", person_box=None, face_box=None, phone_box=None):
    if frame is None:
        return None

    fh, fw = frame.shape[:2]

    for i, poly in enumerate(polys):
        color = COLOR_HOME_ZONE if i == home_zone else COLOR_OTHER_ZONE
        thickness = 3 if i == home_zone else 2
        cv2.polylines(frame, [poly.astype(int)], True, color, thickness)
        cx_z = int(np.mean(poly[:, 0]))
        cy_z = int(np.mean(poly[:, 1]))
        zone_label = f"Zone {i}" + (" (YOURS)" if i == home_zone else "")
        cv2.putText(frame, zone_label, (cx_z - 40, cy_z),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    draw_box(frame, person_box, COLOR_PERSON, label=f"P{pid}" if person_box else None,
             thickness=2, font_scale=0.4)
    draw_box(frame, face_box, COLOR_FACE, label="FACE", thickness=2, font_scale=0.4)
    draw_box(frame, phone_box, COLOR_PHONE, label="PHONE", thickness=2, font_scale=0.4)

    viol_text = f"{label} | Person {pid}"
    (tw, th), _ = cv2.getTextSize(viol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    lx = 10 if person_box is None else int(person_box[0])
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
    conn.execute(
        """
        INSERT INTO events
            (timestamp, person_id, event_type, duration_seconds,
             crop_path, vlm_summary, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             zone_id, zone_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            round(ts, 3),
            pid,
            event_type,
            round(duration, 2),
            crop,
            vlm,
            int(x1) if x1 is not None else None,
            int(y1) if y1 is not None else None,
            int(x2) if x2 is not None else None,
            int(y2) if y2 is not None else None,
            zone_id,
            zone_name,
        ),
    )
    conn.commit()
    conn.close()


def enters_afk_before_return_home(future_dets, outside_start, afk_threshold, polys, home_zone):
    """
    True if the person remains away from home long enough to become AFK before
    any future detection returns them to their home zone.
    """
    for det in future_dets:
        ts, _, x1, y1, x2, y2 = det
        cx, cy = box_center((x1, y1, x2, y2))
        curr_zone = get_zone(cx, cy, polys)
        if curr_zone == home_zone:
            return False
        if ts - outside_start >= afk_threshold:
            return True
    return False


def phone_face_score(phone_box, face_box):
    if phone_box is None or face_box is None:
        return 0.0

    px, py = box_center(phone_box)
    fx, fy = box_center(face_box)
    fw, fh = box_size(face_box)
    face_scale = max(fw, fh, 1.0)

    dist = float(np.hypot(px - fx, py - fy))
    max_dist = PHONE_FACE_MAX_CENTER_DIST_RATIO * face_scale
    dist_score = max(0.0, 1.0 - (dist / max_dist))
    iou_score = overlap_ratio(phone_box, face_box)

    return max(dist_score, iou_score)


def assign_phone_to_person(phone_box, person_boxes, face_boxes, face_cache, ts):
    """
    Returns (pid, face_box, person_box, score) for the best phone owner.
    The phone must overlap the person's box and be close to that person's face.
    """
    best = None

    for pid, person_box in person_boxes.items():
        person_overlap = overlap_ratio(phone_box, person_box)
        if person_overlap < PHONE_MIN_PERSON_OVERLAP:
            continue

        face_box = face_boxes.get(pid)
        if face_box is None and pid in face_cache:
            face_ts, cached_face = face_cache[pid]
            if ts - face_ts <= PHONE_FACE_RECENCY_SECONDS:
                face_box = cached_face

        score = phone_face_score(phone_box, face_box)
        if score < PHONE_FACE_MIN_SCORE:
            continue

        candidate = (score, pid, face_box, person_box)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None, None, None, 0.0

    score, pid, face_box, person_box = best
    return pid, face_box, person_box, score


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n=== AGENT 2 STARTING: VIOLATION DETECTION ===")

    conn = sqlite3.connect(DATABASE_PATH)
    deleted = conn.execute("DELETE FROM events WHERE event_type LIKE 'violation_%'").rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"[INFO] Cleared {deleted} previous violation(s) from DB.")

    zones_data = load_zones()

    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
    cap.release()

    polys = build_polys(zones_data, w, h)
    print(f"[INFO] {len(polys)} zones | Video duration: {video_duration:.0f} seconds")
    print(f"[INFO] AFK: {AFK_THRESHOLD}s | Zone/Loiter: {ZONE_THRESHOLD}s | Warmup: {WARMUP_SECONDS}s")
    print(f"[INFO] Phone: {PHONE_MIN_CONSECUTIVE_SECONDS}s near face | Sleep: {SLEEP_MIN_DURATION_SECONDS}s sustained\n")

    conn = sqlite3.connect(DATABASE_PATH)
    rows = conn.execute(
        """
        SELECT timestamp, person_id, event_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence
        FROM events
        WHERE event_type IN ('detected', 'face_detected', 'phone_detected', 'sleeping_detected')
        ORDER BY timestamp, person_id
        """
    ).fetchall()
    conn.close()

    frames = defaultdict(lambda: {
        "persons": {},
        "faces": {},
        "phones": [],
        "sleeping": {},
    })

    detections_by_person = defaultdict(list)

    for ts, pid, event_type, x1, y1, x2, y2, conf in rows:
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        box = (int(x1), int(y1), int(x2), int(y2))
        ts = float(ts)

        if event_type == "detected" and pid is not None:
            pid = int(pid)
            frames[ts]["persons"][pid] = box
            detections_by_person[pid].append((ts, box))
        elif event_type == "face_detected" and pid is not None:
            frames[ts]["faces"][int(pid)] = box
        elif event_type == "phone_detected":
            frames[ts]["phones"].append(box)
        elif event_type == "sleeping_detected" and pid is not None:
            frames[ts]["sleeping"].setdefault(int(pid), []).append(box)

    if not detections_by_person:
        print("[WARN] No person detections found in DB.")
        return

    print(f"[INFO] People found in DB: {sorted(detections_by_person.keys())}\n")

    timestamps = sorted(frames.keys())
    violations_written = 0

    # --- PASS 1: zone / AFK / loitering / unauthorized zone ---
    for pid, detections in detections_by_person.items():
        print(f"── Person {pid} ({len(detections)} detections) ──")

        zone_votes = {}
        for ts, box in detections:
            if ts > WARMUP_SECONDS:
                break
            cx, cy = box_center(box)
            z = get_zone(cx, cy, polys)
            if z is not None:
                zone_votes[z] = zone_votes.get(z, 0) + 1

        if not zone_votes:
            print("  [WARN] No warmup zone data, skipping.\n")
            continue

        home_zone = max(zone_votes, key=zone_votes.get)
        print(f"  Home zone: Zone {home_zone}")

        outside_start = None
        outside_start_det = None
        afk_logged = False
        zone_logged = False

        for i, (ts, box) in enumerate(detections):
            if ts <= WARMUP_SECONDS:
                continue

            cx, cy = box_center(box)
            curr_zone = get_zone(cx, cy, polys)

            if curr_zone == home_zone:
                outside_start = None
                outside_start_det = None
                afk_logged = False
                zone_logged = False
                continue

            if outside_start is None:
                outside_start = ts
                outside_start_det = box

            time_away = ts - outside_start
            future_dets = detections[i:]

            # AFK takes priority for the whole episode.
            if not afk_logged and time_away >= AFK_THRESHOLD:
                if enters_afk_before_return_home(future_dets, outside_start, AFK_THRESHOLD, polys, home_zone):
                    _, _, ox1, oy1, ox2, oy2 = outside_start_det
                    print(f"  [VIOLATION] Person {pid} was away from home long enough to be AFK ({time_away:.0f}s)")

                    mid_ts = outside_start + (AFK_THRESHOLD / 2.0)
                    f1 = grab_frame(mid_ts)
                    crop1 = annotate_and_save(
                        f1, pid, home_zone, polys, "AFK", time_away,
                        f"{int(mid_ts)}_empty_chair",
                        person_box=None,
                        face_box=None,
                        phone_box=None,
                    )
                    vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
                    log_violation(
                        pid, "violation_afk", outside_start, time_away,
                        crop1, vlm, ox1, oy1, ox2, oy2, home_zone, f"Zone {home_zone}"
                    )
                    print(f"  [DB] Logged AFK | VLM says: {vlm}")
                    violations_written += 1
                    afk_logged = True
                    zone_logged = True
                continue

            # Shorter away-from-home episodes.
            if time_away >= LOITERING_THRESHOLD and not zone_logged and not afk_logged:
                if enters_afk_before_return_home(future_dets, outside_start, AFK_THRESHOLD, polys, home_zone):
                    continue

                _, _, ox1, oy1, ox2, oy2 = outside_start_det
                if curr_zone is None:
                    print(f"  [VIOLATION] Person {pid} is loitering outside zones for {time_away:.0f}s")
                    label = "LOITERING"
                    event_type = "violation_loitering"
                    zone_id = None
                    zone_name = "Open space"
                else:
                    print(f"  [VIOLATION] Person {pid} is in unauthorized Zone {curr_zone} for {time_away:.0f}s")
                    label = "UNAUTH"
                    event_type = "violation_unauth_zone"
                    zone_id = curr_zone
                    zone_name = f"Zone {curr_zone}"

                f1 = grab_frame(ts)
                crop1 = annotate_and_save(
                    f1, pid, home_zone, polys, label, time_away,
                    f"{int(ts)}_{label.lower()}",
                    person_box=box,
                )
                vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
                log_violation(
                    pid, event_type, outside_start, time_away,
                    crop1, vlm, ox1, oy1, ox2, oy2,
                    zone_id, zone_name
                )
                print(f"  [DB] Logged | VLM says: {vlm}")
                violations_written += 1
                zone_logged = True

        # PHASE 2: LEFT AND NEVER RETURNED
        last_ts = detections[-1][0]
        time_since_last = video_duration - last_ts
        if time_since_last >= AFK_THRESHOLD and not afk_logged and not zone_logged:
            _, last_box = detections[-1]
            x1, y1, x2, y2 = last_box
            print(f"  [VIOLATION] Person {pid} left at {last_ts:.0f}s and never returned ({time_since_last:.0f}s away)")

            f1 = grab_frame(max(0.0, video_duration - 5))
            crop1 = annotate_and_save(
                f1, pid, home_zone, polys, "LEFT", time_since_last,
                f"{int(video_duration)}_empty",
                person_box=None,
            )
            vlm = agent3.run_agent3_auditor(crop1, pid, home_zone)
            log_violation(
                pid, "violation_left_frame", last_ts, time_since_last,
                crop1, vlm, x1, y1, x2, y2, home_zone, f"Zone {home_zone}"
            )
            print(f"  [DB] Logged LEFT | VLM says: {vlm}")
            violations_written += 1

        print()

    # --- PASS 3: phone usage episodes ---
    face_cache = {}  # pid -> (timestamp, face_box)
    phone_state = defaultdict(lambda: {
        "start_ts": None,
        "last_ts": None,
        "logged": False,
        "last_phone_box": None,
        "last_face_box": None,
        "last_person_box": None,
        "home_zone": None,
    })

    # Reuse each person's home zone from the zone pass.
    home_zone_by_pid = {}
    for pid, detections in detections_by_person.items():
        zone_votes = {}
        for ts, box in detections:
            if ts > WARMUP_SECONDS:
                break
            cx, cy = box_center(box)
            z = get_zone(cx, cy, polys)
            if z is not None:
                zone_votes[z] = zone_votes.get(z, 0) + 1
        if zone_votes:
            home_zone_by_pid[pid] = max(zone_votes, key=zone_votes.get)

    for ts in timestamps:
        frame = frames[ts]
        current_faces = frame["faces"]
        current_persons = frame["persons"]
        current_phones = frame["phones"]

        for pid, face_box in current_faces.items():
            face_cache[pid] = (ts, face_box)

        assigned_this_ts = set()

        for phone_box in current_phones:
            assigned_pid, assigned_face_box, assigned_person_box, score = assign_phone_to_person(
                phone_box, current_persons, current_faces, face_cache, ts
            )
            if assigned_pid is None:
                continue

            assigned_this_ts.add(assigned_pid)
            pst = phone_state[assigned_pid]
            pst["home_zone"] = home_zone_by_pid.get(assigned_pid)

            if pst["start_ts"] is None or (pst["last_ts"] is not None and ts - pst["last_ts"] > PHONE_GAP_SECONDS):
                pst["start_ts"] = ts
                pst["logged"] = False

            pst["last_ts"] = ts
            pst["last_phone_box"] = phone_box
            pst["last_face_box"] = assigned_face_box
            pst["last_person_box"] = assigned_person_box

            sustained = ts - pst["start_ts"]
            if not pst["logged"] and sustained >= PHONE_MIN_CONSECUTIVE_SECONDS:
                home_zone = pst["home_zone"]
                if home_zone is None:
                    continue

                print(f"  [VIOLATION] Person {assigned_pid} used a phone near the face for {sustained:.0f}s")
                f1 = grab_frame(ts)
                crop1 = annotate_and_save(
                    f1, assigned_pid, home_zone, polys, "PHONE", sustained,
                    f"{int(ts)}_phone",
                    person_box=assigned_person_box,
                    face_box=assigned_face_box,
                    phone_box=phone_box,
                )
                log_violation(
                    assigned_pid, "violation_phone_usage", pst["start_ts"], sustained,
                    crop1, "PHONE_USAGE",
                    assigned_person_box[0], assigned_person_box[1], assigned_person_box[2], assigned_person_box[3],
                    home_zone, f"Zone {home_zone}"
                )
                print("  [DB] Logged PHONE_USAGE")
                violations_written += 1
                pst["logged"] = True

        for pid in list(phone_state.keys()):
            pst = phone_state[pid]
            if pst["last_ts"] is None:
                continue
            if pid not in assigned_this_ts and ts - pst["last_ts"] > PHONE_GAP_SECONDS:
                pst["start_ts"] = None
                pst["last_ts"] = None
                pst["logged"] = False
                pst["last_phone_box"] = None
                pst["last_face_box"] = None
                pst["last_person_box"] = None
                pst["home_zone"] = None

    # --- PASS 4: sleeping episodes ---
    sleep_state = defaultdict(lambda: {
        "start_ts": None,
        "last_ts": None,
        "logged": False,
        "last_box": None,
    })

    for pid, detections in detections_by_person.items():
        home_zone = home_zone_by_pid.get(pid)
        if home_zone is None:
            continue

        for ts in timestamps:
            frame = frames[ts]
            sleeping_boxes = frame["sleeping"].get(pid, [])
            person_box = frame["persons"].get(pid)
            sstate = sleep_state[pid]

            if sleeping_boxes:
                sbox = sleeping_boxes[0]
                if sstate["start_ts"] is None or (sstate["last_ts"] is not None and ts - sstate["last_ts"] > SLEEP_GAP_SECONDS):
                    sstate["start_ts"] = ts
                    sstate["logged"] = False

                sstate["last_ts"] = ts
                sstate["last_box"] = sbox

                sustained_sleep = ts - sstate["start_ts"]
                if not sstate["logged"] and sustained_sleep >= SLEEP_MIN_DURATION_SECONDS:
                    print(f"  [VIOLATION] Person {pid} has been sleeping for {sustained_sleep:.0f}s")
                    f1 = grab_frame(ts)
                    crop1 = annotate_and_save(
                        f1, pid, home_zone, polys, "SLEEP", sustained_sleep,
                        f"{int(ts)}_sleep",
                        person_box=person_box,
                    )
                    log_violation(
                        pid, "violation_sleeping", sstate["start_ts"], sustained_sleep,
                        crop1, "SLEEPING",
                        person_box[0], person_box[1], person_box[2], person_box[3],
                        home_zone, f"Zone {home_zone}"
                    )
                    print("  [DB] Logged SLEEPING")
                    violations_written += 1
                    sstate["logged"] = True
            else:
                if sstate["last_ts"] is not None and ts - sstate["last_ts"] > SLEEP_GAP_SECONDS:
                    sstate["start_ts"] = None
                    sstate["last_ts"] = None
                    sstate["logged"] = False
                    sstate["last_box"] = None

    # SUMMARY
    conn = sqlite3.connect(DATABASE_PATH)
    violations = conn.execute(
        """
        SELECT person_id, event_type, duration_seconds, timestamp
        FROM events WHERE event_type LIKE 'violation_%'
        ORDER BY timestamp
        """
    ).fetchall()
    conn.close()

    print("########################################")
    print("  AGENT 2 COMPLETE: VIOLATIONS LOGGED  ")
    print("########################################")
    print(f"Total violations found: {len(violations)}")
    for v in violations:
        print(f"  Person {v[0]} | {v[1]} | Duration {v[2]:.0f}s | At {v[3]:.0f}s in video")


if __name__ == "__main__":
    main()
