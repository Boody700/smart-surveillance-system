# face_id/identify_people.py
#
# Reads agent1's already-logged face_detected events (timestamp, person_id,
# bbox) straight from the DB, re-crops those regions from the video,
# embeds them with InsightFace (buffalo_l - same model as enrollment, so
# embeddings are directly comparable), and matches against the
# named_faces gallery built by build_face_gallery.py.
#
# Does NOT re-run face detection over the whole video, and does NOT touch
# agent1.py or agent1's own tables - it purely reads what agent1 already
# wrote, and writes its own separate `person_identities` table mapping
# agent1's numeric person_id -> a human name.
#
# Run this after agent1 (and after build_face_gallery.py has been run at
# least once):
#   python face_id/identify_people.py

import os
import sys
import sqlite3
import cv2
import numpy as np
from collections import defaultdict
from insightface.app import FaceAnalysis

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import DATABASE_PATH, VIDEO_PATH

CTX_ID = -1
DET_SIZE = (320, 320)  # crops are already tight around one face - no need for the full 640 enrollment det_size

# How many sampled face_detected rows to actually check per person_id.
# agent1 already throttles face detection (FACE_DETECT_EVERY_N_FRAMES), so
# a video-length session can still leave hundreds of rows per person -
# capping keeps this fast without needing to check every single one.
MAX_SAMPLES_PER_PERSON = 40

# Cosine similarity floor for a single crop-to-gallery match to count at
# all. buffalo_l/ArcFace on aligned faces: same person is typically
# 0.5-0.7+, different people usually well under 0.4 - but sanity check
# this against your own footage/gallery (print raw sims first) before
# trusting it blindly, camera quality shifts this a lot.
MATCH_THRESHOLD = 0.45

# Of the samples that clear MATCH_THRESHOLD against a given name, this
# fraction must agree on the SAME name before we commit to it - stops one
# lucky/unlucky frame from mislabeling someone.
MIN_AGREEMENT_RATIO = 0.5
MIN_VOTES = 3  # also require at least this many matching samples, not just a ratio

# Padding around agent1's face bbox before re-detecting/aligning - agent1's
# box is already tight, a little slack helps insightface's own detector
# get clean landmarks.
CROP_PAD_RATIO = 0.25


def get_conn():
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_identities (
            person_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            confidence REAL,
            num_votes INTEGER,
            num_samples INTEGER
        )
    """)
    conn.commit()
    return conn


def load_gallery(conn):
    rows = conn.execute("SELECT name, embedding FROM named_faces").fetchall()
    gallery = defaultdict(list)
    for name, blob in rows:
        gallery[name].append(np.frombuffer(blob, dtype=np.float32))
    return gallery


def best_gallery_match(vec, gallery):
    """Compare against EVERY stored embedding for every name (not an
    average per name - a front photo and a side photo of the same person
    shouldn't be blended into one centroid, same reasoning as agent1's own
    face-vs-body embedding handling). Returns (best_name, best_sim)."""
    best_name, best_sim = None, -1.0
    for name, vectors in gallery.items():
        for gv in vectors:
            sim = float(np.dot(vec, gv))
            if sim > best_sim:
                best_sim, best_name = sim, name
    return best_name, best_sim


def expand_box(box, pad_ratio, fw, fh):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return (max(0, x1 - px), max(0, y1 - py), min(fw - 1, x2 + px), min(fh - 1, y2 + py))


def sample_rows(rows, max_samples):
    """Evenly spread samples across the whole session instead of just
    taking the first N - a person's earliest appearance might be lower
    quality (further from camera, back turned briefly) than later ones."""
    if len(rows) <= max_samples:
        return rows
    step = len(rows) / max_samples
    return [rows[int(i * step)] for i in range(max_samples)]


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found at {VIDEO_PATH}")
        sys.exit(1)

    conn = get_conn()
    gallery = load_gallery(conn)
    if not gallery:
        print("[ERROR] named_faces gallery is empty - run build_face_gallery.py first.")
        sys.exit(1)
    print(f"[INFO] Loaded gallery: {', '.join(f'{n} ({len(v)})' for n, v in gallery.items())}")

    face_rows = conn.execute("""
        SELECT timestamp, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM events WHERE event_type = 'face_detected'
        ORDER BY person_id, timestamp
    """).fetchall()
    if not face_rows:
        print("[ERROR] No face_detected events in the DB - run agent1 first.")
        sys.exit(1)

    by_person = defaultdict(list)
    for ts, pid, x1, y1, x2, y2 in face_rows:
        by_person[pid].append((ts, x1, y1, x2, y2))

    print("[INFO] Loading InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur = conn.cursor()
    cur.execute("DELETE FROM person_identities")

    for pid, rows in by_person.items():
        samples = sample_rows(rows, MAX_SAMPLES_PER_PERSON)
        votes = defaultdict(int)
        sims_for_winner = defaultdict(list)
        checked = 0

        for ts, x1, y1, x2, y2 in samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if not ok:
                continue

            ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), CROP_PAD_RATIO, fw, fh)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            faces = app.get(crop)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            vec = face.normed_embedding.astype(np.float32)

            checked += 1
            name, sim = best_gallery_match(vec, gallery)
            if name is not None and sim >= MATCH_THRESHOLD:
                votes[name] += 1
                sims_for_winner[name].append(sim)

        if checked == 0:
            print(f"[SKIP] Person {pid}: no usable face crops (couldn't re-detect a face in any sample).")
            continue

        if not votes:
            print(f"[UNKNOWN] Person {pid}: {checked} face(s) checked, none matched the gallery "
                  f"above {MATCH_THRESHOLD:.2f} - probably not enrolled, or angle too extreme.")
            continue

        winner, winner_votes = max(votes.items(), key=lambda kv: kv[1])
        agreement = winner_votes / checked

        if winner_votes < MIN_VOTES or agreement < MIN_AGREEMENT_RATIO:
            print(f"[UNSURE] Person {pid}: best guess '{winner}' only got {winner_votes}/{checked} "
                  f"votes ({agreement:.0%}) - below confidence bar, leaving unnamed.")
            continue

        avg_sim = float(np.mean(sims_for_winner[winner]))
        cur.execute("""
            INSERT INTO person_identities (person_id, name, confidence, num_votes, num_samples)
            VALUES (?, ?, ?, ?, ?)
        """, (pid, winner, round(avg_sim, 4), winner_votes, checked))
        print(f"[NAMED] Person {pid} -> {winner}  "
              f"({winner_votes}/{checked} votes, avg sim {avg_sim:.3f})")

    conn.commit()
    conn.close()
    cap.release()
    print("\n[DONE] person_identities table updated.")


if __name__ == "__main__":
    main()