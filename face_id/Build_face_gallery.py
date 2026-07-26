# face_id/build_face_gallery.py
#
# One-time (or re-run-whenever-you-add-photos) enrollment step.
#
# Supports TWO folder layouts, auto-detected per entry:
#
#   1) Flat, name-prefixed files (this is what's already sitting in
#      agent1_tracking/known_faces/ - e.g. "Abdalrahman_10.png",
#      "Omar_2.jpg"). The name is taken as everything before the final
#      "_<number>" in the filename, so you don't need to reorganize photos
#      you've already collected.
#
#   2) Subfolder-per-person, if you prefer it for new people going forward:
#        enrollment/Sara/front.jpg, enrollment/Sara/side.jpg, ...
#
# Builds a named face gallery in the shared sqlite DB (table: named_faces).
# This does NOT touch agent1.py, agent1's own person_gallery table, or the
# events table - it's a completely separate identity layer that sits on
# top of what agent1 already produces.
#
# Run this whenever you add/change enrollment photos:
#   python face_id/build_face_gallery.py

import os
import re
import sys
import sqlite3
import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import DATABASE_PATH

# Points at your existing photos by default. If you later want to enroll
# new people via subfolders instead, either add subfolders in here too
# (both layouts are supported side by side) or point this at a separate
# "enrollment" dir.
ENROLLMENT_DIR = os.path.join(ROOT_DIR, "agent1_tracking", "known_faces")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FLAT_NAME_PATTERN = re.compile(r"^(.*?)_\d+$")  # "Abdalrahman_10" -> "Abdalrahman"

# ctx_id=-1 -> CPU. Set to 0 if you have onnxruntime-gpu + a CUDA GPU
# available - buffalo_l runs noticeably faster on GPU, but CPU is fine for
# a one-time enrollment pass over a handful of photos.
CTX_ID = -1
DET_SIZE = (640, 640)


def get_conn():
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS named_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source_image TEXT
        )
    """)
    conn.commit()
    return conn


def main():
    if not os.path.isdir(ENROLLMENT_DIR):
        print(f"[ERROR] No enrollment folder found at {ENROLLMENT_DIR}")
        print("        Create it with one subfolder per person, e.g.:")
        print("        enrollment/Mahmoud/front.jpg, enrollment/Mahmoud/side_left.jpg ...")
        sys.exit(1)

    print("[INFO] Loading InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    conn = get_conn()
    cur = conn.cursor()

    # name -> list of (full_path, display_filename)
    people = {}

    for entry in sorted(os.listdir(ENROLLMENT_DIR)):
        entry_path = os.path.join(ENROLLMENT_DIR, entry)

        if os.path.isdir(entry_path):
            # Layout 2: subfolder-per-person
            for img_name in sorted(os.listdir(entry_path)):
                if img_name.lower().endswith(IMAGE_EXTS):
                    people.setdefault(entry, []).append(
                        (os.path.join(entry_path, img_name), img_name)
                    )
        elif entry.lower().endswith(IMAGE_EXTS):
            # Layout 1: flat "Name_number.ext" files
            stem = os.path.splitext(entry)[0]
            match = FLAT_NAME_PATTERN.match(stem)
            name = match.group(1) if match else stem
            people.setdefault(name, []).append((entry_path, entry))

    if not people:
        print(f"[ERROR] No photos or person subfolders found inside {ENROLLMENT_DIR}")
        sys.exit(1)

    # Wipe and rebuild every run - simplest way to avoid stale/duplicate
    # entries when you add, remove, or replace photos. Cheap since
    # enrollment sets are small.
    cur.execute("DELETE FROM named_faces")
    conn.commit()

    total_saved, total_skipped = 0, 0

    for name, images in sorted(people.items()):
        saved_for_person = 0
        for img_path, img_name in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] {name}/{img_name}: couldn't read image, skipping.")
                total_skipped += 1
                continue

            faces = app.get(img)
            if not faces:
                print(f"[WARN] {name}/{img_name}: no face detected, skipping "
                      f"(too extreme an angle, too small, or bad lighting).")
                total_skipped += 1
                continue

            # If more than one face got picked up in the photo, assume the
            # largest bbox is the intended subject.
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            vec = face.normed_embedding.astype(np.float32)  # already L2-normalized

            cur.execute(
                "INSERT INTO named_faces (name, embedding, source_image) VALUES (?, ?, ?)",
                (name, vec.tobytes(), img_name)
            )
            saved_for_person += 1
            total_saved += 1

        conn.commit()
        print(f"[OK] {name}: {saved_for_person}/{len(images)} photo(s) enrolled.")

    conn.close()
    print(f"\n[DONE] {total_saved} face embedding(s) saved across {len(people)} "
          f"people ({total_skipped} photo(s) skipped).")


if __name__ == "__main__":
    main()