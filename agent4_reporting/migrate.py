# migrate_general_violations.py
#
# One-time cleanup: existing violation_* rows with person_id IS NULL get
# reassigned to person_id = 0, a reserved sentinel meaning "general /
# whole-room, not a specific tracked person" - never a real person_id, since
# PersonGallery.next_person_id() (SELECT MAX(person_id)+1 FROM person_gallery)
# always starts real people at 1.
#
# Run this ONCE against your existing database to clean up rows already
# written before this fix existed. New rows going forward still need
# whatever writes them (Agent 2's rule engine) updated to use 0 instead of
# NULL directly - this script only fixes what's already there.

import sqlite3
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from config import DATABASE_PATH

GENERAL_PERSON_ID = 0

conn = sqlite3.connect(DATABASE_PATH, timeout=30)
cur = conn.cursor()

# Preview first - show exactly what will change before touching anything.
rows = cur.execute("""
    SELECT id, event_type, timestamp FROM events
    WHERE person_id IS NULL AND event_type LIKE 'violation_%'
""").fetchall()

print(f"[INFO] Found {len(rows)} violation row(s) with person_id IS NULL:")
for row_id, etype, ts in rows:
    print(f"  - row id={row_id}  type={etype}  timestamp={ts}")

if not rows:
    print("[INFO] Nothing to migrate.")
else:
    confirm = input(f"\nReassign these {len(rows)} row(s) to person_id={GENERAL_PERSON_ID}? [y/N] ")
    if confirm.strip().lower() == "y":
        cur.execute("""
            UPDATE events SET person_id = ?
            WHERE person_id IS NULL AND event_type LIKE 'violation_%'
        """, (GENERAL_PERSON_ID,))
        conn.commit()
        print(f"[SUCCESS] Updated {cur.rowcount} row(s).")
    else:
        print("[INFO] Cancelled - no changes made.")

conn.close()