# database_setup.py
# Run this ONCE to create the shared SQLite database and events table.
# All 4 agents read and write from this single file.
#
# person_id is TEXT — stores names from face recognition (e.g. "Abdalrahman")

import sqlite3
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import DATABASE_PATH

os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS events")

cursor.execute("""
    CREATE TABLE events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        REAL,        -- Video time in seconds when this event occurred
        person_id        TEXT,        -- Person name from face recognition (Agent 1)
        event_type       TEXT,        -- 'detected', 'violation_away', 'violation_phone'
        duration_seconds REAL,        -- How long the violation lasted (Agent 2 fills this)
        confidence       REAL,        -- YOLO detection confidence score
        crop_path        TEXT,        -- Path to saved crop image (Agent 2 fills this)
        vlm_summary      TEXT,        -- LLaVA description (Agent 3 fills this)
        bbox_x1          INTEGER,     -- Bounding box coordinates (Agent 1 fills these)
        bbox_y1          INTEGER,
        bbox_x2          INTEGER,
        bbox_y2          INTEGER,
        zone_id          INTEGER,     -- Zone the person was detected in (Agent 2)
        zone_name        TEXT         -- Zone label e.g. "Desk 1" (Agent 2)
    )
""")

conn.commit()
conn.close()

print("[SUCCESS] sentinel.db created with events table ready.")
print(f"[INFO] Location: {DATABASE_PATH}")
print("[INFO] person_id is TEXT — stores names from face recognition.")