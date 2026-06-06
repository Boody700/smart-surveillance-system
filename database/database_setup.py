# database_setup.py
# Run this ONCE to create the shared SQLite database and events table.
# All 4 agents read and write from this single file.

import sqlite3
import os

# Import the database path from our shared config
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from config import DATABASE_PATH

# Create the database folder if it doesn't exist yet
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# Connect to (or create) the database file
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Create the single shared events table
# This is the backbone that all 4 agents communicate through
cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        REAL,        -- Video time in seconds when this event occurred
        person_id        INTEGER,     -- Clean sequential ID from Agent 1
        event_type       TEXT,        -- 'detected', 'violation_away', 'violation_phone'
        duration_seconds REAL,        -- How long the violation lasted (Agent 2 fills this)
        confidence       REAL,        -- YOLO detection confidence score
        crop_path        TEXT,        -- Path to saved crop image (Agent 2 fills this)
        vlm_summary      TEXT         -- LLaVA description (Agent 3 fills this)
    )
""")

conn.commit()
conn.close()

print("[SUCCESS] sentinel.db created with events table ready.")
print(f"Location: {DATABASE_PATH}")