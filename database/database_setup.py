import sqlite3
import os
# We can import config directly now because of the -m flag
from config import DATABASE_PATH

# Ensure directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

print(f"[ACTION] Using database at: {DATABASE_PATH}")

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS events")

cursor.execute("""
    CREATE TABLE events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        REAL,
        person_id        TEXT,
        event_type       TEXT,
        duration_seconds REAL,
        confidence       REAL,
        crop_path        TEXT,
        vlm_summary      TEXT,
        bbox_x1          INTEGER,
        bbox_y1          INTEGER,
        bbox_x2          INTEGER,
        bbox_y2          INTEGER,
        zone_id          INTEGER,
        zone_name        TEXT
    )
""")
conn.commit()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events';")
if cursor.fetchone():
    print("[SUCCESS] 'events' table created and verified.")
else:
    print("[ERROR] Table creation failed.")

conn.close()