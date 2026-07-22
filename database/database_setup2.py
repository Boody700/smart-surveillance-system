import sqlite3
import os
# We can import config directly now because of the -m flag
from config2 import DATABASE_PATH

# Ensure directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

print(f"[ACTION] Using database at: {DATABASE_PATH}")

conn = sqlite3.connect(DATABASE_PATH, timeout=30)
cursor = conn.cursor()

# WAL mode: agent1 now writes to BOTH `events` (wiped every run) and
# `person_gallery` (persists across runs) in the same connection lifecycle
# that other tools may also be reading from mid-run. WAL lets readers and a
# single writer proceed without blocking each other / hitting
# "database is locked" - default rollback-journal mode does not.
cursor.execute("PRAGMA journal_mode=WAL")
cursor.execute("PRAGMA busy_timeout=30000")

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

# --- PERSISTENT IDENTITY GALLERY ---
# Required by agent1's OSNet/FAISS persistent Re-ID (PersonGallery reads this
# on startup and writes to it every run). This table did NOT exist in the
# previous database_setup.py - running the friend's agent1 against that
# schema would fail immediately with "no such table: person_gallery".
#
# Deliberately CREATE TABLE IF NOT EXISTS, NOT dropped/recreated like
# `events` above - that's what makes identities persist across separate
# runs/videos instead of resetting every time you re-run this setup script.
# Do not add a DELETE/DROP for this table unless you intend to reset every
# known identity.
cursor.execute("""
    CREATE TABLE IF NOT EXISTS person_gallery (
        person_id        INTEGER PRIMARY KEY,
        embedding        BLOB NOT NULL,
        num_samples      INTEGER DEFAULT 1,
        first_seen_frame INTEGER,
        last_seen_frame  INTEGER
    )
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")

conn.commit()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events';")
events_ok = cursor.fetchone() is not None
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_gallery';")
gallery_ok = cursor.fetchone() is not None

if events_ok and gallery_ok:
    print("[SUCCESS] 'events' and 'person_gallery' tables created and verified.")
else:
    print("[ERROR] Table creation failed.")
    print(f"        events present: {events_ok} | person_gallery present: {gallery_ok}")

conn.close()