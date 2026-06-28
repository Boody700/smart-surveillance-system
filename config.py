import os

# Get the directory where config.py lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build an absolute path to the database
DATABASE_PATH = os.path.join(ROOT_DIR, "database", "sentinel.db")

# Other settings
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "Phone and Afk.mp4")
CROPS_DIR = os.path.join(ROOT_DIR, "crops")
MODEL_NAME = "yolo12n.pt"