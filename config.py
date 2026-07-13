import os

# Get the directory where config.py lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(ROOT_DIR, "face_detection_model.pt")
PHONE_MODEL_PATH = os.path.join(ROOT_DIR, "Phone_best.pt")
# Build an absolute path to the database

DATABASE_PATH = os.path.join(ROOT_DIR, "database", "hfdb.db")

# Other settings
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "Chaos.mp4")
CROPS_DIR = os.path.join(ROOT_DIR, "crops")
MODEL_NAME = "yolo12n.pt"