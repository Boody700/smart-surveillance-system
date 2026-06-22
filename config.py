# config.py - Shared settings for all agents
import os

# Project Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
<<<<<<< Updated upstream
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "input.mp4") # Change 'input.mp4' to your filename
=======
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "input.mp4") 
>>>>>>> Stashed changes
DATABASE_PATH = os.path.join(ROOT_DIR, "database", "sentinel.db")
CROPS_DIR = os.path.join(ROOT_DIR, "crops")

# Model Settings
<<<<<<< Updated upstream
MODEL_NAME = "yolo26n.pt"  # The latest YOLO model for 2026
=======
MODEL_NAME = "yolo11n.pt"  # The latest YOLO model for 2026
>>>>>>> Stashed changes
