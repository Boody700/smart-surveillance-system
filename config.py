# config.py - Shared settings for all agents
import os

# Project Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "input1.mp4") 
DATABASE_PATH = os.path.join(ROOT_DIR, "database", "sentinel.db")
CROPS_DIR = os.path.join(ROOT_DIR, "crops")

# Model Settings
MODEL_NAME = "yolo26s.pt"  # The latest YOLO model for 2026