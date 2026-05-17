# download_test_video.py
# Helper script to download our new test video directly into the project structure

import os
import yt_dlp

# Define our destination folder matching config.py structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(ROOT_DIR, "videos")
os.makedirs(videos_dir, exist_ok=True)

# The destination path we want for the file
output_file_path = os.path.join(videos_dir, "input1.mp4")

# YouTube URL provided
video_url = "https://youtu.be/LOSv_iojT7E?si=3qsDNWPHeQmYw5Lg"

print(f"[INFO] Initializing download for: {video_url}")

# Configure yt-dlp options
# ext=mp4 forces an MP4 container, and we pull the best standard quality merge
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp4]/best[ext=mp4]',
    'outtmpl': output_file_path,
    'quiet': False
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"\n########################################")
    print(f" SUCCESS: Video downloaded successfully!")
    print(f" Saved to: {output_file_path}")
    print(f"########################################\n")
except Exception as e:
    print(f"\n[ERROR] Failed to download video: {e}")