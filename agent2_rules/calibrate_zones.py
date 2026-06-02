import os
import sys
import json
import cv2
import numpy as np

# Ensure root directory is in python path to import config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import VIDEO_PATH, DATABASE_PATH

# Global storage states
zones = []          # Stores final normalized [min_x, min_y, max_x, max_y] boxes
current_points = [] # Temporary tracking click points (up to 4)
frame_display = None

def mouse_click(event, x, y, flags, param):
    global current_points, frame_display, zones
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"[POINT ADDED] Corner {len(current_points)}: ({x}, {y})")
        
        # Draw a yellow feedback dot for the click
        cv2.circle(frame_display, (x, y), 5, (0, 255, 255), -1)
        
        # Once 4 corners are selected, process the polygon bounding metrics
        if len(current_points) == 4:
            pts = np.array(current_points, np.int32)
            # Draw a green geometric indicator box over the frame canvas
            cv2.polylines(frame_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            h, w = frame_display.shape[:2]
            # Unpack coordinates to derive min/max bounds as requested
            xs = [pt[0] for pt in current_points]
            ys = [pt[1] for pt in current_points]
            
            # Form standard normalized coordinates [left, top, right, bottom]
            normalized_box = [min(xs)/w, min(ys)/h, max(xs)/w, max(ys)/h]
            zones.append(normalized_box)
            
            print(f"[ZONE CREATED] Zone {len(zones)} saved: {normalized_box}")
            current_points.clear() # Reset temporary click index for next desk
            
        cv2.imshow("Calibrate Zones", frame_display)

def main():
    global frame_display, zones, current_points
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[ERROR] Could not open video file at: {VIDEO_PATH}")
        sys.exit(1)
        
    frame_display = frame.copy()
    
    # Overlay lightweight helper instructions directly on video head window
    cv2.putText(frame_display, "Click 4 corners per desk | S=Save | R=Reset | Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.namedWindow("Calibrate Zones")
    cv2.setMouseCallback("Calibrate Zones", mouse_click)
    cv2.imshow("Calibrate Zones", frame_display)
    
    print("\n--- Desk Calibration Guide ---")
    print("1. Click exactly 4 corners matching a desk workspace layout.")
    print("2. Box will flash green, allowing you to move directly to next desk.")
    print("3. Press 'S' to write zones.json | 'R' to clear all shapes | 'Q' to exit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if zones:
                json_path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump({"zones": zones}, f, indent=2)
                print(f"\n[SUCCESS] Saved {len(zones)} desk zones directly to {json_path}")
            else:
                print("[WARNING] No zones drawn yet. Canvas profile is empty.")
        elif key == ord('r'):
            zones.clear()
            current_points.clear()
            frame_display = frame.copy()
            cv2.putText(frame_display, "Click 4 corners per desk | S=Save | R=Reset | Q=Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Calibrate Zones", frame_display)
            print("[RESET] Canvas cleared. Redraw zones from scratch.")
        elif key == ord('q'):
            print("[INFO] Closing Calibration panel.")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()