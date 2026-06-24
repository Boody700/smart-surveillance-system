import cv2
import json
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from config import VIDEO_PATH, DATABASE_PATH

# Storage
zones = []
current_points = []
frame_display = None

def mouse_click(event, x, y, flags, param):
    global current_points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Point added: ({x}, {y})")

        # Draw dot on screen
        cv2.circle(frame_display, (x, y), 5, (0, 255, 255), -1)

        # If 4 points collected, close the zone
        if len(current_points) == 4:
            pts = np.array(current_points, np.int32)
            cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(frame_display.copy(), [pts], (0, 255, 0))
            
            h, w = frame_display.shape[:2]
            normalized = [
                current_points[0][0]/w, current_points[0][1]/h,
                current_points[2][0]/w, current_points[2][1]/h
            ]
            zones.append(normalized)
            print(f"Zone {len(zones)} saved: {normalized}")
            current_points.clear()

        cv2.imshow("Calibrate Zones", frame_display)

def main():
    global frame_display

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read video")
        return

    frame_display = frame.copy()
    h, w = frame.shape[:2]

    # Instructions on screen
    cv2.putText(frame_display, "Click 4 corners per desk zone | S=Save | R=Reset | Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.namedWindow("Calibrate Zones")
    cv2.setMouseCallback("Calibrate Zones", mouse_click)
    cv2.imshow("Calibrate Zones", frame_display)

    print("\nClick 4 corners around each desk to define a zone.")
    print("S = Save all zones | R = Reset | Q = Quit\n")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if zones:
                os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
                with open(os.path.join(os.path.dirname(DATABASE_PATH), "zones.json"), "w") as f:
                    json.dump({"zones": zones}, f, indent=2)
                print(f"\nSaved {len(zones)} zones to zones.json")
            else:
                print("No zones drawn yet")

        elif key == ord('r'):
            zones.clear()
            current_points.clear()
            frame_display = frame.copy()
            cv2.putText(frame_display, "Click 4 corners per desk zone | S=Save | R=Reset | Q=Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Calibrate Zones", frame_display)
            print("Reset. Start again.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()