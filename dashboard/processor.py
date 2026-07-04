import os
import cv2
import sqlite3
import sys
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config import MODEL_NAME, DATABASE_PATH


def run_pipeline(video_path, on_finish_callback):

    print("\n=== AGENT 1 STARTING: TRACKING + DATABASE FEED ===")

    # --- CONNECT DB ---
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM events")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
    conn.commit()

    print(f"[INFO] Connected to database: {DATABASE_PATH}")
    print("[INFO] Previous events cleared. Starting fresh analysis.")

    # --- OPEN VIDEO ---
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("[ERROR] Cannot open video")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video loaded: {total_frames} frames")

    # --- OUTPUT VIDEO (optional) ---
    output_dir = os.path.join(ROOT_DIR, "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Test2.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # --- MODEL ---
    model = YOLO(MODEL_NAME)

    # --- YOUR REGISTRIES (UNCHANGED) ---
    raw_id_counters = {}
    last_seen_frame = {}
    id_registry = {}
    next_clean_id = 1

    smoothed_boxes = {}
    EMA_ALPHA = 0.7

    MIN_FRAMES_TO_CONFIRM = 35
    GRACE_PERIOD_FRAMES = 45
    MIN_WRITE_CONF = 0.25

    frame_count = 0
    raw_ids = []

    # --- EMA FUNCTION (UNCHANGED) ---
    def ema_smooth(raw_id, new_box):
        new_box_f = [float(v) for v in new_box]

        if raw_id not in smoothed_boxes:
            smoothed_boxes[raw_id] = new_box_f
        else:
            prev = smoothed_boxes[raw_id]
            smoothed_boxes[raw_id] = [
                EMA_ALPHA * prev[i] + (1 - EMA_ALPHA) * new_box_f[i]
                for i in range(4)
            ]

        return [int(v) for v in smoothed_boxes[raw_id]]

    # =========================
    # MAIN LOOP (YOUR CORE CODE)
    # =========================
    while True:

        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        timestamp = frame_count / fps

        results = model.track(
            frame,
            persist=True,
            tracker="agent1_tracking/custom_tracker.yaml",
            classes=[0, 67],
            iou=0.2,
            imgsz=1536,
            conf=0.2,
            verbose=False
        )

        res = results[0]

        # -------------------------
        # PHONE DETECTIONS
        # -------------------------
        if res.boxes is not None:

            classes = res.boxes.cls.int().tolist() if res.boxes.cls is not None else []
            boxes = res.boxes.xyxy.int().tolist() if res.boxes.xyxy is not None else []
            confs = res.boxes.conf.tolist() if res.boxes.conf is not None else []

            for cls, box, conf in zip(classes, boxes, confs):

                if cls == 67 and conf >= MIN_WRITE_CONF:

                    x1, y1, x2, y2 = box

                    cursor.execute("""
                        INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        round(timestamp, 3),
                        None,
                        "phone_detected",
                        round(conf, 4),
                        x1, y1, x2, y2
                    ))

        # -------------------------
        # PERSON TRACKING
        # -------------------------
        if res.boxes.id is not None:

            all_ids = res.boxes.id.int().tolist()
            all_classes = res.boxes.cls.int().tolist()
            all_boxes = res.boxes.xyxy.int().tolist()
            all_confs = res.boxes.conf.tolist()

            person_data = [
                (rid, box, conf)
                for rid, cls, box, conf in zip(all_ids, all_classes, all_boxes, all_confs)
                if cls == 0
            ]

            raw_ids = [d[0] for d in person_data]
            boxes = [d[1] for d in person_data]
            confidences = [d[2] for d in person_data]

            # -------------------------
            # ID STABILIZATION LOGIC
            # -------------------------
            for raw_id in raw_ids:

                if raw_id in last_seen_frame and (frame_count - last_seen_frame[raw_id]) <= GRACE_PERIOD_FRAMES:
                    raw_id_counters[raw_id] = raw_id_counters.get(raw_id, 0) + 1
                else:
                    raw_id_counters[raw_id] = 1

                last_seen_frame[raw_id] = frame_count

                if raw_id_counters[raw_id] >= MIN_FRAMES_TO_CONFIRM and raw_id not in id_registry:
                    id_registry[raw_id] = next_clean_id
                    next_clean_id += 1
                    print(f"[NEW PERSON] Person {id_registry[raw_id]} confirmed")

            # -------------------------
            # DB WRITES + DRAWING
            # -------------------------
            for raw_id, box, conf in zip(raw_ids, boxes, confidences):

                if raw_id in id_registry and conf >= MIN_WRITE_CONF:

                    clean_id = id_registry[raw_id]
                    smooth_box = ema_smooth(raw_id, box)

                    x1, y1, x2, y2 = smooth_box

                    cursor.execute("""
                        INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        round(timestamp, 3),
                        clean_id,
                        "detected",
                        round(conf, 4),
                        x1, y1, x2, y2
                    ))

        # -------------------------
        # COMMIT PERIODICALLY
        # -------------------------
        if frame_count % 10 == 0:
            conn.commit()

        video_writer.write(frame)

    # =========================
    # CLEANUP
    # =========================
    conn.commit()
    conn.close()
    video_capture.release()
    video_writer.release()

    print("[INFO] Pipeline finished")

    on_finish_callback()