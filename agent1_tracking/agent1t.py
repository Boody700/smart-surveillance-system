# agent1_tracking/agent1_persistent_reid.py
#
# Agent 1: Full production pipeline with PERSISTENT, APPEARANCE-BASED
# person re-identification.
#
#   YOLO -> BoT-SORT -> Track ID -> OSNet -> 512-D embedding -> FAISS -> Persistent Person ID
#
# How this differs from agent1_reid.py (the spatial-distance version):
#   agent1_reid.py remembers WHERE a person was last seen and matches by
#   position - good for a few seconds of occlusion behind a cubicle wall,
#   but it breaks down if the person is gone for a long time or reappears
#   somewhere else entirely.
#
#   This version instead recognizes people by WHAT THEY LOOK LIKE (an OSNet
#   appearance embedding matched via FAISS against a persistent gallery
#   stored in sentinel.db), so identities survive much longer gaps -
#   minutes, a trip out of frame and back, or even a person reappearing in a
#   completely separate run of this script on a later day.

import os
import sys
import cv2
import sqlite3
import json
import numpy as np
import faiss
import time
import torch
from collections import deque
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# --- DEVICE SELECTION ---
# Auto-detects CUDA so this runs on GPU wherever it's available, without
# needing a code change - falls back to CPU cleanly if no GPU/CUDA install
# is present. Used for every model in the pipeline (YOLO detector, phone/
# face/pose models, and the OSNet/ResNet50 embedder).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[INFO] CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] CUDA not available - running on CPU.")

from config import (
    VIDEO_PATH, MODEL_NAME, DATABASE_PATH,
    REID_MODEL_NAME, REID_WEIGHTS_PATH, EMBEDDING_DIM, REID_MATCH_THRESHOLD, MIN_CROP_SIZE,
    HYBRID_MATCH_WEIGHTS, TEMPORAL_MEMORY_LENGTH, RECHECK_INTERVAL_FRAMES,
    MIN_FRAMES_TO_CONFIRM, MAX_FRAMES_OCCLUDED, OCCLUSION_SEARCH_SCALE,
    DETECTION_CONF_THRESHOLD,
    # --- added: phone + face detection passes (own models, not part of
    # person tracking/Re-ID above) ---
    FACE_MODEL_PATH, PHONE_MODEL_PATH,
    PHONE_CONF, PHONE_IOU, PHONE_COCO_FALLBACK_CONF, PHONE_COCO_FALLBACK_IOU,
    PHONE_DETECT_EVERY_N_FRAMES,
    FACE_CONF, FACE_IOU, FACE_CROP_PAD_RATIO, FACE_DETECT_EVERY_N_FRAMES,
    # --- added: pose estimation pass for sleeping detection ---
    POSE_MODEL_PATH, POSE_CONF, POSE_DETECT_EVERY_N_FRAMES,
    SLEEP_HEAD_DROP_RATIO, SLEEP_FACE_VISIBILITY_THRESHOLD, SLEEP_FACE_RECENCY_FRAMES,
    SLEEP_MIN_CONSECUTIVE_FRAMES, SLEEP_GRACE_FRAMES,
    # --- added: only count a phone as "in use" if it overlaps a tracked
    # person's box, not just anywhere in frame (e.g. charging on a table) ---
    PHONE_MIN_PERSON_OVERLAP
)

# app.py's "Start Detection" button calls this script as:
#   subprocess.Popen([sys.executable, agent1_path] + [selected_video], ...)
# i.e. it passes whatever video was uploaded through the Streamlit UI as
# sys.argv[1]. Without this override, that upload was silently ignored and
# agent1 always processed config2.py's hardcoded VIDEO_PATH instead.
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    VIDEO_PATH = sys.argv[1]
    print(f"[INFO] Using video path passed from app.py: {VIDEO_PATH}")

# app.py's live-preview panel (Tab 1) polls this exact path and displays
# whatever's there whenever it sees a "PROGRESS:" line on stdout - see the
# write + print calls added at the end of the main loop below. Must match
# app.py's LIVE_FRAME_PATH = os.path.join(ROOT_DIR, "live_frame.jpg") exactly
# (same ROOT_DIR-from-__file__ computation on both sides, so this resolves
# to the same path as long as this script's own directory nesting matches
# app.py's).
LIVE_FRAME_PATH = os.path.join(ROOT_DIR, "live_frame.jpg")
LIVE_FRAME_WRITE_EVERY_N_FRAMES = 10  # matches app.py's own UPDATE_EVERY_FRAMES cadence
print("\n=== AGENT 1 STARTING: TRACKING + PERSISTENT RE-ID + DATABASE FEED ===")

# ---------------------------------------------------------------------------
# CLASS: Temporal Memory - tracking previous positions
# ---------------------------------------------------------------------------
class TemporalMemory:
    """
    Stores previous positions and features for each ID to track movement
    and improve matching.
    """
    def __init__(self, max_history=TEMPORAL_MEMORY_LENGTH):
        self.history = {}  # person_id -> deque of (frame, bbox, embedding)
        self.max_history = max_history
    
    def add_entry(self, person_id, frame_count, bbox, embedding):
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.max_history)
        self.history[person_id].append((frame_count, bbox, embedding))
    
    def get_spatial_score(self, person_id, current_bbox, max_distance=150):
        """Compute spatial similarity against previous positions."""
        if person_id not in self.history or not self.history[person_id]:
            return 0.0
        
        # Center of the current box
        cx1 = (current_bbox[0] + current_bbox[2]) / 2
        cy1 = (current_bbox[1] + current_bbox[3]) / 2
        
        # Minimum distance from the last 5 positions
        min_distance = float('inf')
        for _, last_bbox, _ in list(self.history[person_id])[-5:]:
            cx2 = (last_bbox[0] + last_bbox[2]) / 2
            cy2 = (last_bbox[1] + last_bbox[3]) / 2
            
            distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        # Convert distance to a similarity score (0-1)
        return max(0, 1 - min_distance / max_distance)
    
    def get_temporal_score(self, person_id, current_frame, max_gap=300):
        """Compute temporal similarity - the more recent the last sighting, the higher the confidence."""
        if person_id not in self.history or not self.history[person_id]:
            return 0.0
        
        last_frame, _, _ = self.history[person_id][-1]
        frames_gap = current_frame - last_frame
        
        if frames_gap > max_gap:
            return 0.0
        
        return max(0.5, 1 - frames_gap / max_gap)
    
    def get_predicted_bbox(self, person_id, current_bbox):
        """Predict position using a simple motion model."""
        if person_id not in self.history or len(self.history[person_id]) < 3:
            return current_bbox
        
        # Compute average velocity from the last 5 frames
        recent = list(self.history[person_id])[-5:]
        if len(recent) < 2:
            return current_bbox
        
        velocities = []
        for i in range(1, len(recent)):
            _, prev_bbox, _ = recent[i-1]
            _, curr_bbox, _ = recent[i]
            
            dx = (curr_bbox[0] + curr_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2
            dy = (curr_bbox[1] + curr_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
            velocities.append((dx, dy))
        
        if velocities:
            avg_vel = np.mean(velocities, axis=0)
            predicted = [
                int(current_bbox[0] + avg_vel[0]),
                int(current_bbox[1] + avg_vel[1]),
                int(current_bbox[2] + avg_vel[0]),
                int(current_bbox[3] + avg_vel[1])
            ]
            return predicted
        
        return current_bbox

# ---------------------------------------------------------------------------
# CLASS: Occlusion Handler
# ---------------------------------------------------------------------------
class OcclusionHandler:
    """
    Tracks occlusion states and predicts positions during occlusion.
    """
    def __init__(self, max_frames=MAX_FRAMES_OCCLUDED, search_scale=OCCLUSION_SEARCH_SCALE):
        self.max_frames = max_frames
        self.search_scale = search_scale
        self.occluded = {}  # track_id -> {'frames': count, 'last_bbox': bbox}
    
    def is_occluded(self, track_id):
        return track_id in self.occluded
    
    def update(self, track_id, is_occluded, current_bbox):
        if is_occluded:
            if track_id not in self.occluded:
                self.occluded[track_id] = {'frames': 0, 'last_bbox': current_bbox}
            self.occluded[track_id]['frames'] += 1
            self.occluded[track_id]['last_bbox'] = current_bbox
        else:
            if track_id in self.occluded:
                del self.occluded[track_id]
    
    def get_search_region(self, track_id, frame_shape):
        """Expand the search region while occluded."""
        if track_id not in self.occluded:
            return None
        
        bbox = self.occluded[track_id]['last_bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = (bbox[2] - bbox[0]) * self.search_scale
        height = (bbox[3] - bbox[1]) * self.search_scale
        
        h, w = frame_shape[:2]
        search_region = [
            max(0, int(center_x - width/2)),
            max(0, int(center_y - height/2)),
            min(w, int(center_x + width/2)),
            min(h, int(center_y + height/2))
        ]
        return search_region

# ---------------------------------------------------------------------------
# STAGE: OSNet embedder (the "OSNet / FastReID -> 512-D embedding" box)
# ---------------------------------------------------------------------------
class PersonEmbedder:
    """
    Wraps an OSNet (torchreid) feature extractor to turn a person crop into a
    512-D appearance embedding.

    Falls back to a plain ImageNet-pretrained ResNet50's penultimate layer if
    torchreid / OSNet's pretrained weights aren't available in this
    environment, so the pipeline still runs end-to-end (with noticeably
    weaker appearance discrimination) instead of crashing outright. For real
    production accuracy, make sure torchreid + OSNet weights are properly
    installed - the fallback is a safety net, not a target.
    """

    def __init__(self, model_name=REID_MODEL_NAME, device=DEVICE):
        self.device = device
        self.dim = EMBEDDING_DIM
        self.backend = None
        self.embed_cache = {}  # cache for embeddings to speed up processing

        try:
            from torchreid.reid.utils import FeatureExtractor

            # FIX: model_path="" does NOT give you a re-id-trained model - it
            # only initializes OSNet's backbone with generic ImageNet
            # CLASSIFICATION weights (torchreid prints "Successfully loaded
            # imagenet pretrained weights ... layers discarded:
            # classifier.weight/bias" when this happens). Those weights were
            # never trained to distinguish one person from another, so
            # matching quality is close to the ResNet50 fallback despite
            # "OSNet" technically loading without error. Real re-id weights
            # must be downloaded separately (see REID_WEIGHTS_PATH in
            # config.py) and passed explicitly via model_path.
            weights_path = REID_WEIGHTS_PATH if REID_WEIGHTS_PATH and os.path.isfile(REID_WEIGHTS_PATH) else ""

            self.extractor = FeatureExtractor(
                model_name=model_name,
                model_path=weights_path,
                device=device
            )
            self.backend = "torchreid"

            if weights_path:
                self.reid_quality = "trained"
                print(f"[REID] Loaded OSNet ({model_name}) with RE-ID-TRAINED weights "
                      f"from {weights_path}. Embedding dim = {self.dim}.")
            else:
                # Not a crash, but definitely not production quality either -
                # make this impossible to miss, same as the full fallback below.
                self.reid_quality = "imagenet_only"
                print("\n" + "!" * 70)
                print("[REID][WARNING] REID_WEIGHTS_PATH is empty or the file wasn't found.")
                print("[REID][WARNING] OSNet loaded with GENERIC IMAGENET weights only -")
                print("[REID][WARNING] these were never trained to re-identify people, so")
                print("[REID][WARNING] expect ResNet50-fallback-level matching quality despite")
                print("[REID][WARNING] 'OSNet' appearing to load successfully.")
                print("[REID][WARNING] Download real re-id weights from:")
                print("[REID][WARNING]   https://huggingface.co/kaiyangzhou/osnet")
                print("[REID][WARNING] and set REID_WEIGHTS_PATH in config.py to that file.")
                print("!" * 70 + "\n")

        except Exception as e:
            print("\n" + "!" * 70)
            print("[REID][WARNING] torchreid/OSNet FAILED TO LOAD - FALLING BACK")
            print(f"[REID][WARNING] Reason: {e}")
            print("[REID][WARNING] Using generic ResNet50 (ImageNet) instead.")
            print("[REID][WARNING] This is NOT a person-re-id-trained model - expect")
            print("[REID][WARNING] noticeably more identity switches and fragmentation.")
            print("[REID][WARNING] Fix: pip uninstall torchreid && pip install torchreid")
            print("!" * 70 + "\n")
            import torch
            import torchvision.models as tv_models
            import torchvision.transforms as T

            self.torch = torch
            resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
            resnet.fc = torch.nn.Identity()   # strip the classifier head -> 2048-D features
            resnet.eval().to(device)
            self.model = resnet
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.backend = "resnet50_fallback"
            self.reid_quality = "resnet50_fallback"
            self.dim = 2048  # NOTE: fallback embedding is 2048-D, not 512-D like OSNet

    def embed(self, crop_bgr, use_cache=False):
        """crop_bgr: HxWx3 uint8 BGR image (straight from cv2). Returns an L2-normalized np.float32 vector."""
        if use_cache:
            # Use a hash of the crop as a (rough) cache key
            import hashlib
            crop_hash = hashlib.md5(crop_bgr.tobytes()).hexdigest()
            if crop_hash in self.embed_cache:
                return self.embed_cache[crop_hash]
        
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "torchreid":
            feats = self.extractor([crop_rgb])   # torchreid handles its own resize/normalize
            vec = feats.cpu().numpy()[0].astype(np.float32)
        else:
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                vec = self.model(tensor)[0].cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        if use_cache:
            self.embed_cache[crop_hash] = vec
            if len(self.embed_cache) > 100:  # clear the cache periodically
                self.embed_cache.clear()
        
        return vec

# ---------------------------------------------------------------------------
# STAGE: FAISS-backed persistent gallery (the "FAISS search -> Persistent
# Person ID" box)
# ---------------------------------------------------------------------------
class PersonGallery:
    """
    Holds one running-average embedding per Persistent Person ID, backed by a
    FAISS flat inner-product index (== cosine similarity, since every vector
    is L2-normalized before it goes in). Mirrors everything into the
    person_gallery SQL table so identities survive across separate runs of
    this script - that table is never wiped, unlike `events`.
    """

    def __init__(self, conn, dim):
        self.conn = conn
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.person_ids = []      # kept for iteration/reporting convenience
        self.embeddings = {}      # person_id -> current running-average vector
        self.sample_counts = {}   # person_id -> how many crops have been averaged in
        self.last_seen_frame = {} # person_id -> last frame seen
        self._load_existing()

    def _load_existing(self):
        cur = self.conn.cursor()
        cur.execute("SELECT person_id, embedding, num_samples FROM person_gallery")
        rows = cur.fetchall()
        for person_id, blob, num_samples in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != self.dim:
                print(f"[REID][WARN] Skipping stored embedding for person {person_id}: "
                      f"stored dim {vec.shape[0]} != current embedder dim {self.dim}. "
                      f"(Did you switch embedders since this gallery was built?)")
                continue
            self._add_to_index(person_id, vec.copy(), num_samples)
        if self.person_ids:
            print(f"[REID] Loaded {len(self.person_ids)} persistent identities from previous runs.")
        else:
            print("[REID] Starting with an empty identity gallery.")

    def _add_to_index(self, person_id, vec, num_samples):
        self.index.add_with_ids(vec.reshape(1, -1), np.array([person_id], dtype=np.int64))
        if person_id not in self.person_ids:
            self.person_ids.append(person_id)
        self.embeddings[person_id] = vec
        self.sample_counts[person_id] = num_samples

    def match(self, vec, threshold=REID_MATCH_THRESHOLD):
        if self.index.ntotal == 0:
            return None, 0.0
        sims, idxs = self.index.search(vec.reshape(1, -1), min(3, self.index.ntotal))
        for i in range(len(idxs[0])):
            best_sim = float(sims[0][i])
            best_person_id = int(idxs[0][i])
            if best_person_id == -1:
                continue
            if best_sim >= threshold:
                return best_person_id, best_sim
        if len(idxs[0]) > 0 and idxs[0][0] != -1:
            return None, float(sims[0][0])
        return None, 0.0

    def upsert(self, person_id, vec, frame_count, is_new, confidence=1.0):
        cur = self.conn.cursor()
        if is_new:
            self._add_to_index(person_id, vec, 1)
            cur.execute("""
                INSERT INTO person_gallery (person_id, embedding, num_samples, first_seen_frame, last_seen_frame)
                VALUES (?, ?, 1, ?, ?)
            """, (person_id, vec.astype(np.float32).tobytes(), frame_count, frame_count))
        else:
            n = self.sample_counts[person_id]
            old_vec = self.embeddings[person_id]
            alpha = min(0.5, 0.3 * confidence)
            new_vec = (1 - alpha) * old_vec + alpha * vec
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            self.embeddings[person_id] = new_vec
            self.sample_counts[person_id] = n + 1
            self.last_seen_frame[person_id] = frame_count
            self.index.remove_ids(np.array([person_id], dtype=np.int64))
            self.index.add_with_ids(new_vec.reshape(1, -1), np.array([person_id], dtype=np.int64))
            cur.execute("""
                UPDATE person_gallery
                SET embedding = ?, num_samples = ?, last_seen_frame = ?
                WHERE person_id = ?
            """, (new_vec.astype(np.float32).tobytes(), n + 1, frame_count, person_id))
        self.conn.commit()

    def next_person_id(self):
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(person_id) FROM person_gallery")
        row = cur.fetchone()
        return (row[0] or 0) + 1
    
    def merge_identities(self, primary_id, duplicate_id, frame_count):
        if duplicate_id not in self.embeddings or primary_id not in self.embeddings:
            return
        print(f"[MERGE] Merging person {duplicate_id} into {primary_id}")
        primary_embed = self.embeddings[primary_id]
        duplicate_embed = self.embeddings[duplicate_id]
        n1 = self.sample_counts[primary_id]
        n2 = self.sample_counts[duplicate_id]
        merged_embed = (primary_embed * n1 + duplicate_embed * n2) / (n1 + n2)
        merged_embed = merged_embed / np.linalg.norm(merged_embed)
        self.embeddings[primary_id] = merged_embed
        self.sample_counts[primary_id] = n1 + n2
        self.last_seen_frame[primary_id] = max(
            self.last_seen_frame.get(primary_id, 0),
            self.last_seen_frame.get(duplicate_id, 0)
        )
        if duplicate_id in self.person_ids:
            self.person_ids.remove(duplicate_id)
        self.index.remove_ids(np.array([duplicate_id], dtype=np.int64))
        self.index.remove_ids(np.array([primary_id], dtype=np.int64))
        self.index.add_with_ids(merged_embed.reshape(1, -1), np.array([primary_id], dtype=np.int64))
        cur = self.conn.cursor()
        cur.execute("DELETE FROM person_gallery WHERE person_id = ?", (duplicate_id,))
        cur.execute("""
            UPDATE person_gallery 
            SET embedding = ?, num_samples = ?, last_seen_frame = ?
            WHERE person_id = ?
        """, (merged_embed.astype(np.float32).tobytes(), n1 + n2, frame_count, primary_id))
        self.conn.commit()
        self.embeddings.pop(duplicate_id, None)
        self.sample_counts.pop(duplicate_id, None)
        self.last_seen_frame.pop(duplicate_id, None)

# ---------------------------------------------------------------------------
# CLASS: Performance Monitor
# ---------------------------------------------------------------------------
class PerformanceMonitor:
    """Tracks system performance and produces reports."""
    def __init__(self):
        self.metrics = {
            'total_detections': 0,
            'unique_persons': 0,
            'id_switches': 0,
            'new_identities': 0,
            'reid_matches': 0,
            'reid_misses': 0,
            'total_embed_time': 0,
            'embed_count': 0
        }
        self.switch_log = []
        self.start_time = time.time()
    
    def log_id_switch(self, old_id, new_id, reason):
        self.metrics['id_switches'] += 1
        self.switch_log.append({
            'timestamp': time.time(),
            'old_id': old_id,
            'new_id': new_id,
            'reason': reason
        })
    
    def log_detection(self):
        self.metrics['total_detections'] += 1
    
    def log_identity(self, is_new):
        if is_new:
            self.metrics['new_identities'] += 1
        else:
            self.metrics['reid_matches'] += 1
    
    def log_embed_time(self, time_seconds):
        self.metrics['total_embed_time'] += time_seconds
        self.metrics['embed_count'] += 1
    
    def get_avg_embed_time(self):
        if self.metrics['embed_count'] > 0:
            return self.metrics['total_embed_time'] / self.metrics['embed_count']
        return 0
    
    def print_report(self, frame_count, gallery_size):
        elapsed = time.time() - self.start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print("\n" + "="*60)
        print("              PERFORMANCE REPORT")
        print("="*60)
        print(f"Frames processed          : {frame_count}")
        print(f"Processing time (seconds) : {elapsed:.2f}")
        print(f"Average FPS               : {fps:.2f}")
        print(f"Total detections          : {self.metrics['total_detections']}")
        print(f"Unique persons in gallery : {gallery_size}")
        print(f"New identities created    : {self.metrics['new_identities']}")
        print(f"ReID matches              : {self.metrics['reid_matches']}")
        print(f"ReID misses               : {self.metrics['reid_misses']}")
        print(f"ID switches               : {self.metrics['id_switches']}")
        print(f"Avg embed time (ms)       : {self.get_avg_embed_time()*1000:.2f}")
        if self.switch_log:
            print("\nRecent ID switches:")
            for entry in self.switch_log[-5:]:
                print(f"  {entry['old_id']} -> {entry['new_id']} ({entry['reason']})")
        print("="*60 + "\n")

# ---------------------------------------------------------------------------
# CONNECT TO DATABASE
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DATABASE_PATH, timeout=30)
cursor = conn.cursor()
cursor.execute("PRAGMA busy_timeout=30000")

cursor.execute("DELETE FROM events")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='events'")
conn.commit()
print(f"[INFO] Connected to database: {DATABASE_PATH}")
print("[INFO] Events cleared for this run. Persistent identity gallery kept intact.")

# ---------------------------------------------------------------------------
# OPEN VIDEO
# ---------------------------------------------------------------------------
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video: {total_frames} frames at {fps:.2f} FPS")

# --- LOAD ZONES ---
zones_json_path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")
zone_polys_px = []
if os.path.exists(zones_json_path):
    try:
        with open(zones_json_path, "r") as f:
            zones_data = json.load(f).get("zones", [])
        zone_polys_px = [
            np.array([[p[0] * frame_width, p[1] * frame_height] for p in z], dtype=np.int32)
            for z in zones_data
        ]
        print(f"[INFO] Loaded {len(zone_polys_px)} zone(s) from {zones_json_path}")
    except Exception as e:
        print(f"[WARN] Failed to load zones.json ({e}) - continuing without zone overlay.")
else:
    print(f"[WARN] No zones.json found at {zones_json_path} - continuing without zone overlay.")

ZONE_COLOR = (255, 200, 0)

def draw_zones(frame):
    for i, poly in enumerate(zone_polys_px):
        cv2.polylines(frame, [poly], True, ZONE_COLOR, 2)
        cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
        cv2.putText(frame, f"Zone {i}", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)

# --- SET UP OUTPUT VIDEO FOR VISUAL VERIFICATION ---
output_dir = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_agent1_new7.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
model = YOLO(MODEL_NAME)
model.to(DEVICE)
embedder = PersonEmbedder()
gallery = PersonGallery(conn, dim=embedder.dim)

phone_model = None
PHONE_USES_COCO_FALLBACK = False
if os.path.exists(PHONE_MODEL_PATH):
    print(f"[INFO] Loading custom phone model: {PHONE_MODEL_PATH}")
    phone_model = YOLO(PHONE_MODEL_PATH)
    phone_model.to(DEVICE)
else:
    print(f"[WARN] Phone model not found at {PHONE_MODEL_PATH} — "
          f"falling back to COCO class 67 ('cell phone') on the main person model.")
    PHONE_USES_COCO_FALLBACK = True

face_model = None
if os.path.exists(FACE_MODEL_PATH):
    print(f"[INFO] Loading face model: {FACE_MODEL_PATH}")
    face_model = YOLO(FACE_MODEL_PATH)
    face_model.to(DEVICE)
else:
    print(f"[WARN] Face model not found at {FACE_MODEL_PATH} — face detection disabled for this run.")

print(f"[INFO] Loading pose model: {POSE_MODEL_PATH}")
pose_model = YOLO(POSE_MODEL_PATH)
pose_model.to(DEVICE)

sleeping_state = {}
last_face_seen_frame = {}

def expand_box(box, pad_ratio, fw, fh):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return (max(0, x1 - px), max(0, y1 - py), min(fw - 1, x2 + px), min(fh - 1, y2 + py))

def overlap_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area

temporal_memory = TemporalMemory()
occlusion_handler = OcclusionHandler()
performance_monitor = PerformanceMonitor()

track_to_person = {}
track_bbox_history = {}
frames_since_recheck = {}
track_hit_counts = {}
person_id_to_track = {}

frame_count = 0

def get_adjusted_bbox(track_id, bbox):
    if track_id not in track_bbox_history:
        track_bbox_history[track_id] = []
    track_bbox_history[track_id].append(bbox)
    if len(track_bbox_history[track_id]) > 10:
        track_bbox_history[track_id].pop(0)
    if len(track_bbox_history[track_id]) >= 5:
        heights = [b[3] - b[1] for b in track_bbox_history[track_id][-5:]]
        avg_height = np.mean(heights)
        current_height = bbox[3] - bbox[1]
        if current_height > 0 and abs(current_height - avg_height) / avg_height > 0.3:
            scale = avg_height / current_height
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = (bbox[2] - bbox[0]) * scale
            new_bbox = [
                int(center_x - width/2),
                int(center_y - avg_height/2),
                int(center_x + width/2),
                int(center_y + avg_height/2)
            ]
            return new_bbox
    return bbox

def filter_crop(crop, min_size=MIN_CROP_SIZE, blur_threshold=50):
    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score > blur_threshold

def get_ranked_candidates(track_id, track_person_id, current_bbox, embedding, frame_count, top_k=3):
    candidates = {}
    if track_person_id is not None and track_person_id in gallery.embeddings:
        current_embed = gallery.embeddings[track_person_id]
        visual_sim = float(np.dot(embedding, current_embed))
        if visual_sim > 0.6:
            candidates[track_person_id] = max(candidates.get(track_person_id, -1.0), visual_sim)
    if gallery.index.ntotal > 0:
        k = min(top_k, gallery.index.ntotal)
        sims, idxs = gallery.index.search(embedding.reshape(1, -1), k)
        for i in range(len(idxs[0])):
            person_id = int(idxs[0][i])
            if person_id == -1:
                continue
            visual_sim = float(sims[0][i])
            if visual_sim < REID_MATCH_THRESHOLD:
                continue
            spatial_score = temporal_memory.get_spatial_score(person_id, current_bbox)
            temporal_score = temporal_memory.get_temporal_score(person_id, frame_count)
            bonus = (
                spatial_score * HYBRID_MATCH_WEIGHTS['spatial'] +
                temporal_score * HYBRID_MATCH_WEIGHTS['temporal']
            )
            hybrid_score = min(1.0, visual_sim + (1 - visual_sim) * bonus)
            candidates[person_id] = max(candidates.get(person_id, -1.0), hybrid_score)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return ranked

def detect_and_fix_duplicate_ids():
    active_ids = list(set(track_to_person.values()))
    duplicate_groups = []
    checked = set()
    for i, id1 in enumerate(active_ids):
        if id1 in checked:
            continue
        group = [id1]
        for id2 in active_ids[i+1:]:
            if id2 in checked:
                continue
            if id1 in gallery.embeddings and id2 in gallery.embeddings:
                sim = np.dot(gallery.embeddings[id1], gallery.embeddings[id2])
                if sim > 0.90:
                    group.append(id2)
                    checked.add(id2)
        if len(group) > 1:
            duplicate_groups.append(group)
            checked.update(group)
    for group in duplicate_groups:
        primary_id = min(group)
        for duplicate_id in group:
            if duplicate_id != primary_id:
                gallery.merge_identities(primary_id, duplicate_id, frame_count)
                for track_id, person_id in list(track_to_person.items()):
                    if person_id == duplicate_id:
                        track_to_person[track_id] = primary_id
                        performance_monitor.log_id_switch(
                            duplicate_id, primary_id, "duplicate_merge"
                        )

# --- MAIN PROCESSING LOOP ---
while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count += 1
    current_timestamp = frame_count / fps

    draw_zones(frame)

    for track_id in list(occlusion_handler.occluded.keys()):
        if track_id in track_to_person:
            occlusion_handler.occluded[track_id]['frames'] += 1
        if occlusion_handler.occluded[track_id]['frames'] > occlusion_handler.max_frames:
            print(f"[OCCLUSION] Track {track_id} occluded > {occlusion_handler.max_frames} "
                  f"frames - giving up on position-based recovery, will rely on FAISS re-id.")
            del occlusion_handler.occluded[track_id]
            track_to_person.pop(track_id, None)

    results = model.track(
        frame,
        persist=True,
        tracker="agent1_tracking/custom_tracker_botsort.yaml",
        classes=[0],
        conf=DETECTION_CONF_THRESHOLD,
        iou=0.5,
        imgsz=1536,
        device=DEVICE,
        verbose=False
    )

    frame_results = results[0]
    detected_track_ids = []

    reembed_proposals = []
    passthrough_hits = []
    claimed_this_frame = set()

    if frame_results.boxes.id is not None:
        track_ids = frame_results.boxes.id.int().tolist()
        boxes = frame_results.boxes.xyxy.int().tolist()
        confidences = frame_results.boxes.conf.tolist()

        for track_id, box, conf in zip(track_ids, boxes, confidences):
            performance_monitor.log_detection()
            detected_track_ids.append(track_id)

            was_occluded = track_id in occlusion_handler.occluded
            search_region = occlusion_handler.get_search_region(track_id, frame.shape) if was_occluded else None
            if was_occluded:
                del occlusion_handler.occluded[track_id]

            track_hit_counts[track_id] = track_hit_counts.get(track_id, 0) + 1
            if track_hit_counts[track_id] < MIN_FRAMES_TO_CONFIRM:
                continue

            adjusted_box = get_adjusted_bbox(track_id, box)
            if search_region:
                adjusted_box = search_region

            x1, y1, x2, y2 = adjusted_box
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame_width, x2), min(frame_height, y2)
            crop = frame[y1c:y2c, x1c:x2c]

            crop_is_usable = crop.size > 0 and filter_crop(crop)

            need_embed = (
                track_id not in track_to_person or
                frames_since_recheck.get(track_id, RECHECK_INTERVAL_FRAMES) >= RECHECK_INTERVAL_FRAMES
            )

            current_person_id = track_to_person.get(track_id)

            if need_embed and crop_is_usable:
                embed_start = time.time()
                vec = embedder.embed(crop, use_cache=True)
                embed_time = time.time() - embed_start
                performance_monitor.log_embed_time(embed_time)

                candidates = get_ranked_candidates(
                    track_id, current_person_id, adjusted_box, vec, frame_count
                )
                reembed_proposals.append({
                    "track_id": track_id, "box": adjusted_box, "conf": conf,
                    "vec": vec, "current_person_id": current_person_id,
                    "candidates": candidates,
                })
                frames_since_recheck[track_id] = 0
            else:
                frames_since_recheck[track_id] = frames_since_recheck.get(track_id, 0) + 1
                if current_person_id is not None:
                    claimed_this_frame.add(current_person_id)
                passthrough_hits.append({"track_id": track_id, "box": adjusted_box, "conf": conf})

    reembed_proposals.sort(
        key=lambda p: (p["candidates"][0][1] if p["candidates"] else -1.0),
        reverse=True
    )

    for p in reembed_proposals:
        track_id = p["track_id"]
        current_person_id = p["current_person_id"]
        vec = p["vec"]
        adjusted_box = p["box"]
        conf = p["conf"]

        assigned_id = None
        assigned_score = 0.0
        for person_id, score in p["candidates"]:
            if person_id not in claimed_this_frame:
                assigned_id = person_id
                assigned_score = score
                break

        if current_person_id is not None:
            temporal_memory.add_entry(current_person_id, frame_count, adjusted_box, vec)

        if assigned_id is not None:
            if current_person_id is not None and current_person_id != assigned_id:
                performance_monitor.log_id_switch(current_person_id, assigned_id, "reid_match")
            performance_monitor.log_identity(is_new=False)
            confidence_factor = min(1.0, conf * 1.2 * assigned_score)
            gallery.upsert(assigned_id, vec, frame_count, is_new=False, confidence=confidence_factor)
            print(f"[RE-ID] Track {track_id} -> Person {assigned_id} (sim: {assigned_score:.3f})")
        else:
            if current_person_id is not None and current_person_id not in claimed_this_frame:
                assigned_id = current_person_id
                performance_monitor.log_identity(is_new=False)
                gallery.upsert(assigned_id, vec, frame_count, is_new=False,
                                confidence=min(1.0, conf * 0.5))
                best_sim = p["candidates"][0][1] if p["candidates"] else 0.0
                print(f"[KEEP] Track {track_id} -> Person {assigned_id} "
                      f"(no confident re-match, best alt sim: {best_sim:.3f} - kept prior identity)")
            else:
                assigned_id = gallery.next_person_id()
                performance_monitor.log_identity(is_new=True)
                gallery.upsert(assigned_id, vec, frame_count, is_new=True, confidence=1.0)
                best_sim = p["candidates"][0][1] if p["candidates"] else 0.0
                print(f"[NEW] Track {track_id} -> Person {assigned_id} (best sim: {best_sim:.3f})")

        claimed_this_frame.add(assigned_id)
        track_to_person[track_id] = assigned_id
        person_id_to_track[assigned_id] = track_id
        temporal_memory.add_entry(assigned_id, frame_count, adjusted_box, vec)
        p["assigned_id"] = assigned_id

    for hit in reembed_proposals + passthrough_hits:
        track_id = hit["track_id"]
        if track_id not in track_to_person:
            continue
        person_id = track_to_person[track_id]
        x1, y1, x2, y2 = hit["box"]
        conf = hit["conf"]

        cursor.execute("""
            INSERT INTO events
                (timestamp, person_id, event_type, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            round(current_timestamp, 3),
            person_id,
            "detected",
            round(conf, 4),
            x1, y1, x2, y2
        ))

        color = (0, 255, 0) if person_id in gallery.embeddings else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"ID:{person_id}"
        font_scale = 0.32
        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - txt_h - 4), (x1 + txt_w + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    person_boxes_this_frame = [
        hit["box"] for hit in reembed_proposals + passthrough_hits
        if hit["track_id"] in track_to_person
    ]

    if frame_count % PHONE_DETECT_EVERY_N_FRAMES == 0:
        if phone_model is not None:
            phone_results = phone_model.predict(
                frame, conf=PHONE_CONF, iou=PHONE_IOU, imgsz=1536, device=DEVICE, verbose=False
            )
        else:
            phone_results = model.predict(
                frame, classes=[67], conf=PHONE_COCO_FALLBACK_CONF,
                iou=PHONE_COCO_FALLBACK_IOU, imgsz=1536, device=DEVICE, verbose=False
            )

        pboxes = phone_results[0].boxes
        if pboxes is not None and len(pboxes) > 0:
            for pbox, pconf in zip(pboxes.xyxy.int().tolist(), pboxes.conf.tolist()):
                px1, py1, px2, py2 = pbox
                max_overlap = max(
                    (overlap_ratio((px1, py1, px2, py2), pb) for pb in person_boxes_this_frame),
                    default=0.0
                )
                if max_overlap < PHONE_MIN_PERSON_OVERLAP:
                    continue

                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round(current_timestamp, 3),
                    None,
                    "phone_detected",
                    round(pconf, 4),
                    px1, py1, px2, py2
                ))
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 1)
                plabel = "PHONE"
                (pw, ph), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (px1, py1 - ph - 6), (px1 + pw + 4, py1), (0, 255, 255), -1)
                cv2.putText(frame, plabel, (px1 + 2, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if face_model is not None and frame_count % FACE_DETECT_EVERY_N_FRAMES == 0:
        for hit in reembed_proposals + passthrough_hits:
            track_id = hit["track_id"]
            if track_id not in track_to_person:
                continue
            person_id = track_to_person[track_id]
            x1, y1, x2, y2 = hit["box"]
            ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), FACE_CROP_PAD_RATIO, frame_width, frame_height)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            face_results = face_model.predict(crop, conf=FACE_CONF, iou=FACE_IOU, imgsz=320, device=DEVICE, verbose=False)
            fboxes = face_results[0].boxes
            if fboxes is None or len(fboxes) == 0:
                continue

            fconfs = fboxes.conf.tolist()
            best_i = max(range(len(fconfs)), key=lambda i: fconfs[i])
            fx1, fy1, fx2, fy2 = fboxes.xyxy.int().tolist()[best_i]
            fconf = fconfs[best_i]

            last_face_seen_frame[person_id] = frame_count

            fx1_full, fy1_full = fx1 + ex1, fy1 + ey1
            fx2_full, fy2_full = fx2 + ex1, fy2 + ey1

            cursor.execute("""
                INSERT INTO events
                    (timestamp, person_id, event_type, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                round(current_timestamp, 3),
                person_id,
                "face_detected",
                round(fconf, 4),
                fx1_full, fy1_full, fx2_full, fy2_full
            ))

            cv2.rectangle(frame, (fx1_full, fy1_full), (fx2_full, fy2_full), (255, 100, 0), 1)
            flabel = f"FACE P{person_id}"
            (fw_txt, fh_txt), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (fx1_full, fy1_full - fh_txt - 5),
                          (fx1_full + fw_txt + 3, fy1_full), (200, 80, 0), -1)
            cv2.putText(frame, flabel, (fx1_full + 2, fy1_full - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    FACE_KPTS = [0, 1, 2, 3, 4]
    NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6

    if pose_model is not None and frame_count % POSE_DETECT_EVERY_N_FRAMES == 0:
        pose_check_hits = [
            hit for hit in reembed_proposals + passthrough_hits
            if hit["track_id"] in track_to_person
        ]

        for hit in pose_check_hits:
            track_id = hit["track_id"]
            person_id = track_to_person[track_id]
            x1, y1, x2, y2 = hit["box"]
            crop = frame[max(0, y1):min(frame_height, y2), max(0, x1):min(frame_width, x2)]
            if crop.size == 0:
                continue

            state = sleeping_state.setdefault(person_id, {"consecutive": 0, "absent": 0, "logged": False})

            pose_results = pose_model.predict(crop, conf=POSE_CONF, imgsz=320, device=DEVICE, verbose=False)
            pboxes = pose_results[0].boxes
            keypoints = pose_results[0].keypoints
            if pboxes is None or len(pboxes) == 0 or keypoints is None or keypoints.conf is None:
                state["absent"] += 1
                if state["absent"] > SLEEP_GRACE_FRAMES:
                    state["consecutive"] = 0
                    state["logged"] = False
                continue

            best_i = int(pboxes.conf.argmax())
            kpts_conf = keypoints.conf[best_i].tolist()
            kpts_xy = keypoints.xy[best_i].tolist()
            if len(kpts_conf) < 7:
                state["absent"] += 1
                continue

            crop_h = crop.shape[0]
            face_visible_score = max(kpts_conf[i] for i in FACE_KPTS)
            l_sh_conf, r_sh_conf = kpts_conf[L_SHOULDER], kpts_conf[R_SHOULDER]
            shoulders_visible = l_sh_conf >= 0.3 or r_sh_conf >= 0.3

            head_drop_ratio = None
            if shoulders_visible:
                shoulder_y = np.mean([y for (x, y), c in
                                       zip([kpts_xy[L_SHOULDER], kpts_xy[R_SHOULDER]], [l_sh_conf, r_sh_conf])
                                       if c >= 0.3])
                nose_y = kpts_xy[NOSE][1]
                head_drop_ratio = (nose_y - shoulder_y) / crop_h

            is_sleeping_candidate = (
                shoulders_visible and
                head_drop_ratio is not None and
                head_drop_ratio > SLEEP_HEAD_DROP_RATIO and
                face_visible_score < SLEEP_FACE_VISIBILITY_THRESHOLD
            )

            if is_sleeping_candidate:
                state["consecutive"] += 1
                state["absent"] = 0
            else:
                state["absent"] += 1
                if state["absent"] > SLEEP_GRACE_FRAMES:
                    state["consecutive"] = 0
                    state["logged"] = False

            if state["consecutive"] >= SLEEP_MIN_CONSECUTIVE_FRAMES and head_drop_ratio is not None:
                cursor.execute("""
                    INSERT INTO events
                        (timestamp, person_id, event_type, confidence,
                         bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    round(current_timestamp, 3),
                    person_id,
                    "sleeping_detected",
                    round(min(1.0, head_drop_ratio), 4),
                    x1, y1, x2, y2
                ))
                state["logged"] = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
                slabel = f"SLEEPING P{person_id}"
                (sw, sh_txt), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y2), (x1 + sw + 4, y2 + sh_txt + 6), (0, 0, 200), -1)
                cv2.putText(frame, slabel, (x1 + 2, y2 + sh_txt + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for track_id in list(track_to_person.keys()):
        if track_id not in detected_track_ids:
            if track_id not in occlusion_handler.occluded:
                last_known_bbox = None
                if track_bbox_history.get(track_id):
                    last_known_bbox = track_bbox_history[track_id][-1]
                if last_known_bbox is not None:
                    occlusion_handler.update(track_id, True, last_known_bbox)

    if frame_count % 100 == 0:
        if getattr(embedder, "reid_quality", None) == "trained":
            detect_and_fix_duplicate_ids()
        else:
            print("[INFO] Skipping auto-merge this cycle - embedder isn't "
                  "reliable enough for identity merging (reid_quality="
                  f"{getattr(embedder, 'reid_quality', 'unknown')}).")
        conn.commit()
        print(f" -> Frame {frame_count}/{total_frames} | "
              f"Active persistent IDs: {sorted(set(track_to_person.values()))}")

    if frame_count % 30 == 0:
        conn.commit()

    reid_quality = getattr(embedder, "reid_quality", "unknown")
    if reid_quality == "trained":
        backend_label = "ReID: OSNet (osnet_x1_0) - RE-ID TRAINED WEIGHTS"
        backend_color = (0, 200, 0)
    elif reid_quality == "imagenet_only":
        backend_label = "ReID: OSNet arch, IMAGENET-ONLY weights (NOT re-id trained!)"
        backend_color = (0, 165, 255)
    else:
        backend_label = "ReID: FALLBACK ResNet50 (NOT re-id trained!)"
        backend_color = (0, 0, 255)
    cv2.putText(frame, backend_label, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, backend_color, 2)

    if frame_count % LIVE_FRAME_WRITE_EVERY_N_FRAMES == 0 or frame_count >= total_frames:
        cv2.imwrite(LIVE_FRAME_PATH, frame)
        print(f"PROGRESS:{frame_count}:{total_frames}")

    video_writer.write(frame)

# --- Final performance report ---
performance_monitor.print_report(frame_count, len(gallery.person_ids))

# --- FINAL COMMIT AND CLEANUP ---
conn.commit()
conn.close()
video_capture.release()
video_writer.release()

print("\n########################################")
print("  AGENT 1 COMPLETE: PERSISTENT RE-ID DONE  ")
print("########################################")
print(f"Total frames processed            : {frame_count}")
print(f"Persistent identities in gallery   : {len(gallery.person_ids)}")
print(f"Database location                 : {DATABASE_PATH}")
print(f"Video saved                       : {output_path}\n")