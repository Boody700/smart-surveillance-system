# # agent1_tracking/agent1_persistent_reid.py
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
import numpy as np
import faiss
import time
from collections import deque
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config2 import (
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
    FACE_CONF, FACE_IOU, FACE_CROP_PAD_RATIO, FACE_DETECT_EVERY_N_FRAMES
)
print("\n=== AGENT 1 STARTING: TRACKING + PERSISTENT RE-ID + DATABASE FEED ===")

# ---------------------------------------------------------------------------
# CLASS: Temporal Memory - تتبع المواقع السابقة
# ---------------------------------------------------------------------------
class TemporalMemory:
    """
    تخزين المواقع والمميزات السابقة لكل معرف لتتبع الحركة وتحسين المطابقة
    """
    def __init__(self, max_history=TEMPORAL_MEMORY_LENGTH):
        self.history = {}  # person_id -> deque of (frame, bbox, embedding)
        self.max_history = max_history
    
    def add_entry(self, person_id, frame_count, bbox, embedding):
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.max_history)
        self.history[person_id].append((frame_count, bbox, embedding))
    
    def get_spatial_score(self, person_id, current_bbox, max_distance=150):
        """حساب التشابه المكاني مع المواقع السابقة"""
        if person_id not in self.history or not self.history[person_id]:
            return 0.0
        
        # حساب مركز الصندوق الحالي
        cx1 = (current_bbox[0] + current_bbox[2]) / 2
        cy1 = (current_bbox[1] + current_bbox[3]) / 2
        
        # أقل مسافة من آخر 5 مواقع
        min_distance = float('inf')
        for _, last_bbox, _ in list(self.history[person_id])[-5:]:
            cx2 = (last_bbox[0] + last_bbox[2]) / 2
            cy2 = (last_bbox[1] + last_bbox[3]) / 2
            
            distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        # تحويل المسافة إلى درجة تشابه (0-1)
        return max(0, 1 - min_distance / max_distance)
    
    def get_temporal_score(self, person_id, current_frame, max_gap=300):
        """حساب التشابه الزمني - كلما كان آخر ظهور قريباً، زادت الثقة"""
        if person_id not in self.history or not self.history[person_id]:
            return 0.0
        
        last_frame, _, _ = self.history[person_id][-1]
        frames_gap = current_frame - last_frame
        
        if frames_gap > max_gap:
            return 0.0
        
        return max(0.5, 1 - frames_gap / max_gap)
    
    def get_predicted_bbox(self, person_id, current_bbox):
        """توقع الموقع باستخدام نموذج حركة بسيط"""
        if person_id not in self.history or len(self.history[person_id]) < 3:
            return current_bbox
        
        # حساب متوسط السرعة من آخر 5 إطارات
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
# CLASS: Occlusion Handler - التعامل مع الحجب
# ---------------------------------------------------------------------------
class OcclusionHandler:
    """
    تتبع حالات الحجب وتوقع المواقع أثناء الحجب
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
        """توسيع منطقة البحث عند الحجب"""
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

    def __init__(self, model_name=REID_MODEL_NAME, device="cpu"):
        self.device = device
        self.dim = EMBEDDING_DIM
        self.backend = None
        self.embed_cache = {}  # تخزين مؤقت للمميزات لتسريع المعالجة

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
            # FIX: this warning used to be two easy-to-miss print() lines
            # buried among hundreds of other log lines. Since the fallback
            # backend silently degrades match quality (and is the #1
            # suspect behind identity-fragmentation symptoms like the ones
            # reported on video 2), make it impossible to miss in the
            # console AND bake it into the output video itself (see the
            # on-frame banner drawn every frame below), so a reviewer
            # watching the .mp4 later doesn't have to go dig through logs
            # to know which embedder produced it.
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
            # استخدام تجزئة الصورة كمفتاح للتخزين المؤقت (تقريبياً)
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
            if len(self.embed_cache) > 100:  # تنظيف التخزين المؤقت
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
        # IndexIDMap lets us tag each vector with its own person_id and
        # remove/replace a single vector in O(1) - a plain IndexFlatIP only
        # supports appending, which forced a full index.reset() + rebuild
        # from every embedding on every single upsert() call (O(n) per
        # detection instead of O(1), and grows worse the longer the video
        # runs and the more people appear in it).
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
                # Gallery was built with a different embedder/dim - skip rather than crash.
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
        """Returns (person_id, similarity) for the best match above threshold, else (None, best_sim_seen)."""
        if self.index.ntotal == 0:
            return None, 0.0
        sims, idxs = self.index.search(vec.reshape(1, -1), min(3, self.index.ntotal))  # أفضل 3 مطابقات (أو أقل إذا كان المعرض صغيراً)

        # فحص أفضل المطابقات - مع IndexIDMap، idxs تحتوي على person_id مباشرة
        # (وليس رقم صف يحتاج لترجمة عبر self.person_ids كما كان سابقاً)
        for i in range(len(idxs[0])):
            best_sim = float(sims[0][i])
            best_person_id = int(idxs[0][i])
            if best_person_id == -1:
                continue
            if best_sim >= threshold:
                return best_person_id, best_sim

        # إذا لم يكن هناك مطابقة فوق العتبة، نعيد أفضل مطابقة مع درجة التشابه
        if len(idxs[0]) > 0 and idxs[0][0] != -1:
            return None, float(sims[0][0])
        return None, 0.0

    def upsert(self, person_id, vec, frame_count, is_new, confidence=1.0):
        """Add a brand-new identity, or fold a new sample into an existing running-average embedding."""
        cur = self.conn.cursor()

        if is_new:
            self._add_to_index(person_id, vec, 1)
            cur.execute("""
                INSERT INTO person_gallery (person_id, embedding, num_samples, first_seen_frame, last_seen_frame)
                VALUES (?, ?, 1, ?, ?)
            """, (person_id, vec.astype(np.float32).tobytes(), frame_count, frame_count))
        else:
            # المتوسط المتحرك المرجح - إعطاء وزن أكبر للمميزات الجديدة الموثوقة
            n = self.sample_counts[person_id]
            old_vec = self.embeddings[person_id]
            
            # عامل التعلم التكيفي - كلما زاد عدد العينات، قل وزن العينة الجديدة
            alpha = min(0.5, 0.3 * confidence)  # معدل التعلم التكيفي
            new_vec = (1 - alpha) * old_vec + alpha * vec
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm

            self.embeddings[person_id] = new_vec
            self.sample_counts[person_id] = n + 1
            self.last_seen_frame[person_id] = frame_count

            # تحديث المتجه في الفهرس مباشرة (O(1)) بدلاً من إعادة بناء الفهرس
            # بالكامل من كل المعرفات في كل مرة - كان هذا يصبح أبطأ كلما زاد
            # عدد الأشخاص وطالت مدة الفيديو
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
        """دمج معرفين في معرف واحد - إصلاح المعرفات المكررة"""
        if duplicate_id not in self.embeddings or primary_id not in self.embeddings:
            return
        
        print(f"[MERGE] Merging person {duplicate_id} into {primary_id}")
        
        # دمج المميزات
        primary_embed = self.embeddings[primary_id]
        duplicate_embed = self.embeddings[duplicate_id]
        n1 = self.sample_counts[primary_id]
        n2 = self.sample_counts[duplicate_id]
        
        merged_embed = (primary_embed * n1 + duplicate_embed * n2) / (n1 + n2)
        merged_embed = merged_embed / np.linalg.norm(merged_embed)
        
        # تحديث المميزات
        self.embeddings[primary_id] = merged_embed
        self.sample_counts[primary_id] = n1 + n2
        self.last_seen_frame[primary_id] = max(
            self.last_seen_frame.get(primary_id, 0),
            self.last_seen_frame.get(duplicate_id, 0)
        )
        
        # حذف المعرف المكرر من القائمة والفهرس، وتحديث متجه المعرف الأساسي
        # بالنتيجة المدموجة - كل هذا الآن O(1) بدلاً من إعادة بناء الفهرس بالكامل
        if duplicate_id in self.person_ids:
            self.person_ids.remove(duplicate_id)
        self.index.remove_ids(np.array([duplicate_id], dtype=np.int64))
        self.index.remove_ids(np.array([primary_id], dtype=np.int64))
        self.index.add_with_ids(merged_embed.reshape(1, -1), np.array([primary_id], dtype=np.int64))

        # حذف من قاعدة البيانات
        cur = self.conn.cursor()
        cur.execute("DELETE FROM person_gallery WHERE person_id = ?", (duplicate_id,))
        cur.execute("""
            UPDATE person_gallery 
            SET embedding = ?, num_samples = ?, last_seen_frame = ?
            WHERE person_id = ?
        """, (merged_embed.astype(np.float32).tobytes(), n1 + n2, frame_count, primary_id))
        self.conn.commit()
        
        # إزالة من الذاكرة المؤقتة
        self.embeddings.pop(duplicate_id, None)
        self.sample_counts.pop(duplicate_id, None)
        self.last_seen_frame.pop(duplicate_id, None)

# ---------------------------------------------------------------------------
# CLASS: Performance Monitor - مراقبة الأداء
# ---------------------------------------------------------------------------
class PerformanceMonitor:
    """مراقبة أداء النظام وتقديم تقارير"""
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
cursor.execute("PRAGMA busy_timeout=30000")  # wait up to 30s on a lock instead of failing immediately (other agents share this DB)

# Only `events` gets wiped per run - it's the per-video detection log that
# Agents 2/3/4 read. `person_gallery` is NEVER wiped here: that's what makes
# the person IDs persistent instead of resetting every run.
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

# --- SET UP OUTPUT VIDEO FOR VISUAL VERIFICATION ---
output_dir = os.path.join(ROOT_DIR, "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_agent1_persistent_reid_improved.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------
model = YOLO(MODEL_NAME)
embedder = PersonEmbedder()
gallery = PersonGallery(conn, dim=embedder.dim)

# --- PHONE DETECTION MODEL (own pass, separate from person tracking above) ---
phone_model = None
PHONE_USES_COCO_FALLBACK = False
if os.path.exists(PHONE_MODEL_PATH):
    print(f"[INFO] Loading custom phone model: {PHONE_MODEL_PATH}")
    phone_model = YOLO(PHONE_MODEL_PATH)
else:
    print(f"[WARN] Phone model not found at {PHONE_MODEL_PATH} — "
          f"falling back to COCO class 67 ('cell phone') on the main person model.")
    PHONE_USES_COCO_FALLBACK = True

# --- FACE DETECTION MODEL (own pass, scoped to each tracked person's box) ---
face_model = None
if os.path.exists(FACE_MODEL_PATH):
    print(f"[INFO] Loading face model: {FACE_MODEL_PATH}")
    face_model = YOLO(FACE_MODEL_PATH)
else:
    print(f"[WARN] Face model not found at {FACE_MODEL_PATH} — face detection disabled for this run.")

def expand_box(box, pad_ratio, fw, fh):
    """Pad a bbox by pad_ratio on each side, clamped to frame bounds."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return (max(0, x1 - px), max(0, y1 - py), min(fw - 1, x2 + px), min(fh - 1, y2 + py))

# تهيئة المكونات الإضافية
temporal_memory = TemporalMemory()
occlusion_handler = OcclusionHandler()
performance_monitor = PerformanceMonitor()

# ---------------------------------------------------------------------------
# TRACK ID -> PERSISTENT PERSON ID CACHE
# ---------------------------------------------------------------------------
track_to_person = {}          # bot_sort track_id -> persistent person_id
track_bbox_history = {}       # track_id -> list of recent bboxes
frames_since_recheck = {}     # bot_sort track_id -> frames since last embedding refresh
track_hit_counts = {}         # bot_sort track_id -> consecutive frames seen (probation)
person_id_to_track = {}       # person_id -> current track_id (للكشف عن تغييرات المعرف)

frame_count = 0

def get_adjusted_bbox(track_id, bbox):
    """تعديل الصندوق حسب آخر حجم معروف للشخص"""
    if track_id not in track_bbox_history:
        track_bbox_history[track_id] = []
    
    track_bbox_history[track_id].append(bbox)
    if len(track_bbox_history[track_id]) > 10:
        track_bbox_history[track_id].pop(0)
    
    if len(track_bbox_history[track_id]) >= 5:
        heights = [b[3] - b[1] for b in track_bbox_history[track_id][-5:]]
        avg_height = np.mean(heights)
        current_height = bbox[3] - bbox[1]
        
        # إذا تغير الارتفاع بنسبة كبيرة، استخدم متوسط الارتفاع
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
    """تجاهل المقاطع الصغيرة جداً أو غير الواضحة"""
    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        return False
    
    # فحص الوضوح باستخدام تباين Laplacian
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score > blur_threshold

def get_ranked_candidates(track_id, track_person_id, current_bbox, embedding, frame_count, top_k=3):
    """
    FIX (core bug from the screenshot - 3 different people all labeled ID:1
    at once): the old match_with_hybrid() returned only a SINGLE best
    person_id per track, computed completely independently for every track
    in the frame. Nothing stopped two (or three) different tracks in the
    SAME frame from each independently deciding "my best match is person 1"
    - there was no cross-track bookkeeping at all, so all three got ID:1
    simultaneously.

    That can't be fixed inside a per-track function alone - it needs a
    frame-level conflict resolver (see PASS 2 in the main loop below) that
    looks at every track's proposal together and enforces one person_id per
    frame. But the resolver needs each track to offer more than one option,
    otherwise the "loser" of a conflict has nothing to fall back on except
    "spawn a brand new identity" even when its correct identity was simply
    claimed first by a higher-scoring track this frame.

    So this function keeps the same visual + spatial + temporal hybrid-score
    math as before, but returns a RANKED LIST of (person_id, score)
    candidates instead of one winner. The frame-level resolver then walks
    each track's list and gives it the best candidate that hasn't already
    been claimed by a stronger track this frame.
    """
    candidates = {}  # person_id -> best hybrid score seen for it via any path below

    # Path 1: fast-path continuity with this track's own current identity.
    if track_person_id is not None and track_person_id in gallery.embeddings:
        current_embed = gallery.embeddings[track_person_id]
        visual_sim = float(np.dot(embedding, current_embed))
        if visual_sim > 0.6:
            candidates[track_person_id] = max(candidates.get(track_person_id, -1.0), visual_sim)

    # Path 2: search the whole gallery for the top-k candidates, same hybrid
    # scoring (visual as a floor, spatial/temporal as a bonus on top - see
    # the reasoning preserved from the original bug fix comment) but for
    # EVERY candidate above threshold, not just the single best one.
    if gallery.index.ntotal > 0:
        k = min(top_k, gallery.index.ntotal)
        sims, idxs = gallery.index.search(embedding.reshape(1, -1), k)
        for i in range(len(idxs[0])):
            person_id = int(idxs[0][i])
            if person_id == -1:
                continue
            visual_sim = float(sims[0][i])
            if visual_sim < REID_MATCH_THRESHOLD:
                continue  # gallery.match()'s original threshold gate, preserved per-candidate

            spatial_score = temporal_memory.get_spatial_score(person_id, current_bbox)
            temporal_score = temporal_memory.get_temporal_score(person_id, frame_count)
            bonus = (
                spatial_score * HYBRID_MATCH_WEIGHTS['spatial'] +
                temporal_score * HYBRID_MATCH_WEIGHTS['temporal']
            )
            # visual_sim is a floor, spatial/temporal only add confidence on
            # top of it - never subtract - for the same reason as before:
            # a person absent for a long time (temporal -> 0) reappearing
            # somewhere new (spatial -> 0) must not get penalized below an
            # already-valid visual match.
            hybrid_score = min(1.0, visual_sim + (1 - visual_sim) * bonus)
            candidates[person_id] = max(candidates.get(person_id, -1.0), hybrid_score)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return ranked

def detect_and_fix_duplicate_ids():
    """كشف وإصلاح المعرفات المكررة (نفس الشخص بمعرفات مختلفة)"""
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
            
            # مقارنة المميزات بين المعرفين
            if id1 in gallery.embeddings and id2 in gallery.embeddings:
                sim = np.dot(gallery.embeddings[id1], gallery.embeddings[id2])
                if sim > 0.90:  # عتبة عالية للدمج
                    group.append(id2)
                    checked.add(id2)
        
        if len(group) > 1:
            duplicate_groups.append(group)
            checked.update(group)
    
    # دمج المعرفات المكررة
    for group in duplicate_groups:
        primary_id = min(group)
        for duplicate_id in group:
            if duplicate_id != primary_id:
                gallery.merge_identities(primary_id, duplicate_id, frame_count)
                
                # تحديث خريطة track_to_person
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

    # تحديث حالة الحجب لكل معرف
    # FIX: OCCLUSION EVICTION BUG - OcclusionHandler.max_frames (from
    # MAX_FRAMES_OCCLUDED in config.py) was stored on the object but NEVER
    # actually checked anywhere in the file. That meant an occluded
    # track_id could sit in occlusion_handler.occluded forever once
    # BoT-SORT itself recycled that track_id (after its own track_buffer of
    # 300 frames), leaking memory and, more importantly, meaning a
    # long-absent track never got explicitly "given up on" - long-gap
    # re-identification was silently relying 100% on the FAISS gallery
    # search in get_ranked_candidates() rather than this handler doing
    # anything useful past a few seconds. This loop now actually evicts
    # entries once max_frames is exceeded, and drops the stale
    # track_to_person mapping so a reappearing person is forced through a
    # fresh FAISS match (the mechanism that's actually designed for
    # long-gap recovery) instead of lingering on dead bookkeeping.
    for track_id in list(occlusion_handler.occluded.keys()):
        if track_id in track_to_person:
            # إذا كان المعرف لا يزال موجوداً ولم يظهر، نستمر في التتبع
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
        classes=[0],     # Only detect people (class 0 in COCO dataset)
        # conf: minimum YOLO confidence to keep a "person" detection at all.
        # Now pulled from config.py (DETECTION_CONF_THRESHOLD) instead of a
        # hardcoded literal here, so it's visible and tunable in one place.
        conf=DETECTION_CONF_THRESHOLD,
        # iou: NMS overlap threshold for merging boxes - unrelated to conf.
        # FIX: was 0.20 - far too aggressive. At 0.20, two people's boxes
        # overlapping by just 20% get merged into one, so people standing
        # near each other were drawn as a single oversized box. 0.5 is the
        # standard NMS threshold and keeps each person's box tight.
        iou=0.5,
        imgsz=1536,      # Higher resolution scan to catch far-away people
        verbose=False
    )

    frame_results = results[0]
    detected_track_ids = []

    # --- PASS 1: gather every detection that survives probation, compute
    # crops/embeddings, and collect RANKED candidate identities per track -
    # without committing anything to track_to_person / gallery / DB yet.
    # Committing immediately (as the old code did) is exactly what let two
    # different tracks in the same frame both grab person_id 1: each track
    # was resolved in isolation with no idea what the others were doing.
    reembed_proposals = []   # tracks that ran matching this frame, need conflict resolution
    passthrough_hits = []    # tracks that kept their existing identity, no re-matching this frame
    claimed_this_frame = set()  # person_ids already "in use" somewhere in this frame

    if frame_results.boxes.id is not None:
        track_ids = frame_results.boxes.id.int().tolist()
        boxes = frame_results.boxes.xyxy.int().tolist()
        confidences = frame_results.boxes.conf.tolist()

        for track_id, box, conf in zip(track_ids, boxes, confidences):
            performance_monitor.log_detection()
            detected_track_ids.append(track_id)

            # هذا الشخص ظهر مجدداً - تحقق مما إذا كان مُعلَّماً كمحجوب *قبل*
            # حذفه من occlusion_handler (كان الكود القديم يحذفه فوراً هنا، مما
            # يجعل فحص get_search_region أدناه ميت الكود دائماً لأنه لم يكن
            # يجد track_id في occluded بعد الآن)
            was_occluded = track_id in occlusion_handler.occluded
            search_region = occlusion_handler.get_search_region(track_id, frame.shape) if was_occluded else None
            if was_occluded:
                del occlusion_handler.occluded[track_id]

            # --- Probation: لا نثق بالمعرف الجديد فوراً ---
            track_hit_counts[track_id] = track_hit_counts.get(track_id, 0) + 1
            if track_hit_counts[track_id] < MIN_FRAMES_TO_CONFIRM:
                continue

            # --- تعديل الصندوق حسب حجم الشخص ---
            adjusted_box = get_adjusted_bbox(track_id, box)

            # --- التعامل مع الحجب: توسيع منطقة البحث إذا كان هذا الشخص
            # محجوباً في الإطارات الأخيرة ---
            if search_region:
                adjusted_box = search_region

            x1, y1, x2, y2 = adjusted_box
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame_width, x2), min(frame_height, y2)
            crop = frame[y1c:y2c, x1c:x2c]

            # التحقق من جودة المقاطع
            crop_is_usable = crop.size > 0 and filter_crop(crop)

            # --- الحصول على المميزات ---
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
                # لم يُعَد استخراج مميزات هذا التتبع في هذا الإطار، لكنه ما
                # زال يحمل معرّفاً حالياً - يجب اعتبار هذا المعرف "محجوزاً"
                # في هذا الإطار بالذات، وإلا فقد يُعطى لاحقاً في PASS 2 لتتبع
                # آخر يقترحه أيضاً (وهذا كان جزءاً من نفس الثغرة).
                if current_person_id is not None:
                    claimed_this_frame.add(current_person_id)
                passthrough_hits.append({"track_id": track_id, "box": adjusted_box, "conf": conf})

    # --- PASS 2: resolve identity conflicts across ALL tracks in this frame.
    # Process the strongest proposals first so a track with a clear, high-
    # confidence match claims its person_id before a weaker/ambiguous
    # proposal can grab it. Any track whose top choice is already claimed
    # falls through to its next-best ranked candidate; if it runs out of
    # candidates entirely, it gets a genuinely new identity instead of
    # colliding with someone else's ID (this is the direct fix for the
    # "three people all labeled ID:1" bug in the screenshot).
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
            # مطابقة ناجحة (قد تكون المرشح الأول أو مرشح بديل بعد تعارض)
            if current_person_id is not None and current_person_id != assigned_id:
                performance_monitor.log_id_switch(current_person_id, assigned_id, "reid_match")

            performance_monitor.log_identity(is_new=False)
            # FIX (gallery drift): confidence_factor used to come only from
            # YOLO's detection confidence (how sure the model is "this is a
            # person"), never from how well the crop actually matched the
            # stored identity. A confidently-detected person under bad
            # lighting/pose/partial occlusion could pass filter_crop() and
            # still get folded into the running-average embedding at nearly
            # full weight even on a borderline match, gradually pulling the
            # gallery's embedding away from the person's typical appearance.
            # Blending in assigned_score (the re-id match confidence) means
            # a marginal match near REID_MATCH_THRESHOLD nudges the average
            # only slightly, while a strong, clear match still updates it
            # close to full strength.
            confidence_factor = min(1.0, conf * 1.2 * assigned_score)
            gallery.upsert(assigned_id, vec, frame_count, is_new=False, confidence=confidence_factor)
            print(f"[RE-ID] Track {track_id} -> Person {assigned_id} (sim: {assigned_score:.3f})")
        else:
            # FIX (core bug behind "ID:3 -> ID:6" for the same continuously-
            # tracked person): this branch used to ALWAYS spawn a brand-new
            # identity whenever no unclaimed candidate cleared the
            # threshold - even for a track that already had a stable
            # current_person_id and never lost its BoT-SORT track_id. A
            # track's own continuity (same track_id, never re-spawned by
            # the tracker) is itself strong evidence of same identity -
            # often stronger than a single re-embed snapshot taken right
            # when the person turned around, changed pose, or got
            # partially occluded. Losing that identity anchor just because
            # one recheck cycle came back ambiguous was needlessly
            # destructive. Now: if this track already had an identity and
            # nobody else claimed it this frame, keep it. Only spawn a
            # genuinely new person_id when the track had no prior identity
            # at all (a real brand-new track) or its old identity was
            # legitimately claimed by a stronger proposal this frame.
            if current_person_id is not None and current_person_id not in claimed_this_frame:
                assigned_id = current_person_id
                performance_monitor.log_identity(is_new=False)
                gallery.upsert(assigned_id, vec, frame_count, is_new=False,
                                confidence=min(1.0, conf * 0.5))  # low confidence: ambiguous recheck, nudge gallery gently
                best_sim = p["candidates"][0][1] if p["candidates"] else 0.0
                print(f"[KEEP] Track {track_id} -> Person {assigned_id} "
                      f"(no confident re-match, best alt sim: {best_sim:.3f} - kept prior identity)")
            else:
                # شخص جديد فعلاً (أو تعارض بدون أي مرشح بديل متاح)
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

    # --- PASS 3: write DB rows + draw boxes for everything detected this
    # frame, now that every track_id -> person_id mapping is final and
    # conflict-free for this frame.
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

        # رسم صندوق وتعليق أصغر وأضيق على الفيديو - لون أخضر للمعرفات المستقرة
        color = (0, 255, 0) if person_id in gallery.embeddings else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"ID:{person_id}"
        font_scale = 0.32
        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - txt_h - 4), (x1 + txt_w + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    # --- PASS 3b: PHONE DETECTION — own model/pass, not part of person
    # tracking above. No association to a specific person_id here (matches
    # the original single-model pipeline's behavior); a rule engine can
    # associate phone boxes with the nearest person bbox downstream if needed.
    if frame_count % PHONE_DETECT_EVERY_N_FRAMES == 0:
        if phone_model is not None:
            phone_results = phone_model.predict(
                frame, conf=PHONE_CONF, iou=PHONE_IOU, imgsz=1536, verbose=False
            )
        else:
            phone_results = model.predict(
                frame, classes=[67], conf=PHONE_COCO_FALLBACK_CONF,
                iou=PHONE_COCO_FALLBACK_IOU, imgsz=1536, verbose=False
            )

        pboxes = phone_results[0].boxes
        if pboxes is not None and len(pboxes) > 0:
            for pbox, pconf in zip(pboxes.xyxy.int().tolist(), pboxes.conf.tolist()):
                px1, py1, px2, py2 = pbox
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

    # --- PASS 3c: FACE DETECTION — own model/pass, scoped to each confirmed
    # person's own bbox this frame (padded), throttled since faces don't need
    # re-checking every single frame. Logged with person_id = that person's
    # persistent ID, so it's directly attributable per confirmed identity.
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

            face_results = face_model.predict(crop, conf=FACE_CONF, iou=FACE_IOU, imgsz=320, verbose=False)
            fboxes = face_results[0].boxes
            if fboxes is None or len(fboxes) == 0:
                continue

            fconfs = fboxes.conf.tolist()
            best_i = max(range(len(fconfs)), key=lambda i: fconfs[i])
            fx1, fy1, fx2, fy2 = fboxes.xyxy.int().tolist()[best_i]
            fconf = fconfs[best_i]

            # translate the face box from crop-local coords back to full-frame
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

    # --- كشف المعرفات التي اختفت ---
    for track_id in list(track_to_person.keys()):
        if track_id not in detected_track_ids:
            # هذا المعرف لم يظهر في الإطار الحالي
            if track_id not in occlusion_handler.occluded:
                # استخدام آخر صندوق حقيقي معروف لهذا المعرف بدلاً من None -
                # get_search_region يفك bbox[0]..bbox[3] حسابياً، وتمرير None
                # هنا كان سيسبب TypeError أول مرة تُستخدم فيها منطقة البحث
                last_known_bbox = None
                if track_bbox_history.get(track_id):
                    last_known_bbox = track_bbox_history[track_id][-1]
                if last_known_bbox is not None:
                    occlusion_handler.update(track_id, True, last_known_bbox)

    # --- كشف وإصلاح المعرفات المكررة كل 100 إطار ---
    # FIX: only run auto-merge when using the real OSNet embedder. The
    # ResNet50 fallback is a generic ImageNet classifier, not trained for
    # person re-id - two DIFFERENT people wearing similarly-colored clothes
    # can easily cross the 0.85 merge threshold on that embedder, which is
    # exactly what the switch log showed (8->7, 9->2, 1->7, 2->7, 7->6 -
    # distinct people getting collapsed into the same ID). Merging on an
    # unreliable embedder does more harm than good, so skip it until OSNet
    # (embedder.backend == "torchreid") is actually loaded.
    if frame_count % 100 == 0:
        # FIX: was `embedder.backend == "torchreid"`, which is true even
        # when OSNet loaded with untrained ImageNet-only weights (see
        # PersonEmbedder.__init__) - that embedder is just as unreliable
        # for merging as the ResNet50 fallback, so gate on reid_quality
        # instead of just "did torchreid technically load".
        if getattr(embedder, "reid_quality", None) == "trained":
            detect_and_fix_duplicate_ids()
        else:
            print("[INFO] Skipping auto-merge this cycle - embedder isn't "
                  "reliable enough for identity merging (reid_quality="
                  f"{getattr(embedder, 'reid_quality', 'unknown')}).")
        conn.commit()
        print(f" -> Frame {frame_count}/{total_frames} | "
              f"Active persistent IDs: {sorted(set(track_to_person.values()))}")

    # Commit to database every 30 frames
    if frame_count % 30 == 0:
        conn.commit()

    # --- FIX: burn the active ReID backend into every frame of the output
    # video. This was the single most useful piece of missing information
    # when reviewing a recorded run after the fact - without it you can't
    # tell from the .mp4 alone whether identity switches came from a
    # genuinely hard case (occlusion, pose) or simply from the ResNet50
    # fallback being silently active instead of OSNet. ---
    reid_quality = getattr(embedder, "reid_quality", "unknown")
    if reid_quality == "trained":
        backend_label = "ReID: OSNet (osnet_x0_25) - RE-ID TRAINED WEIGHTS"
        backend_color = (0, 200, 0)  # green = the real, trained re-id model
    elif reid_quality == "imagenet_only":
        backend_label = "ReID: OSNet arch, IMAGENET-ONLY weights (NOT re-id trained!)"
        backend_color = (0, 165, 255)  # orange = loaded fine, but not production quality
    else:
        backend_label = "ReID: FALLBACK ResNet50 (NOT re-id trained!)"
        backend_color = (0, 0, 255)  # red = degraded matching, expect switches
    cv2.putText(frame, backend_label, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, backend_color, 2)

    # Write annotated frame to output video
    video_writer.write(frame)

# --- تقرير الأداء النهائي ---
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