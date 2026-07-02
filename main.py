"""
surveillance.py – Office CCTV Engine (Final Production)
- ByteTrack, OSNet MSMT17, temporal voting (3 confirmations)
- Active owner timeout (2 sec) – prevents stale locks
- Released similarity gate 0.72 – realistic CCTV matches
- Stores up to 12 quality‑scored embeddings per person
"""

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import warnings
import threading
from collections import deque, Counter
import torch
import torchreid
import sqlite3
import pickle

warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ==================================================================
# CONFIGURATION
# ==================================================================
REID_THRESHOLD = 0.68                # min similarity for a potential match
RELEASED_GATE = 0.72                 # if ID recently released, require >=0.72 to reuse
MIN_FRAMES_BEFORE_ID = 10            # frames to observe before new ID
CONFIRMATION_NEEDED = 3              # temporal voting: need 3 consistent matches
CONFIRMATION_WINDOW = 5.0            # seconds to collect confirmations
COOLDOWN_REID = 1.0                  # seconds between ReID attempts
EMBEDDING_ALPHA = 0.90               # track EMA smoothing (for matching, not DB)
ACTIVE_TRACK_TIMEOUT = 45.0
OWNER_TIMEOUT = 2.0                  # release owner if track not seen for 2 sec
RECENT_RELEASE_SECONDS = 30.0
MIN_CROP_HEIGHT = 70
MAX_MOVEMENT_SKIP = 180
SNAPSHOT_INTERVAL_SEC = 10.0
MIN_BOX_AREA = 12000
MIN_QUALITY_FOR_STORAGE = 0.65       # quality = crop_height / 160
MAX_EMBEDDINGS_PER_PERSON = 12       # increased from 5

# Model paths
POSE_MODEL_PATH = "models/yolo11m-pose.pt"
OBJ_MODEL_PATH = "models/yolov8n.pt"
REID_WEIGHTS_PATH = "osnet_x1_0_msmt17.pth"
BRAIN_MODEL_PATH = "models/office_action_model.pkl"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ==================================================================
# OSNet ReID (MSMT17 weights – classifier removed)
# ==================================================================
def load_reid_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=False
    )
    if os.path.exists(REID_WEIGHTS_PATH):
        state_dict = torch.load(REID_WEIGHTS_PATH, map_location='cpu')
        for key in list(state_dict.keys()):
            if key.startswith('classifier.'):
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        print("[Surveillance] Loaded OSNet MSMT17 weights")
    else:
        print(f"[WARNING] {REID_WEIGHTS_PATH} missing – ReID will be weak")
    model.eval()
    model.classifier = torch.nn.Identity()
    return model

reid_model = load_reid_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reid_model = reid_model.to(device)

def extract_embedding(crop_bgr, box_area=None):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_HEIGHT:
        return None, 0.0
    if box_area is not None and box_area < MIN_BOX_AREA:
        return None, 0.0
    quality = min(1.0, h / 160.0)
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (128, 256))
        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = (img - mean) / std
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = reid_model(img)
        feat = feat.cpu().numpy().flatten()
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm, quality
    except:
        return None, 0.0

# ==================================================================
# SQLite Database – Counter table + quality‑scored embeddings
# ==================================================================
DB_PATH = "persons.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS persons
                 (id TEXT PRIMARY KEY,
                  first_seen REAL,
                  last_seen REAL,
                  first_camera INTEGER,
                  last_camera INTEGER,
                  embeddings BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS id_counter
                 (id INTEGER PRIMARY KEY)''')
    c.execute("INSERT OR IGNORE INTO id_counter (id) VALUES (0)")
    conn.commit()
    conn.close()
init_db()

def get_next_person_id():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    c = conn.cursor()
    c.execute("UPDATE id_counter SET id = id + 1")
    c.execute("SELECT id FROM id_counter")
    new_id = c.fetchone()[0]
    conn.close()
    return str(new_id).zfill(3)

def find_best_match(embedding, min_sim=REID_THRESHOLD):
    if embedding is None:
        return None, 0.0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, embeddings FROM persons")
    best_id, best_sim = None, 0.0
    for pid, blob in c.fetchall():
        emb_list = pickle.loads(blob)
        max_sim = 0.0
        for emb, quality, _ in emb_list:
            sim = np.dot(embedding, emb)
            if sim > max_sim:
                max_sim = sim
        if max_sim > best_sim:
            best_sim = max_sim
            best_id = pid
    conn.close()
    return (best_id, best_sim) if best_sim >= min_sim else (None, best_sim)

def save_person(person_id, new_embedding, camera_id, quality):
    if quality < MIN_QUALITY_FOR_STORAGE:
        return
    new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT embeddings FROM persons WHERE id=?", (person_id,))
    row = c.fetchone()
    if row:
        emb_list = pickle.loads(row[0])
        # deduplicate: skip if very similar
        should_add = True
        for emb, q, ts in emb_list:
            sim = np.dot(new_embedding, emb)
            if sim > 0.92:
                should_add = False
                break
        if should_add:
            emb_list.append((new_embedding, quality, time.time()))
            if len(emb_list) > MAX_EMBEDDINGS_PER_PERSON:
                emb_list.pop(0)
        c.execute("UPDATE persons SET last_seen=?, last_camera=?, embeddings=? WHERE id=?",
                  (time.time(), camera_id, pickle.dumps(emb_list), person_id))
    else:
        emb_list = [(new_embedding, quality, time.time())]
        c.execute("INSERT INTO persons VALUES (?,?,?,?,?,?)",
                  (person_id, time.time(), time.time(), camera_id, camera_id, pickle.dumps(emb_list)))
    conn.commit()
    conn.close()

# ==================================================================
# Pose feature extraction (21 features)
# ==================================================================
def angle_between(a,b,c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    cos_a = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1.0,1.0))))

def point_distance(a,b):
    return float(np.linalg.norm(np.array(a)-np.array(b)))

def extract_features(kp):
    features = []
    def pt(i): return kp[i][:2]
    def vis(i): return kp[i][0] > 5 and kp[i][1] > 5

    features.append(angle_between(pt(5),pt(7),pt(9))    if all(vis(i) for i in [5,7,9])    else 0.0)
    features.append(angle_between(pt(6),pt(8),pt(10))   if all(vis(i) for i in [6,8,10])   else 0.0)
    features.append(angle_between(pt(11),pt(5),pt(7))   if all(vis(i) for i in [11,5,7])   else 0.0)
    features.append(angle_between(pt(12),pt(6),pt(8))   if all(vis(i) for i in [12,6,8])   else 0.0)
    features.append(angle_between(pt(5),pt(11),pt(13))  if all(vis(i) for i in [5,11,13])  else 0.0)
    features.append(angle_between(pt(6),pt(12),pt(14))  if all(vis(i) for i in [6,12,14])  else 0.0)
    features.append(angle_between(pt(11),pt(13),pt(15)) if all(vis(i) for i in [11,13,15]) else 0.0)
    features.append(angle_between(pt(12),pt(14),pt(16)) if all(vis(i) for i in [12,14,16]) else 0.0)
    if all(vis(i) for i in [0,5,6]):
        mid_sh = (pt(5)+pt(6))/2
        features.append(angle_between(pt(0), mid_sh, mid_sh+np.array([0,100])))
    else:
        features.append(0.0)
    sh_w = max(point_distance(pt(5),pt(6)),1.0) if vis(5) and vis(6) else 1.0
    features.append(point_distance(pt(9),pt(0))/sh_w if vis(9) and vis(0) else 5.0)
    features.append(point_distance(pt(10),pt(0))/sh_w if vis(10) and vis(0) else 5.0)
    features.append(point_distance(pt(9),pt(11))/sh_w if vis(9) and vis(11) else 5.0)
    features.append(point_distance(pt(10),pt(12))/sh_w if vis(10) and vis(12) else 5.0)
    features.append(point_distance(pt(9),pt(10))/sh_w if vis(9) and vis(10) else 0.0)
    if vis(0) and vis(11) and vis(12):
        mid_hip = (pt(11)+pt(12))/2
        features.append(point_distance(pt(0),mid_hip)/sh_w)
    else:
        features.append(0.0)
    if vis(0) and vis(5) and vis(6):
        mid_sh_y = (pt(5)[1]+pt(6)[1])/2
        features.append((pt(0)[1]-mid_sh_y)/sh_w)
    else:
        features.append(0.0)
    features.append(1.0 if (vis(9) and vis(5) and pt(9)[1] < pt(5)[1]) else 0.0)
    features.append(1.0 if (vis(10) and vis(6) and pt(10)[1] < pt(6)[1]) else 0.0)
    if vis(5) and vis(6):
        dy = pt(5)[1]-pt(6)[1]; dx = pt(5)[0]-pt(6)[0]+1e-6
        features.append(float(np.degrees(np.arctan2(dy,dx))))
    else:
        features.append(0.0)
    features.append((pt(11)[1]-pt(5)[1])/sh_w if vis(5) and vis(11) else 0.0)
    features.append(0.0)
    return features

# ==================================================================
# Stream reader
# ==================================================================
import queue as _queue
class RawStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = _queue.Queue(maxsize=1)
        self.stopped = False
        threading.Thread(target=self._reader, daemon=True).start()
    def _reader(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1); continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1); continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            self.q.put(frame)
    def get_frame(self):
        try: return self.q.get(timeout=3.0)
        except: return None

# ==================================================================
# MAIN SURVEILLANCE
# ==================================================================
def run_surveillance(rtsp_url, camera_id):
    stream = RawStream(rtsp_url)
    pose_model = YOLO(POSE_MODEL_PATH)
    obj_model = YOLO(OBJ_MODEL_PATH)

    tracker_type = "bytetrack.yaml"
    track_state = {}
    active_id_owner = {}
    recently_released_ids = {}
    pose_windows = {}
    last_cleanup = time.time()

    brain = None
    if os.path.exists(BRAIN_MODEL_PATH):
        import joblib
        brain = joblib.load(BRAIN_MODEL_PATH)
        print("[Surveillance] Action brain loaded")

    print(f"[Cam{camera_id}] Started – ByteTrack, confirm={CONFIRMATION_NEEDED}, frames={MIN_FRAMES_BEFORE_ID}")

    while True:
        frame = stream.get_frame()
        if frame is None: continue
        display = cv2.resize(frame, (854, 480))
        H, W = display.shape[:2]
        now = time.time()

        # Cleanup dead tracks and stale owners
        if now - last_cleanup > 4.0:
            dead = [tid for tid, ts in track_state.items()
                    if now - ts.get("last_seen", now) > ACTIVE_TRACK_TIMEOUT]
            for tid in dead:
                pid = track_state[tid].get("person_id")
                if pid and pid in active_id_owner:
                    del active_id_owner[pid]
                    recently_released_ids[pid] = now
                del track_state[tid]
                if tid in pose_windows: del pose_windows[tid]
            # Release owners that have been inactive for OWNER_TIMEOUT seconds
            for pid, owner_tid in list(active_id_owner.items()):
                if owner_tid in track_state:
                    if now - track_state[owner_tid]["last_seen"] > OWNER_TIMEOUT:
                        del active_id_owner[pid]
                        recently_released_ids[pid] = now
                else:
                    # owner track no longer exists
                    del active_id_owner[pid]
                    recently_released_ids[pid] = now
            # Clean old release entries
            for pid in list(recently_released_ids.keys()):
                if now - recently_released_ids[pid] > RECENT_RELEASE_SECONDS:
                    del recently_released_ids[pid]
            last_cleanup = now

        # Phone detection
        try:
            obj_r = obj_model(display, conf=0.4, verbose=False, imgsz=320)
            phone_boxes = []
            for b in obj_r[0].boxes:
                if obj_model.names[int(b.cls[0])] == 'cell phone':
                    c = b.xyxy[0].cpu().numpy()
                    phone_boxes.append(c)
        except:
            phone_boxes = []

        # ByteTrack tracking
        try:
            pose_r = pose_model.track(display, persist=True, tracker=tracker_type,
                                      conf=0.40, verbose=False, imgsz=640)
        except:
            continue

        for r in pose_r:
            if r.keypoints is None or len(r.keypoints.data)==0: continue
            if r.boxes is None or r.boxes.id is None: continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            for i, kpt in enumerate(r.keypoints.data):
                if i >= len(ids): continue
                tid = int(ids[i])
                kp = kpt.cpu().numpy()
                rx, ry = int(kp[0][0]), int(kp[0][1])
                if rx<=0 and ry<=0: continue

                bbox = (float(boxes[i][0]), float(boxes[i][1]),
                        float(boxes[i][2]), float(boxes[i][3])) if i<len(boxes) else \
                       (float(rx-40), float(ry-60), float(rx+40), float(ry+80))

                x1b, y1b, x2b, y2b = map(int, bbox)
                box_area = (x2b - x1b) * (y2b - y1b)

                # Init new track
                if tid not in track_state:
                    track_state[tid] = {
                        "person_id": None,
                        "embedding": None,
                        "conf": 0.0,
                        "last_reid_time": 0.0,
                        "last_seen": now,
                        "new_flash": 0.0,
                        "history": deque(maxlen=40),
                        "history_counter": Counter(),
                        "seen_frames": 0,
                        "prev_center": (rx, ry),
                        "last_snap_time": 0,
                        "candidate_id": None,
                        "candidate_score_sum": 0.0,
                        "candidate_hits": 0,
                        "candidate_first_seen": 0.0,
                    }
                    pose_windows[tid] = deque(maxlen=10)

                ts = track_state[tid]
                ts["last_seen"] = now
                ts["seen_frames"] += 1

                # Motion stability
                cx, cy = rx, ry
                prev_cx, prev_cy = ts.get("prev_center", (cx, cy))
                movement = np.hypot(cx - prev_cx, cy - prev_cy)
                ts["prev_center"] = (cx, cy)
                too_much_motion = movement > MAX_MOVEMENT_SKIP

                # Crop & embedding
                pad = 30
                x1 = max(0, x1b-pad); y1 = max(0, y1b-pad)
                x2 = min(W, x2b+pad); y2 = min(H, y2b+pad)
                crop = display[y1:y2, x1:x2]
                emb, quality = (None, 0.0)
                if not too_much_motion and crop.size > 0:
                    emb, quality = extract_embedding(crop, box_area=box_area)

                # Update track embedding (EMA)
                if emb is not None:
                    if ts["embedding"] is None:
                        ts["embedding"] = emb
                    else:
                        ts["embedding"] = (EMBEDDING_ALPHA * ts["embedding"] +
                                           (1 - EMBEDDING_ALPHA) * emb)
                        ts["embedding"] /= (np.linalg.norm(ts["embedding"])+1e-8)

                # ----- Re‑ID with temporal voting -----
                if ts["person_id"] is None and ts["embedding"] is not None:
                    if now - ts["last_reid_time"] >= COOLDOWN_REID:
                        ts["last_reid_time"] = now
                        matched_id, score = find_best_match(ts["embedding"], min_sim=REID_THRESHOLD)

                        # Gate for recently released IDs (use RELEASED_GATE = 0.72)
                        if matched_id and matched_id in recently_released_ids:
                            if score < RELEASED_GATE:
                                matched_id = None

                        # Active owner check with stale detection
                        if matched_id:
                            if matched_id in active_id_owner:
                                owner_tid = active_id_owner[matched_id]
                                # If owner track still exists and is recent, reject
                                if owner_tid in track_state:
                                    if now - track_state[owner_tid].get("last_seen", 0) <= OWNER_TIMEOUT:
                                        matched_id = None
                                    else:
                                        # stale owner – take over
                                        del active_id_owner[matched_id]
                                        recently_released_ids[matched_id] = now
                                else:
                                    # owner track gone – take over
                                    del active_id_owner[matched_id]
                                    recently_released_ids[matched_id] = now
                            if matched_id:
                                active_id_owner[matched_id] = tid

                        # Temporal voting
                        if matched_id:
                            if ts["candidate_id"] == matched_id:
                                ts["candidate_score_sum"] += score
                                ts["candidate_hits"] += 1
                            else:
                                ts["candidate_id"] = matched_id
                                ts["candidate_score_sum"] = score
                                ts["candidate_hits"] = 1
                                ts["candidate_first_seen"] = now

                            # Confirm if enough hits within window
                            if (ts["candidate_hits"] >= CONFIRMATION_NEEDED and
                                now - ts["candidate_first_seen"] <= CONFIRMATION_WINDOW):
                                ts["person_id"] = matched_id
                                ts["conf"] = ts["candidate_score_sum"] / ts["candidate_hits"]
                                ts["new_flash"] = now
                                save_person(matched_id, emb, camera_id, quality)
                                print(f"[Cam{camera_id}] Track {tid} → ID {matched_id} "
                                      f"(score {ts['conf']:.2f}, confirms={CONFIRMATION_NEEDED})")
                                # Clear voting
                                ts["candidate_id"] = None
                                ts["candidate_hits"] = 0
                        else:
                            # No match – after enough frames, register new person
                            if ts["seen_frames"] >= MIN_FRAMES_BEFORE_ID:
                                new_id = get_next_person_id()
                                ts["person_id"] = new_id
                                ts["conf"] = 1.0
                                ts["new_flash"] = now
                                active_id_owner[new_id] = tid
                                save_person(new_id, emb, camera_id, quality)
                                print(f"[Cam{camera_id}] NEW person after {ts['seen_frames']} frames → ID {new_id}")
                                ts["candidate_id"] = None
                                ts["candidate_hits"] = 0

                # Save snapshot (throttled)
                if ts["person_id"] and crop.size > 0:
                    if now - ts.get("last_snap_time", 0) >= SNAPSHOT_INTERVAL_SEC:
                        person_dir = os.path.join(SNAPSHOT_DIR, ts["person_id"])
                        os.makedirs(person_dir, exist_ok=True)
                        timestamp = int(now * 1000)
                        snap_path = os.path.join(person_dir, f"{timestamp}.jpg")
                        cv2.imwrite(snap_path, crop)
                        ts["last_snap_time"] = now

                # Action recognition (optional)
                pose_feats = extract_features(kp)
                pose_windows[tid].append(pose_feats)
                raw_action = "analyzing"
                if brain and len(pose_windows[tid]) >= 3:
                    avg_feats = np.mean(list(pose_windows[tid]), axis=0)
                    probs = brain.predict_proba(avg_feats.reshape(1,-1))
                    raw_action = brain.predict(avg_feats.reshape(1,-1))[0]
                ts["history"].append(raw_action)
                ts["history_counter"][raw_action] += 1
                smooth_action = ts["history_counter"].most_common(1)[0][0] if ts["history_counter"] else "analyzing"

                # Phone override
                def phone_near(kp, boxes, fw):
                    if not boxes: return False
                    def v(i): return kp[i][0]>5
                    hands = []
                    for idx in [9,10]:
                        if v(idx): hands.append((kp[idx][0],kp[idx][1]))
                    if not hands:
                        for idx in [7,8]:
                            if v(idx): hands.append((kp[idx][0],kp[idx][1]))
                    if not hands: return False
                    thr = fw * 0.12
                    for box in boxes:
                        px = (box[0]+box[2])/2
                        py = (box[1]+box[3])/2
                        for (wx,wy) in hands:
                            if np.hypot(px-wx, py-wy) < thr:
                                return True
                    return False
                has_mobile = phone_near(kp, phone_boxes, W)
                final_status = "TALKING ON PHONE" if has_mobile else smooth_action

                # Display
                if ts["person_id"]:
                    conf_pct = int(ts["conf"]*100)
                    label = f"ID:{ts['person_id']} [{conf_pct}%]"
                    color = (0,255,0) if ts["conf"]>0.7 else (0,165,255)
                else:
                    label = f"Obs ({ts['seen_frames']}/{MIN_FRAMES_BEFORE_ID})"
                    color = (128,128,128)
                cv2.putText(display, f"{label} | {final_status}",
                            (max(0,rx-90), max(25,ry-30)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 2)
                if now - ts.get("new_flash",0) < 3.0:
                    cv2.putText(display, "NEW", (max(0,rx-40), max(50,ry-60)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 2)
                cv2.circle(display, (rx, ry), 6, color, -1)

        # Stats
        known = sum(1 for ts in track_state.values() if ts["person_id"])
        unknown = len(track_state)-known
        cv2.putText(display, f"Cam{camera_id} | Known:{known} Unknown:{unknown}",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.imshow(f"Surveillance Cam{camera_id}", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_id", type=int, required=True)
    ap.add_argument("--rtsp", type=str, required=True)
    args = ap.parse_args()
    run_surveillance(args.rtsp, args.camera_id)