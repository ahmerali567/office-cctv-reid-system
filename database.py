"""
database.py – SQLite backend for persons, embeddings, active tracking
"""
import sqlite3
import time
import numpy as np

DB_PATH = "persons.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Main persons table (one row per identity)
    c.execute('''CREATE TABLE IF NOT EXISTS persons
                 (id TEXT PRIMARY KEY,
                  first_seen REAL,
                  last_seen REAL,
                  first_camera INTEGER,
                  last_camera INTEGER,
                  best_embedding BLOB)''')
    # Track every detection (for cross‑camera linking)
    c.execute('''CREATE TABLE IF NOT EXISTS identity_map
                 (person_id TEXT,
                  camera_id INTEGER,
                  track_id INTEGER,
                  embedding BLOB,
                  timestamp REAL)''')
    # Active persons cache (LRU-like)
    c.execute('''CREATE TABLE IF NOT EXISTS active_persons
                 (person_id TEXT PRIMARY KEY,
                  camera_id INTEGER,
                  track_id INTEGER,
                  last_seen REAL)''')
    conn.commit()
    conn.close()

init_db()

def save_person(person_id, embedding, camera_id, track_id):
    """Store or update a person with averaged embedding."""
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get existing embedding if any
    c.execute("SELECT best_embedding FROM persons WHERE id=?", (person_id,))
    row = c.fetchone()
    if row:
        old_emb = np.frombuffer(row[0], dtype=np.float32)
        new_emb = (old_emb + embedding) / 2.0
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        blob = new_emb.astype(np.float32).tobytes()
        c.execute("UPDATE persons SET last_seen=?, last_camera=?, best_embedding=? WHERE id=?",
                  (time.time(), camera_id, blob, person_id))
    else:
        blob = embedding.astype(np.float32).tobytes()
        c.execute("INSERT INTO persons VALUES (?,?,?,?,?,?)",
                  (person_id, time.time(), time.time(), camera_id, camera_id, blob))
    # Insert into identity_map
    c.execute("INSERT INTO identity_map VALUES (?,?,?,?,?)",
              (person_id, camera_id, track_id, blob, time.time()))
    # Update active_persons
    c.execute("REPLACE INTO active_persons VALUES (?,?,?,?)",
              (person_id, camera_id, track_id, time.time()))
    conn.commit()
    conn.close()

def find_best_match(embedding, min_sim=0.52):
    """
    Returns (person_id, similarity) of best match across all cameras.
    """
    if embedding is None:
        return None, 0.0
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, best_embedding FROM persons")
    best_id = None
    best_sim = 0.0
    for pid, blob in c.fetchall():
        emb = np.frombuffer(blob, dtype=np.float32)
        sim = np.dot(embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_id = pid
    conn.close()
    if best_sim >= min_sim:
        return best_id, best_sim
    return None, best_sim

def cleanup_stale_active(age_seconds=180):
    """Remove persons not seen for age_seconds from active_persons table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = time.time() - age_seconds
    c.execute("DELETE FROM active_persons WHERE last_seen < ?", (cutoff,))
    conn.commit()
    conn.close()

def get_next_person_id():
    """Return next available numeric ID as zero-padded string."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT MAX(CAST(id AS INTEGER)) FROM persons")
    row = c.fetchone()
    max_id = row[0] if row and row[0] else 0
    conn.close()
    return str(max_id + 1).zfill(3)