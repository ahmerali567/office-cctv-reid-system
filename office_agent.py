"""
agent.py – AI Agent that reads person snapshots
- Watches snapshots/{person_id}/ folder for new images
- For each new crop, runs LLaVA + LLaMA3
- Logs alerts and events
"""

import os
import time
import json
import threading
import ollama
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================================================================
# CONFIGURATION
# ==================================================================
SNAPSHOT_BASE = "snapshots"
POLL_INTERVAL = 2.0   # seconds
PROCESSED_RECORD = "processed_snapshots.txt"  # simple log to avoid re-processing

VISION_MODEL = "llava"
LABEL_MODEL = "llama3"
LLM_MAX_TOKENS = 80

ALERT_LOG_FILE = "alert_log.json"
EVENTS_LOG_FILE = "office_events.log"

# Prompt for a single person crop
CROP_PROMPT = """You are analyzing a CROPPED image of a SINGLE person in an office.
Describe what THIS person is doing in MAXIMUM 5 WORDS.
Examples: 'Typing on keyboard', 'Looking at phone', 'Head down on desk', 'Talking on phone'.
Output ONLY the 5‑word description."""

LABEL_SYSTEM_PROMPT = """You are an office activity classifier. Convert the description into EXACTLY ONE snake_case label.

NORMAL: working_computer, walking_normal, standing_idle, sitting_idle, typing_keyboard, standing_talking, reading_document, drinking_coffee, talking_on_phone
ALERT: sleeping_desk, mobile_personal_use, unusual_physical_activity, throwing_objects, fighting

Output ONLY the label."""

# LLM pipelines
llm = OllamaLLM(model=LABEL_MODEL, temperature=0.0)
label_prompt = ChatPromptTemplate.from_messages([
    ("system", LABEL_SYSTEM_PROMPT),
    ("user", "Classify: '{visual_input}'")
])
standardizer = label_prompt | llm | StrOutputParser()

# Already processed set
processed = set()
if os.path.exists(PROCESSED_RECORD):
    with open(PROCESSED_RECORD, 'r') as f:
        processed = set(line.strip() for line in f)

def get_vision_description(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        r = ollama.generate(
            model=VISION_MODEL,
            prompt=CROP_PROMPT,
            images=[img_bytes],
            options={"temperature": 0.0, "num_predict": LLM_MAX_TOKENS}
        )
        return r.get("response", "").strip() or None
    except Exception as e:
        print(f"[Agent] Vision error: {e}")
        return None

def log_analysis(person_id, description, label):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Event log
    with open(EVENTS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Person {person_id} | LLaVA: {description} -> Label: {label}\n")
    # Alert if alert label
    if label in ["sleeping_desk", "mobile_personal_use", "unusual_physical_activity", "throwing_objects", "fighting"]:
        alert = {
            "timestamp": timestamp,
            "person_id": person_id,
            "label": label,
            "description": description
        }
        existing = []
        if os.path.exists(ALERT_LOG_FILE):
            try:
                with open(ALERT_LOG_FILE, "r") as f:
                    existing = json.load(f)
            except:
                pass
        existing.append(alert)
        with open(ALERT_LOG_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[ALERT] Person {person_id} – {label} | {description}")

def scan_and_process():
    while True:
        try:
            # Walk through all person folders
            if not os.path.exists(SNAPSHOT_BASE):
                time.sleep(POLL_INTERVAL)
                continue
            for person_id in os.listdir(SNAPSHOT_BASE):
                person_dir = os.path.join(SNAPSHOT_BASE, person_id)
                if not os.path.isdir(person_dir):
                    continue
                # Get all jpg files
                images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
                for img_file in images:
                    full_path = os.path.join(person_dir, img_file)
                    key = f"{person_id}/{img_file}"
                    if key in processed:
                        continue
                    # Process the crop
                    print(f"[Agent] Analyzing {key}")
                    description = get_vision_description(full_path)
                    if not description:
                        continue
                    label = standardizer.invoke({"visual_input": description})
                    label = label.strip().lower().replace(" ", "_")
                    print(f"  -> {description} -> {label}")
                    log_analysis(person_id, description, label)
                    # Mark as processed
                    processed.add(key)
                    with open(PROCESSED_RECORD, "a") as f:
                        f.write(key + "\n")
        except Exception as e:
            print(f"[Agent] Error: {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    print("[Agent] Started – watching person snapshots")
    scan_and_process()