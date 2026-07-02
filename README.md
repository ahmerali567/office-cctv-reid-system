# Office CCTV Re-Identification & Activity Monitoring System

A real-time computer vision pipeline for office surveillance that performs persistent person re-identification, activity/behavior classification, and rule-based alerting from CCTV/RTSP camera feeds.

> **Note:** This is a demo/showcase build. No real footage, snapshots, or identity database from any live deployment is included in this repository — see [Privacy & Data](#privacy--data) below.

---

## Overview

The system ingests a live RTSP camera stream and, per frame:

1. Detects and tracks people using pose estimation + ByteTrack
2. Extracts an appearance embedding for each detected person (OSNet, MSMT17 weights)
3. Matches that embedding against previously seen people using cosine similarity + temporal voting, to assign a **persistent identity** across time (not just within a single track)
4. Classifies the person's current activity from pose features, using a locally trained classifier
5. Flags configurable "alert" behaviors (e.g. idle too long, phone use, sleeping at desk) against a rule set
6. Logs identity, activity, and alert events for later reporting

## Architecture

```
RTSP Stream
     │
     ▼
YOLO Pose Model  ──────►  ByteTrack (per-frame tracking)
     │                          │
     ▼                          ▼
Pose Feature Extraction   OSNet Embedding Extraction
     │                          │
     ▼                          ▼
Activity Classifier      Re-ID Matching (cosine sim +
(trained locally)         temporal voting, SQLite store)
     │                          │
     └────────────┬─────────────┘
                   ▼
         Alert Rules / Logging / Reporting
```

## Features

- **Persistent Re-Identification** — assigns a stable numeric ID to each person that survives across camera drop-outs and re-appearances, using quality-scored embedding storage (up to 12 embeddings per identity) and a similarity-gated re-acquisition strategy to avoid false matches after a person leaves frame.
- **Temporal Voting** — requires multiple consistent matches within a time window before confirming an identity, reducing false positives from single-frame noise.
- **Pose-Based Activity Recognition** — a custom classifier trained on 21 hand-engineered pose features (joint angles, limb ratios, posture indicators) to label activities like `working_computer`, `standing_talking`, `idle_no_work`, etc.
- **Rule-Based Alerting** — configurable severity levels (LOW/MEDIUM/HIGH/CRITICAL) for behaviors such as prolonged idling, personal phone use, sleeping at a desk, or unauthorized zone access.
- **Zone Awareness (ROI)** — supports defining regions of interest (e.g. restricted areas) for zone-specific rules.
- **Object-Assisted Detection** — combines YOLO object detection (e.g. phone, laptop) with pose data to improve activity classification accuracy.

## Tech Stack

| Component | Technology |
|---|---|
| Pose Estimation | YOLO11 (pose) |
| Object Detection | YOLOv8n |
| Multi-Object Tracking | ByteTrack |
| Re-ID Embeddings | OSNet (osnet_x1_0, MSMT17 pretrained) |
| Activity Classifier | Custom-trained model (scikit-learn) on pose features |
| Storage | SQLite |
| Language | Python 3 |

## Project Structure

```
office_cctv/
├── main.py                  # entry point — runs the full pipeline
├── run.py                   # runner script
├── reid_model.py             # OSNet embedding extraction + matching logic
├── detect_count.py           # object/person detection utilities
├── database.py                # SQLite persistence layer
├── office_agent.py           # activity/alert decision logic
├── train_model.py             # trains the activity classifier from pose data
├── attendence_report.py      # generates reports from logged data
├── config.py                  # camera, zone, alert, and model configuration
├── models/                    # (weights not included — see Model Weights below)
└── .gitignore
```

## Setup

```bash
git clone https://github.com/ahmerali567/office-cctv-reid-system.git
cd office-cctv-reid-system
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You will also need to obtain separately (not included in this repo):
- YOLO pose/object weights (`models/yolo11m-pose.pt`, `models/yolov8n.pt`) — publicly available from Ultralytics
- OSNet MSMT17 weights (`osnet_x1_0_msmt17.pth`) — publicly available pretrained ReID weights
- The trained activity classifier (`office_action_model.pkl`) — **not publicly distributed**; reach out directly if you'd like to discuss access for evaluation purposes

## Model Weights

The pose-based activity classifier included in this pipeline was trained specifically for this project and is kept private. This repository documents and open-sources the **pipeline and methodology** (tracking, re-identification, feature engineering, alerting logic) rather than the trained artifact itself.

Update `config.py` with your camera settings (RTSP URL, zones, alert thresholds) before running.

```bash
python3 main.py --camera_id 1 --rtsp <your_rtsp_url>
```

## Privacy & Data

This is a portfolio/demo project. To avoid any privacy or confidentiality concerns:

- No real camera footage, snapshots, or person database (`persons.db`) from any live environment is included in this repository.
- The activity classifier was trained using self-recorded footage only — no third-party individuals — and the trained weights are not distributed in this repository.
- Camera credentials, RTSP URLs, and any deployment-specific configuration are excluded via `.gitignore` and must be supplied locally.
- This system is intended as a technical demonstration of the re-identification and activity-recognition pipeline, not as a deployed monitoring tool. Any real-world deployment of person-tracking/identification systems should have explicit organizational approval and disclosure to those being monitored, in line with applicable privacy laws.

## Known Limitations

- Re-ID matching uses a linear scan against all stored identities — this does not scale well beyond a few hundred people; a vector index (e.g. FAISS) would be needed for larger deployments.
- Activity classifier is trained on a limited, self-collected dataset and may not generalize to all environments or body types.
- No authentication/encryption is applied to the local SQLite store — not intended for production use as-is.

## License

This project is shared for educational and portfolio purposes.
