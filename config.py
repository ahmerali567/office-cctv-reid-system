"""
config.py – Office CCTV System Configuration (Production v3)
"""
import os

# =====================================================================
# CAMERA SETTINGS
# =====================================================================
ACTIVE_CAMERAS = [1]
FRAME_WIDTH    = 854
FRAME_HEIGHT   = 480
SNAPSHOT_GAP   = 8
FORCED_SNAP    = 30   # seconds

# =====================================================================
# MODEL PATHS
# =====================================================================
POSE_MODEL_PATH  = "models/yolo11m-pose.pt"
OBJ_MODEL_PATH   = "models/yolov8n.pt"
BRAIN_MODEL_PATH = "models/office_action_model.pkl"

# =====================================================================
# LLM SETTINGS
# =====================================================================
VISION_MODEL   = "llava"
LABEL_MODEL    = "llama3"
LLM_MAX_TOKENS = 80

# =====================================================================
# SAFE_LABELS – only these can be auto-learned (by pose model, not by agent)
# =====================================================================
SAFE_LABELS = {
    "working_computer", "walking_normal", "standing_idle",
    "sitting_idle", "typing_keyboard", "standing_talking",
    "reading_document", "drinking_coffee",
}

# =====================================================================
# ROI ZONES (optional)
# =====================================================================
ROI_ZONES = {
    "desk_area":   (50,  100, 800, 400),
    "exit_door":   (750, 50,  854, 480),
    "pantry":      (0,   50,  200, 480),
    "server_room": (600, 0,   854, 200),
    "fire_exit":   (700, 0,   854, 150),
}

# =====================================================================
# ACTIVITY CLASSIFICATION
# =====================================================================
NORMAL_ACTIVITIES = {
    "working_computer", "typing_keyboard", "using_mouse", "reading_document",
    "writing_paper", "sitting_normal", "sitting_idle", "standing_up",
    "standing_talking", "standing_idle", "walking_normal", "carrying_laptop",
    "group_discussion", "meeting_sitting", "handshake", "drinking_coffee",
    "drinking_water", "eating_lunch", "stretching_normal", "cleaning_desk",
    "using_headphones", "using_dustbin", "adjusting_monitor", "wearing_lanyard",
    "opening_drawer", "sitting_posture_good", "mobile_quick_check",
    "talking_on_phone", "looking_around", "waiting",
}

ALERT_ACTIVITIES = {
    "idle_no_work":              {"severity": "LOW",      "msg": "No activity for long time"},
    "mobile_personal_use":       {"severity": "MEDIUM",   "msg": "Personal mobile use detected"},
    "mobile_capturing":          {"severity": "HIGH",     "msg": "Phone camera pointed at monitor!"},
    "sleeping_desk":             {"severity": "HIGH",     "msg": "Sleeping on desk"},
    "sleeping_floor":            {"severity": "CRITICAL", "msg": "Person lying on floor!"},
    "sitting_on_table":          {"severity": "MEDIUM",   "msg": "Sitting on desk/table"},
    "feet_on_desk":              {"severity": "MEDIUM",   "msg": "Feet on desk"},
    "hiding_crouching":          {"severity": "HIGH",     "msg": "Hiding under desk"},
    "large_group_rush":          {"severity": "MEDIUM",   "msg": "Large group gathering"},
    "horseplay_dancing":         {"severity": "MEDIUM",   "msg": "Unusual movement"},
    "unusual_physical_activity": {"severity": "MEDIUM",   "msg": "Weird activity"},
    "unauthorized_visitor":      {"severity": "CRITICAL", "msg": "Unknown face"},
    "weapon_detected":           {"severity": "CRITICAL", "msg": "Weapon detected"},
    "theft_attempt":             {"severity": "CRITICAL", "msg": "Possible theft"},
    "exit_equipment":            {"severity": "CRITICAL", "msg": "Equipment taken out"},
    "blocking_fire_exit":        {"severity": "HIGH",     "msg": "Fire exit blocked"},
    "cctv_tampering":            {"severity": "CRITICAL", "msg": "Camera tampering"},
    "electrical_panel":          {"severity": "CRITICAL", "msg": "Unauthorized panel access"},
    "smoking_vaping":            {"severity": "HIGH",     "msg": "Smoking/vaping"},
    "physical_fight":            {"severity": "CRITICAL", "msg": "Violence"},
    "throwing_objects":          {"severity": "HIGH",     "msg": "Throwing objects"},
    "vandalism":                 {"severity": "CRITICAL", "msg": "Vandalism"},
    "littering":                 {"severity": "LOW",      "msg": "Littering"},
    "spilling_liquid":           {"severity": "MEDIUM",   "msg": "Liquid spill"},
    "pet_in_office":             {"severity": "MEDIUM",   "msg": "Pet detected"},
    "eating_server_room":        {"severity": "HIGH",     "msg": "Eating in restricted area"},
    "obscene_gesture":           {"severity": "HIGH",     "msg": "Inappropriate gesture"},
}

SEVERITY_COLORS = {
    "LOW":      (0, 255, 255),
    "MEDIUM":   (0, 165, 255),
    "HIGH":     (0, 0, 255),
    "CRITICAL": (0, 0, 180),
}

# =====================================================================
# YOLO OBJECT MAPPING
# =====================================================================
ALERT_OBJECTS = {
    "cell phone":  "mobile_personal_use",
    "knife":       "weapon_detected",
    "sports ball": "throwing_objects",
    "dog":         "pet_in_office",
    "cat":         "pet_in_office",
    "cigarette":   "smoking_vaping",
    "cup":         "drinking_coffee",
    "laptop":      "carrying_laptop",
    "book":        "reading_document",
}
EXIT_ALERT_OBJECTS = {"laptop", "tv", "keyboard", "mouse", "monitor"}

# =====================================================================
# POSE & DURATION CONFIG
# =====================================================================
POSE_CONFIG = {
    "sleeping_desk_frames":  90,
    "idle_timeout_seconds":  120,
    "group_rush_count":      4,
    "group_allowed_count":   3,
    "height_drop_threshold": 0.4,
    "fight_speed_threshold": 15.0,
}

DURATION_CONFIG = {
    "min_duration_log":   10,
    "alert_idle_after":   120,
    "alert_mobile_after": 30,
    "alert_sleep_after":  60,
}

# =====================================================================
# LLM PROMPTS
# =====================================================================
CROP_VISION_PROMPT = """You are analyzing a CROPPED image of a SINGLE person from office CCTV.
This image shows ONLY ONE person. Describe what THIS person is doing in MAXIMUM 5 WORDS.

Examples:
- Typing on a keyboard
- Looking at phone screen
- Head down on desk
- Standing with arms raised
- Sitting and reading paper

CRITICAL:
1. One person only – ignore background.
2. No paragraphs, no "I see", no "The person".
3. OUTPUT ONLY THE 5-WORD ACTION.
4. If blurry or unclear, write: sitting at desk"""

LABEL_SYSTEM_PROMPT = """You are an office activity classifier. Convert the text into EXACTLY ONE snake_case label.

NORMAL: working_computer, typing_keyboard, reading_document, writing_paper, sitting_idle, standing_up, standing_talking, standing_idle, walking_normal, carrying_laptop, group_discussion, meeting_sitting, drinking_coffee, stretching_normal, cleaning_desk, mobile_quick_check, talking_on_phone, looking_around

ALERT: idle_no_work, mobile_personal_use, sleeping_desk, sleeping_floor, sitting_on_table, feet_on_desk, horseplay_dancing, unusual_physical_activity, throwing_objects

RULES:
1. walking/moving -> 'walking_normal'
2. standing but no talking -> 'standing_idle' or 'looking_around'
3. talking/discussing -> 'standing_talking' or 'group_discussion'
4. typing/sitting/working -> 'working_computer'
5. Output ONLY the snake_case label."""

VERIFY_SYSTEM_PROMPT = """You are a strict CCTV verifier. Output ONLY YES or NO.

RULES:
1. pose walking & vision sitting -> NO
2. pose working_computer & vision standing -> NO
3. pose standing_idle & vision walking -> NO
4. pose 'analyzing'/'unknown' -> YES (trust vision)
5. Trust POSE more for movement.
6. Answer YES only if actions are compatible.
7. When in doubt -> YES (vision teaches the pose model)"""

# =====================================================================
# LOG FILES
# =====================================================================
ALERT_LOG_FILE   = "alert_log.json"
OFFICE_DATA_FILE = "office_data.csv"
EVENTS_LOG_FILE  = "office_events.log"