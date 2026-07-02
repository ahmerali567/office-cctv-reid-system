"""
attendance_report.py — Office Attendance & Time Tracking Report
Persons DB + Events Log se readable report banao
"""
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

DB_FILE     = "persons_db.json"
EVENTS_FILE = "office_events.log"
ALERT_FILE  = "alert_log.json"

# =====================================================================
def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:   return f"{seconds}s"
    elif seconds < 3600: return f"{seconds//60}m {seconds%60}s"
    else: return f"{seconds//3600}h {(seconds%3600)//60}m"

def time_to_seconds(t_str):
    """HH:MM:SS -> seconds"""
    try:
        parts = t_str.split(":")
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    except:
        return 0

def parse_events_log():
    """office_events.log se per-person activity parse karo"""
    person_activities = defaultdict(list)  # pid -> list of {label, start, end, duration}
    
    if not os.path.exists(EVENTS_FILE):
        return person_activities
    
    with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # Format: [2026-05-25 14:41:50] NORMAL | ID:001 | 'working_computer' | 14:41:50 -> 14:41:50 | Total:0s
                parts = line.split("|")
                if len(parts) < 5: continue
                
                tag     = parts[0].split("]")[1].strip()
                pid     = parts[1].strip().replace("ID:", "")
                label   = parts[2].strip().strip("'")
                times   = parts[3].strip()
                total   = parts[4].strip().replace("Total:", "")
                
                time_parts = times.split("->")
                start = time_parts[0].strip()
                end   = time_parts[1].strip() if len(time_parts) > 1 else start
                
                person_activities[pid].append({
                    "label":    label,
                    "start":    start,
                    "end":      end,
                    "duration": total,
                    "type":     tag
                })
            except:
                continue
    
    return person_activities

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def load_alerts():
    if not os.path.exists(ALERT_FILE):
        return []
    try:
        with open(ALERT_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# =====================================================================
def print_report():
    db         = load_db()
    activities = parse_events_log()
    alerts     = load_alerts()
    
    print("\n" + "="*70)
    print("         OFFICE CCTV — ATTENDANCE & TIME REPORT")
    print("         Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # ── Per Person Summary ────────────────────────────────────────────
    print(f"\n{'ID':<6} {'First Seen':<12} {'Last Seen':<12} {'Total Time':<14} {'Status'}")
    print("-"*60)
    
    for pid, info in sorted(db.items()):
        first  = info.get("first_seen", "?")
        last   = info.get("last_seen",  "?")
        
        # Calculate total time in office
        try:
            fs = time_to_seconds(first)
            ls = time_to_seconds(last)
            total_sec = ls - fs if ls >= fs else 0
            total_str = format_duration(total_sec)
        except:
            total_str = "?"
        
        face = "[F]" if info.get("has_face") else "[K]"
        print(f"{pid:<6} {first:<12} {last:<12} {total_str:<14} {face}")
    
    print(f"\nTotal Persons Detected: {len(db)}")
    
    # ── Activity Breakdown ────────────────────────────────────────────
    if activities:
        print("\n" + "="*70)
        print("              ACTIVITY BREAKDOWN PER PERSON")
        print("="*70)
        
        for pid, acts in sorted(activities.items()):
            print(f"\n  Person: {pid}")
            print(f"  {'Activity':<25} {'Start':<10} {'End':<10} {'Duration':<12} {'Type'}")
            print(f"  {'-'*65}")
            for a in acts:
                print(f"  {a['label']:<25} {a['start']:<10} {a['end']:<10} {a['duration']:<12} {a['type']}")
    
    # ── Alerts Summary ────────────────────────────────────────────────
    if alerts:
        print("\n" + "="*70)
        print("                    ALERTS SUMMARY")
        print("="*70)
        
        for alert in alerts:
            print(f"\n  [{alert.get('severity','?')}] {alert.get('timestamp','?')}")
            print(f"  Person:{alert.get('person_id','?')} | {alert.get('message','?')} | Duration:{alert.get('duration','?')}")
    else:
        print("\n  No alerts recorded.")
    
    # ── CSV Data Summary ──────────────────────────────────────────────
    csv_file = "office_data.csv"
    if os.path.exists(csv_file):
        print("\n" + "="*70)
        print("                  TRAINING DATASET STATUS")
        print("="*70)
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"\n  Total samples: {len(df)}")
            print(f"\n  {'Label':<30} {'Count':<10}")
            print(f"  {'-'*40}")
            for label, count in df['label'].value_counts().items():
                bar = "█" * min(count, 30)
                print(f"  {label:<30} {count:<10} {bar}")
        except Exception as e:
            print(f"  Error reading CSV: {e}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print_report()