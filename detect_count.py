import cv2
import time
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    # Changed default to "0" for local webcam testing
    parser.add_argument("--source", type=str, default="0", 
                        help="Camera source (0 for webcam, or RTSP URL)")
    parser.add_argument("--confidence", type=float, default=0.4, 
                        help="Detection confidence threshold")
    parser.add_argument("--display", action="store_true", 
                        help="Show live video window")
    args = parser.parse_args()

    print("[INFO] Loading YOLO model...")
    # Ensure this points to your specific model path
    model = YOLO('models/yolov8n.pt') 

    # Handle digit input for local webcams
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {source}")
        return
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[INFO] Starting camera... Press Ctrl+C to stop.")
    time.sleep(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 1. SINGLE INFERENCE: Do this once per frame
            results = model(frame, conf=args.confidence, verbose=False)
            result = results[0]

            # 2. DYNAMIC COUNTING: Use YOLO's built-in class names
            counts = {}
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id] 
                    counts[class_name] = counts.get(class_name, 0) + 1

            # 3. TERMINAL OUTPUT
            print("\033c", end="")
            print("=" * 50)
            print("    OFFICE CCTV DETECTION (TESTING)")
            print("=" * 50)
            print(f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
            # Print whatever is actively detected in the frame
            for obj_name, count in sorted(counts.items()):
                print(f"   {obj_name:15} : {count}")
                
            print("-" * 50)
            print(f"   👥 PEOPLE : {counts.get('person', 0)}")
            print("=" * 50)

            # 4. DISPLAY USING SAVED RESULTS
            if args.display:
                annotated = result.plot() # Reuse the result from above
                h, w = annotated.shape[:2]
                if w > 1024:
                    scale = 1024 / w
                    annotated = cv2.resize(annotated, (1024, int(h * scale)))
                
                cv2.imshow("Detection Testing", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""    
python detect_count.py --source "rtsp://admin:admin123456@192.168.100.153:8554/profile0" --confidence 0.5 --display    
"""