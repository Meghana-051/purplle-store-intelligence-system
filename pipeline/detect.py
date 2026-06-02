# pipeline/detect.py
import cv2
import torch
from ultralytics import YOLO

def run_detection_stream(video_path, frame_skip=5):
    """
    High-speed frame-skipping tracking stream engineered for multi-person crowds.
    Processes every Nth frame using hardware-accelerated AI with lower confidence
    thresholds to capture partially occluded people, while caching track positions
    to keep the interface running at zero extra compute cost on skipped frames.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    model = YOLO("yolov8n.pt").to(device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video path: {video_path}")
        return

    frame_idx = 0
    active_tracks_cache = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_detections = []
        annotated_frame = frame.copy()

        # RUN DEEP LEARNING TRACKER ON EVERY N-TH FRAME
        if frame_idx % frame_skip == 0:
            results = model.track(
                frame, 
                persist=True, 
                classes=[0], 
                verbose=False, 
                tracker="bytetrack.yaml",
                device=device,
                imgsz=640,
                conf=0.25
            )
            
            active_tracks_cache.clear()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, conf, track_id in zip(boxes, confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(track_id)
                    
                    det_entry = {
                        "frame": frame_idx,
                        "track_id": tid,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf)
                    }
                    frame_detections.append(det_entry)
                    active_tracks_cache[tid] = ([x1, y1, x2, y2], float(conf))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Cust ID: {tid}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, annotated_frame_rgb, frame_detections

        # FAST RENDERING LOGIC FOR INTERVENING SKIPPED FRAMES
        else:
            for tid, (bbox, conf) in active_tracks_cache.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Cust ID: {tid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                frame_detections.append({
                    "frame": frame_idx,
                    "track_id": tid,
                    "bbox": bbox,
                    "confidence": conf
                })
                
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, annotated_frame_rgb, frame_detections

        frame_idx += 1

    cap.release()