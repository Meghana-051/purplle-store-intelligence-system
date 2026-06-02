# pipeline/heatmap.py
import cv2
import numpy as np

def generate_heatmap(final_people, video_path, output_heatmap_path):
    """
    Accumulates tracking coordinates to output an aggregate store occupancy heatmap.
    Ensures safe data types to prevent cv2.applyColorMap from crashing.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame for heatmap baseline.")
        if cap.isOpened():
            cap.release()
        return
    
    height, width, _ = frame.shape
    cap.release()

    accum_mask = np.zeros((height, width), dtype=np.float32)
    has_points = False

    for person in final_people:
        for point in person.get("history", []):
            x1, y1, x2, y2 = point["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int(y2) 
            
            if 0 <= cx < width and 0 <= cy < height:
                cv2.circle(accum_mask, (cx, cy), 25, 1.0, -1)
                has_points = True

    if has_points:
        accum_mask = cv2.GaussianBlur(accum_mask, (51, 51), 0)
        max_val = np.max(accum_mask)
        if max_val > 0:
            accum_mask = (accum_mask / max_val * 255)

    accum_mask_8u = accum_mask.astype(np.uint8)
    color_heatmap = cv2.applyColorMap(accum_mask_8u, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, color_heatmap, 0.4, 0)
    
    cv2.imwrite(output_heatmap_path, overlay)