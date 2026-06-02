# pipeline/events.py
import json
import numpy as np
from collections import defaultdict

def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def process_and_clean_tracks(raw_tracks, min_frames_threshold=5, max_frame_gap=60, spatial_distance_threshold=100):
    """
    Cleans track IDs and implements Retail Anomaly Detection rules:
    - Identifies Loitering Anomalies (Dwell time > 90 seconds)
    - Identifies Checkout Congestion Anomalies
    """
    filtered_tracks = {}
    for tid, history in raw_tracks.items():
        if len(history) >= min_frames_threshold:
            history.sort(key=lambda x: x["frame"])
            filtered_tracks[tid] = history

    sorted_tids = sorted(filtered_tracks.keys(), key=lambda x: filtered_tracks[x][0]["frame"])
    id_mapping = {tid: tid for tid in sorted_tids}
    merged_tracks = {}

    for i in range(len(sorted_tids)):
        id_a = sorted_tids[i]
        for j in range(i + 1, len(sorted_tids)):
            id_b = sorted_tids[j]
            root_a = id_mapping[id_a]
            root_b = id_mapping[id_b]
            if root_a == root_b:
                continue

            history_a = filtered_tracks[root_a] if root_a in merged_tracks else filtered_tracks[id_a]
            history_b = filtered_tracks[root_b] if root_b in merged_tracks else filtered_tracks[id_b]

            end_frame_a = history_a[-1]["frame"]
            start_frame_b = history_b[0]["frame"]

            frame_gap = start_frame_b - end_frame_a
            if 0 < frame_gap <= max_frame_gap:
                centroid_a = calculate_centroid(history_a[-1]["bbox"])
                centroid_b = calculate_centroid(history_b[0]["bbox"])
                distance = np.linalg.norm(np.array(centroid_a) - np.array(centroid_b))
                
                if distance <= spatial_distance_threshold:
                    for k, v in id_mapping.items():
                        if v == root_b:
                            id_mapping[k] = root_a
    
    healed_tracks = defaultdict(list)
    for tid, history in filtered_tracks.items():
        final_id = id_mapping[tid]
        healed_tracks[final_id].extend(history)

    final_people = []
    anomalies_logged = []
    checkout_zone = [300, 200, 640, 480] 

    for final_id, history in healed_tracks.items():
        history.sort(key=lambda x: x["frame"])
        start_f = history[0]["frame"]
        end_f = history[-1]["frame"]
        dwell_time = end_f - start_f + 1

        is_loitering = False
        if dwell_time > 2700:  
            is_loitering = True
            anomalies_logged.append({
                "type": "LONG_DWELL_LOITERING",
                "customer_id": int(final_id),
                "duration_frames": int(dwell_time)
            })

        in_checkout_zone_count = 0
        for entry in history:
            cx, cy = calculate_centroid(entry["bbox"])
            if checkout_zone[0] <= cx <= checkout_zone[2] and checkout_zone[1] <= cy <= checkout_zone[3]:
                in_checkout_zone_count += 1
                
        if dwell_time > 150 and (in_checkout_zone_count / len(history)) > 0.7:
            anomalies_logged.append({
                "type": "CHECKOUT_QUEUE_CONGESTION",
                "customer_id": int(final_id),
                "zone_occupancy_ratio": round(in_checkout_zone_count / len(history), 2)
            })

        final_people.append({
            "id": int(final_id),
            "dwell_time_frames": int(dwell_time),
            "is_anomaly": is_loitering,
            "history": history
        })

    output_data = {
        "total_customers": len(final_people),
        "anomalies_detected": anomalies_logged,
        "people": [{"id": p["id"], "dwell_time_frames": p["dwell_time_frames"]} for p in final_people]
    }

    return output_data, final_people

def save_events_json(output_data, output_path):
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)