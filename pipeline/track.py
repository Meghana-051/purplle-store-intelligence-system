# pipeline/track.py
from collections import defaultdict

def group_by_tracks(raw_detections):
    """
    Groups frame-by-frame raw video streams into chronological target trajectory blocks.
    """
    tracks = defaultdict(list)
    for det in raw_detections:
        tracks[det["track_id"]].append({
            "frame": det["frame"],
            "bbox": det["bbox"],
            "confidence": det["confidence"]
        })
    return tracks