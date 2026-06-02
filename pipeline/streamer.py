# pipeline/streamer.py
import json
from datetime import datetime

def stream_tracking_event(customer_id, bbox, status="MOVING"):
    """
    Simulates real-time event streaming payload injection.
    In a live store production deployment, this payload would be pushed 
    directly into an event broker like Apache Kafka or Redis Streams.
    """
    event_payload = {
        "event_type": "STORE_TRAJECTORY_UPDATE",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "store_id": "PURPLLE-MALL-HYD-01",
        "data": {
            "customer_id": int(customer_id),
            "coordinates": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3])
            },
            "status": status
        }
    }
    
    # Print to console to simulate logging stream output live
    print(f"[STREAMING EVENT] {json.dumps(event_payload)}")
    return event_payload