# api/server.py
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
from schema.events_schema import StoreAnalyticsResponse, CustomerLog

app = FastAPI(
    title="Purplle Store Intelligence API Engine", 
    description="Production Event-Driven Computer Vision Analytics REST Server",
    version="1.0.0"
)

# Relative Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FPS = 30  # Standard CCTV frame rate multiplier

# 🕒 GLOBAL MEMORY REGISTER (For tracking instant in-store metrics live)
LIVE_STORE_STATE = {
    "current_occupancy": 0,
    "active_customer_ids": []
}

@app.get("/api/v1/analytics", response_model=StoreAnalyticsResponse)
def get_store_analytics():
    """
    Exposes high-level store traffic metrics, calculates real-time customer duration, 
    and returns an array of customer behaviors mapped to the strict Pydantic schema.
    """
    events_file = os.path.join(OUTPUT_DIR, "events.json")
    if not os.path.exists(events_file):
        raise HTTPException(
            status_code=404, 
            detail="Analytics log data missing. Please run the tracking pipeline first."
        )
        
    with open(events_file, "r") as f:
        raw_data = json.load(f)
        
    people_list = []
    total_frames_dwell = 0
    total_cust = raw_data.get("total_customers", 0)
    
    for idx, p in enumerate(raw_data.get("people", [])):
        frames = p.get("dwell_time_frames", 0)
        seconds = round(frames / FPS, 2)
        total_frames_dwell += frames
        
        # Retail Anomaly Rule: Flag customers staying past 60 seconds (or 5-10m in a real store)
        status = "Completed" if seconds <= 60 else "Loitering Anomaly"
            
        people_list.append(CustomerLog(
            customer_id=p.get("id", idx),
            entry_timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            exit_timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            dwell_time_seconds=seconds,
            status=status
        ))
        
    avg_dwell_sec = round((total_frames_dwell / total_cust) / FPS, 2) if total_cust > 0 else 0.0

    return StoreAnalyticsResponse(
        store_id="PURPLLE-MALL-HYD-01",
        timestamp=datetime.now(),
        total_unique_customers=total_cust,
        average_dwell_time_seconds=avg_dwell_sec,
        active_occupancy=LIVE_STORE_STATE["current_occupancy"],
        people_metrics=people_list
    )

@app.get("/api/v1/live_occupancy")
def get_live_occupancy():
    """
    Operational endpoint to check the count of active bodies currently detected 
    inside the physical premises right now.
    """
    return {
        "store_id": "PURPLLE-MALL-HYD-01",
        "timestamp": datetime.now(),
        "live_metrics": {
            "current_occupancy_count": LIVE_STORE_STATE["current_occupancy"],
            "active_tracked_ids": LIVE_STORE_STATE["active_customer_ids"]
        }
    }

@app.get("/api/v1/heatmap")
def get_store_heatmap():
    """
    Serves the colorized aggregate traffic density heatmap asset directly to clients.
    """
    heatmap_path = os.path.join(OUTPUT_DIR, "heatmap.jpg")
    if not os.path.exists(heatmap_path):
        raise HTTPException(
            status_code=404, 
            detail="Heatmap image asset not found. Run the video analyzer to compile it."
        )
    return FileResponse(heatmap_path, media_type="image/jpeg")