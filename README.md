# Purplle Store Intelligence System

End-to-end video analytics and store intelligence platform built with YOLOv8, FastAPI, Streamlit, and tracking optimization.

## Project structure

- `api/server.py` - REST API server
- `pipeline/` - detection, tracking, and streaming scripts
- `schema/events_schema.py` - event schema definitions
- `app.py` - application entrypoint
- `requirements.txt` - Python dependencies
- `data/videos/` - source video assets (ignored)
- `output/` - generated outputs (ignored)
- `yolov8n.pt` - model weights (ignored)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```
