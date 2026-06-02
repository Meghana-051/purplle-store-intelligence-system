# app.py
import os
import cv2
import json
import streamlit as st
from pipeline.detect import run_detection_stream
from pipeline.track import group_by_tracks
from pipeline.events import process_and_clean_tracks, save_events_json
from pipeline.heatmap import generate_heatmap
from pipeline.streamer import stream_tracking_event

# 1. Page Configuration & Professional Theme Styling
st.set_page_config(
    page_title="Purplle Store Intelligence View", 
    page_icon="🏬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for an Enterprise SaaS Look & Feel
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
            width: 100%;
            height: 3em;
            background-color: #ff4b4b;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ff2b2b;
            transform: translateY(-1px);
        }
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1c1c1e;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8e8e93;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Configuration Layout
st.sidebar.image("https://img.icons8.com/fluent/96/000000/shop.png", width=60)
st.sidebar.title("Purplle Edge Admin")
st.sidebar.markdown("---")

st.sidebar.subheader("🎛️ Pipeline Parameters")
min_frames = st.sidebar.slider("Min Frames (Denoise)", 2, 60, 5, help="Ignore tracks shorter than this frame count.")
max_gap = st.sidebar.slider("Max Tracking Gap", 10, 150, 60, help="Max frames allowed before a dropped target splits IDs.")
spatial_dist = st.sidebar.slider("Spatial Healing Range", 30, 200, 100, help="Pixel range to stitch a vanished ID to a new ID.")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Source Video Feed")
video_file = st.sidebar.file_uploader("Upload Store CCTV Video", type=["mp4", "avi", "mov", "mkv"])

# Cache Clearing Safety Trigger
if video_file is not None:
    st.cache_data.clear()
    st.cache_resource.clear()

VIDEO_DIR = "data/videos"
OUTPUT_DIR = "output"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

events_path = os.path.join(OUTPUT_DIR, "events.json")
heatmap_img_path = os.path.join(OUTPUT_DIR, "heatmap.jpg")

if video_file is not None:
    uploaded_video_path = os.path.join(VIDEO_DIR, video_file.name)
    if "current_video" not in st.session_state or st.session_state["current_video"] != video_file.name:
        st.session_state["current_video"] = video_file.name
        if os.path.exists(events_path): os.remove(events_path)
        if os.path.exists(heatmap_img_path): os.remove(heatmap_img_path)
        with open(uploaded_video_path, "wb") as f:
            f.write(video_file.read())
    video_path = uploaded_video_path
else:
    video_path = os.path.join(VIDEO_DIR, "sample.mp4")
    if os.path.exists(video_path) and "current_video" not in st.session_state:
        st.session_state["current_video"] = "sample.mp4"

# 3. Main Workspace Header Layout
st.title("🏬 Store Intelligence Operations Center")
st.markdown(f"**Location ID:** `PURPLLE-MALL-HYD-01` | **Active Stream:** `{st.session_state.get('current_video', 'sample.mp4')}`")
st.markdown("---")

# Tabbed Navigation Implementation
tab1, tab2, tab3 = st.tabs(["📹 Real-Time Video Processor", "📊 Store Performance Analytics", "🧠 System Specifications"])

# ================= TAB 1: VIDEO PROCESSOR =================
with tab1:
    col_ctrl, col_view = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("### Control Actions")
        st.write("Initiate the hardware-accelerated computer vision framework down below.")
        run_pipeline = st.button("🚀 Execute Store AI Engine", type="primary")
        
        st.markdown("---")
        st.markdown("### Runtime Telemetry")
        frame_metric_placeholder = st.empty()
        raw_count_placeholder = st.empty()
        
    with col_view:
        if run_pipeline and os.path.exists(video_path):
            video_frame_placeholder = st.empty()
            raw_detections_accumulator = []
            unique_seen_raw_ids = set()

            for frame_idx, rgb_frame, frame_data in run_detection_stream(video_path, frame_skip=5):
                raw_detections_accumulator.extend(frame_data)
                for det in frame_data:
                    unique_seen_raw_ids.add(det["track_id"])
                    stream_tracking_event(det["track_id"], det["bbox"])
                    
                # Dynamic Metric Container Updates (Fixed Parameters Here)
                frame_metric_placeholder.markdown(f"<div class='metric-card'><div class='metric-label'>🎞️ Current Frame</div><div class='metric-value'>{frame_idx}</div></div>", unsafe_allow_html=True)
                raw_count_placeholder.markdown(f"<div class='metric-card'><div class='metric-label'>⚠️ Raw Unfiltered IDs</div><div class='metric-value'>{len(unique_seen_raw_ids)}</div></div>", unsafe_allow_html=True)
                
                video_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            with st.spinner("Optimizing trajectories & compiling data metrics..."):
                raw_tracks = group_by_tracks(raw_detections_accumulator)
                output_json, final_people = process_and_clean_tracks(
                    raw_tracks, min_frames_threshold=min_frames, max_frame_gap=max_gap, spatial_distance_threshold=spatial_dist
                )
                save_events_json(output_json, events_path)
                generate_heatmap(final_people, video_path, heatmap_img_path)
            st.success("✅ Frame analysis completed successfully! Performance profiles loaded in Tab 2.")
        else:
            st.info("Click the 'Execute Store AI Engine' button to run inference calculations on the video feed.")

# ================= TAB 2: STORE ANALYTICS =================
with tab2:
    if os.path.exists(events_path):
        with open(events_path, "r") as f:
            analytics_data = json.load(f)

        total_customers = analytics_data.get("total_customers", 0)
        dwell_times = [p.get("dwell_time_frames", 0) for p in analytics_data.get("people", [])]
        avg_dwell = round(sum(dwell_times) / len(dwell_times), 1) if dwell_times else 0
        anomalies_count = len(analytics_data.get("anomalies_detected", [])) if "anomalies_detected" in analytics_data else 0

        # Professional Metric Card Row (Fixed Parameters Here)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>🎯 De-duplicated Footfall Count</div><div class='metric-value'>{total_customers}</div></div>", unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>⏱️ Average Customer Stay</div><div class='metric-value'>{avg_dwell} frames</div></div>", unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>🚨 Store Anomalies Flagged</div><div class='metric-value'>{anomalies_count}</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        col_img, col_json = st.columns([3, 2])
        
        with col_img:
            st.subheader("🔥 Foot-Traffic Density Map (Heatmap)")
            if os.path.exists(heatmap_img_path):
                st.image(heatmap_img_path, use_container_width=True)
                
        with col_json:
            st.subheader("📋 Compliant Structured System Logs")
            st.json(analytics_data)
    else:
        st.warning("⚠️ No operational analytical logs found on disk. Navigate to Tab 1 and execute the pipeline processing engine first.")

# ================= TAB 3: SYSTEM SPECIFICATIONS =================
with tab3:
    st.markdown("### 🧠 Embedded AI Engine Hardware Mapping")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.info("**Primary Model Variant:** Ultralytics YOLOv8n (Person Profile Only)")
        st.info("**Asynchronous Output Pipeline:** Integrated Mock Streaming Protocol Active")
    with col_s2:
        st.info("**Downstream Middleware Heuristics:** Spatial Temporal ID Correction Algorithm Enabled")
        st.info("**API Port Routing Status:** FastAPI Engine Host Endpoint Map active on Port 8000")