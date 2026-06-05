import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
import pandas as pd

from src.tracker import PoultryTracker
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.visualization import Visualizer
from src.reporting import ReportGenerator

st.set_page_config(page_title="AI Poultry Monitoring", layout="wide")

st.title("🐔 Poultry Monitoring Dashboard")
st.markdown("Real-time tracking, behavior analysis, and alerting using YOLOv8 & ByteTrack.")

# --- Sidebar ---
st.sidebar.header("Video Input")
input_source = st.sidebar.radio("Select Source", ["Upload Video", "Webcam", "RTSP Stream"])

video_path = None
if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4", type=['mp4', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif input_source == "Webcam":
    video_path = 0 # default webcam
elif input_source == "RTSP Stream":
    video_path = st.sidebar.text_input("RTSP URL", "rtsp://localhost:8554/stream")

st.sidebar.header("Detection Settings")
model_size = st.sidebar.selectbox("Model Size", [
    "Nano (yolov8n.pt - Fast)", 
    "Small (yolov8s.pt - Better)", 
    "Medium (yolov8m.pt - Best/Slow)",
    "World Small (yolov8s-world.pt - Zero-Shot)",
    "World Medium (yolov8m-worldv2.pt - Zero-Shot)"
])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.10, 0.05)

target_classes = None

tracker_algo_ui = st.sidebar.selectbox("Tracking Algorithm", ["ByteTrack (Faster)", "BoTSORT (More Accurate)"])
selected_tracker = "botsort" if "BoTSORT" in tracker_algo_ui else "bytetrack"

model_path_map = {
    "Nano (yolov8n.pt - Fast)": "yolov8n.pt",
    "Small (yolov8s.pt - Better)": "yolov8s.pt",
    "Medium (yolov8m.pt - Best/Slow)": "yolov8m.pt",
    "World Small (yolov8s-world.pt - Zero-Shot)": "yolov8s-world.pt",
    "World Medium (yolov8m-worldv2.pt - Zero-Shot)": "yolov8m-worldv2.pt"
}
selected_model_path = model_path_map[model_size]

start_button = st.sidebar.button("Start Processing")

st.sidebar.header("Identity Tag Manager")
st.sidebar.markdown("Map IDs to names (e.g. `5:Worker John`)")
tag_input = st.sidebar.text_area("Custom Tags", "")
custom_tags = {}
for line in tag_input.split('\n'):
    if ':' in line:
        k, v = line.split(':', 1)
        custom_tags[k.strip()] = v.strip()

# Stop processing using session state since button click reloads
if 'processing' not in st.session_state:
    st.session_state.processing = False

if start_button:
    st.session_state.processing = True

stop_button = st.sidebar.button("Stop")
if stop_button:
    st.session_state.processing = False

# --- Initialize Modules ---
@st.cache_resource
def load_tracker(model_path, tracker_algo):
    return PoultryTracker(model_path, tracker_algo)

if 'analytics' not in st.session_state:
    st.session_state.analytics = PoultryAnalytics()
    
# Layout for Video and Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()
    
    st.subheader("Heatmap")
    heatmap_placeholder = st.empty()

with col2:
    st.subheader("Analytics Dashboard")
    # Metric placeholders
    m_col1, m_col2, m_col_human = st.columns(3)
    m_total = m_col1.empty()
    m_active = m_col2.empty()
    m_humans = m_col_human.empty()
    
    m_col3, m_col4 = st.columns(2)
    m_score = m_col3.empty()
    m_zone = m_col4.empty()
    
    m_col5, m_col6 = st.columns(2)
    m_alerts = m_col5.empty()
    m_uniformity = m_col6.empty()
    
    st.subheader("Live Analytics Charts")
    zone_chart_placeholder = st.empty()
    activity_chart_placeholder = st.empty()
    
    st.subheader("Recent Alerts")
    alert_box = st.empty()
    
    report_placeholder = st.empty()

if st.session_state.processing and video_path is not None:
    tracker = load_tracker(selected_model_path, selected_tracker)
    visualizer = Visualizer()
    analytics = st.session_state.analytics
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video source: {video_path}")
        st.session_state.processing = False
    else:
        # Get frame size for zones
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            zone_manager = ZoneManager(w, h)
        else:
            st.error("Failed to read video stream")
            st.session_state.processing = False
            
        frame_skip = 1 # Process every frame for smooth tracking
        frame_count = 0
        
        while cap.isOpened() and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Process Frame
            detections = tracker.process_frame(frame, conf_threshold=conf_threshold, classes=target_classes)
            
            # Update dynamic zones based on detected pots
            zone_manager.update_pots(detections)
            
            # Update analytics
            for det in detections:
                track_id = det["track_id"]
                cx, cy = det["center"]
                x1, y1, x2, y2 = det["box"]
                box_area = (x2 - x1) * (y2 - y1)
                class_id = det.get("class_id", 14)
                class_name = det.get("class_name", "bird")
                zone = zone_manager.get_zone(cx, cy)
                analytics.update(track_id, cx, cy, zone, class_id=class_id, class_name=class_name, box_area=box_area)
                
            # Run flock-level analytics (huddling)
            analytics.analyze_flock()
            
            # Visualization
            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics, custom_tags=custom_tags)
            
            # Show heatmap optionally
            heatmap = visualizer.get_heatmap_overlay(frame_disp)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            heatmap_placeholder.image(heatmap_rgb, channels="RGB", use_container_width=True)
            
            # Update Dashboard
            stats = analytics.get_summary_stats()
            
            # Use columns inside the placeholders
            m_total.metric("Total Chickens", stats["total_chickens"])
            m_humans.metric("Humans Detected", stats["total_humans"])
            m_active.metric("Active / Inactive", f"{stats['active']} / {stats['inactive']}")
            m_score.metric("Avg Activity Score", f"{stats['avg_activity_score']}%")
            m_zone.metric("Most Visited Zone", stats["most_visited_zone"])
            m_alerts.metric("Alerts", stats["alert_count"])
            m_uniformity.metric("Size Uniformity", stats["size_uniformity"])
            
            # Update Charts
            if stats["timeline"]["timestamps"]:
                # Zone utilization chart
                zone_df = pd.DataFrame({
                    "Feed Zone": stats["timeline"]["feed_zone"],
                    "Water Zone": stats["timeline"]["water_zone"],
                    "Rest Zone": stats["timeline"]["rest_zone"]
                }, index=stats["timeline"]["timestamps"])
                zone_chart_placeholder.line_chart(zone_df)
                
                # Activity trend chart
                act_df = pd.DataFrame({
                    "Avg Activity": stats["timeline"]["avg_activity"]
                }, index=stats["timeline"]["timestamps"])
                activity_chart_placeholder.line_chart(act_df)
            
            # Alerts
            if analytics.alerts:
                formatted_alerts = []
                for a in analytics.alerts[-10:]:
                    if "SEVERE" in a:
                        formatted_alerts.append(f"<span style='color:#ff4b4b; font-weight:bold;'>{a}</span>")
                    elif "erratic" in a or "Panic" in a:
                        formatted_alerts.append(f"<span style='color:#ffa500; font-weight:bold;'>{a}</span>")
                    elif "HUDDLING" in a:
                        formatted_alerts.append(f"<span style='color:#1e90ff; font-weight:bold;'>{a}</span>")
                    else:
                        formatted_alerts.append(a)
                alert_text = "<br>".join(formatted_alerts)
                alert_box.markdown(f"<div style='height: 150px; overflow-y: scroll; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #1e1e1e;'>{alert_text}</div>", unsafe_allow_html=True)
                
        cap.release()
        
    # Generate Reports at the end
    if not st.session_state.processing:
        with report_placeholder.container():
            st.subheader("Export Reports")
            rg = ReportGenerator(analytics)
            csv_file = rg.export_csv()
            pdf_file = rg.export_pdf()
            
            with open(csv_file, "rb") as f:
                st.download_button("Download CSV", f, file_name=os.path.basename(csv_file))
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(pdf_file))
