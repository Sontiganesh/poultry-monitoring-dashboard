import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
import torch
import warnings
import pandas as pd

# Restrict PyTorch to a single thread to prevent it from aggressively consuming 
# all vCPUs (100%+) during OpenMP matrix multiplications.
torch.set_num_threads(1)

from src.tracker import PoultryTracker
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.visualization import Visualizer
from src.reporting import ReportGenerator
from src.webhook import WebhookDispatcher
import uuid

# Create a unique session ID for this browser tab so multiple users don't overwrite each other's frames!
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id

def push_video_frame(jpeg_bytes: bytes):
    """Write the latest frame to the shared file atomically to prevent stream flickering."""
    frame_path = f"/tmp/poultry_frame_{SESSION_ID}.jpg"
    temp_path = frame_path + ".tmp"
    try:
        # Write to a temp file first, then atomically rename it.
        # This completely cures image tearing/flickering!
        with open(temp_path, "wb") as f:
            f.write(jpeg_bytes)
        os.rename(temp_path, frame_path)
    except Exception:
        pass

st.set_page_config(page_title="AI Poultry Monitoring", layout="wide")

# Parse Query Parameters
query_params = st.query_params
mode = query_params.get("mode")

# --- EMBED MODE CONFIGURATION ---
if mode in ["embed", "video_only"]:
    # Hide sidebar and header/footer for clean iframe embedding
    css = """<style>
[data-testid="stSidebar"] { display: none !important; }
header { display: none !important; }
footer { display: none !important; }
</style>"""
    
    if mode == "video_only":
        css += """<style>
.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
</style>"""
        
    st.markdown(css, unsafe_allow_html=True)
    
    # Auto-play demo video
    if "auto_played" not in st.session_state:
        st.session_state.processing = True
        st.session_state.auto_played = True

# --- SIDEBAR & UPLOAD CONTROLS ---
# We still run this code, but it's hidden by CSS if mode=embed
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("Configuration")
is_embedded = False
auto_video_path = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

demo_video = query_params.get("video")
demo_shed = query_params.get("shed")

# Determine video path
if mode in ["embed", "video_only"]:
    is_embedded = True
    auto_video_path = os.path.join(BASE_DIR, "videos", f"{demo_video}.mp4") if demo_video else os.path.join(BASE_DIR, "videos", "demo1.mp4")
elif demo_video:
    is_embedded = True
    auto_video_path = os.path.join(BASE_DIR, "videos", f"{demo_video}.mp4")
elif demo_shed:
    is_embedded = True
    auto_video_path = os.path.join(BASE_DIR, "videos", f"shed{demo_shed}.mp4")

if is_embedded and 'auto_started' not in st.session_state:
    st.session_state.processing = True
    st.session_state.auto_started = True

if mode != "video_only":
    st.title("🐔 Poultry Monitoring Dashboard")
    st.markdown("Real-time tracking, behavior analysis, and alerting using YOLOv8 & ByteTrack.")

# --- Sidebar ---
if not is_embedded:
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
else:
    video_path = auto_video_path

st.sidebar.header("Detection Settings")
model_size = st.sidebar.selectbox("Model Size", [
    "Nano (yolov8n.pt - Fast)", 
    "Small (yolov8s.pt - Better)", 
    "Medium (yolov8m.pt - Best/Slow)",
    "World Small (yolov8s-world.pt - Zero-Shot)",
    "World Medium (yolov8m-worldv2.pt - Zero-Shot)"
], index=0) # Default to Nano for maximum performance on CPU
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.10, 0.05)

target_classes = None # Do NOT filter classes, so misclassified chickens aren't dropped!

tracker_algo_ui = st.sidebar.selectbox("Tracking Algorithm", ["ByteTrack (Faster)", "BoTSORT (More Accurate)"])
# Force ByteTrack on CPU. BoTSORT uses deep learning ReID which maxes out the CPU on weak VMs.
selected_tracker = "bytetrack"

model_path_map = {
    "Nano (yolov8n.pt - Fast)": "yolov8n.pt",
    "Small (yolov8s.pt - Better)": "yolov8s.pt",
    "Medium (yolov8m.pt - Best/Slow)": "yolov8m.pt",
    "World Small (yolov8s-world.pt - Zero-Shot)": "yolov8s-world.pt",
    "World Medium (yolov8m-worldv2.pt - Zero-Shot)": "yolov8m-worldv2.pt"
}
selected_model_path = model_path_map[model_size]

st.sidebar.header("Identity Tag Manager")
st.sidebar.markdown("Map IDs to names (e.g. `5:Worker John`)")
tag_input = st.sidebar.text_area("Custom Tags", "")
custom_tags = {}
for line in tag_input.split('\n'):
    if ':' in line:
        k, v = line.split(':', 1)
        custom_tags[k.strip()] = v.strip()

st.sidebar.header("Integrations")
webhook_url = st.sidebar.text_input("Webhook URL", os.environ.get("WEBHOOK_URL", ""), placeholder="https://your-webhook-endpoint.com")
if 'webhook_dispatcher' not in st.session_state:
    st.session_state.webhook_dispatcher = WebhookDispatcher()

if not is_embedded:
    start_button = st.sidebar.button("Start Processing")

    # Stop processing using session state since button click reloads
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    if start_button:
        st.session_state.processing = True

    stop_button = st.sidebar.button("Stop")
    if stop_button:
        st.session_state.processing = False
else:
    if 'processing' not in st.session_state:
        st.session_state.processing = True

# --- Initialize Modules ---
from ultralytics import YOLOWorld

@st.cache_resource
def load_zone_detector():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["feeding pot", "water pot"])
    return model

@st.cache_resource
def load_tracker(model_path, tracker_algo):
    return PoultryTracker(model_path, tracker_algo)

# video_path is already correctly set above from auto_video_path or sidebar logic
if 'analytics' not in st.session_state:
    st.session_state.analytics = PoultryAnalytics()
    
if mode == "video_only":
    # --- PURE VIDEO EMBED ---
    SERVER_IP = os.environ.get("STREAM_HOST", "4.145.80.121")
    video_placeholder = st.empty()
    video_placeholder.markdown(
        f'<div style="width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; background-color: #0E1117;"><img src="http://{SERVER_IP}:8502/video_feed/{SESSION_ID}" style="max-width: 100%; max-height: 100%; object-fit: contain;"></div>',
        unsafe_allow_html=True
    )
else:
    # --- NORMAL / FULL DASHBOARD EMBED ---
    # Layout for Video and Dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tab1, tab2 = st.tabs(["🔴 Live Feed", "🔥 Heatmap"])
        
        with tab1:
            # Use native MJPEG stream via HTML <img> tag.
            # The session ID completely isolates each user's video feed!
            SERVER_IP = os.environ.get("STREAM_HOST", "4.145.80.121")
            
            video_placeholder = st.empty()
            video_placeholder.markdown(
                f'<img src="http://{SERVER_IP}:8502/video_feed/{SESSION_ID}" style="width: 100%; border-radius: 8px;">',
                unsafe_allow_html=True
            )
            
        with tab2:
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
    zone_detector = load_zone_detector()
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
            # Resize frame to 360p to drastically reduce CPU rendering/encoding bottleneck
            frame = cv2.resize(frame, (640, 360))
            h, w = frame.shape[:2]
            zone_manager = ZoneManager(w, h)
        else:
            st.error("Failed to read video stream")
            st.session_state.processing = False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        target_fps = 30 # Target a perfectly smooth 30 FPS video playback
        frame_delay = 1.0 / target_fps
        
        # Calculate how many frames to skip reading so the video still plays at normal speed
        video_frame_jump = max(1, int(fps / target_fps))
        
        frame_skip = 10 # Run YOLO only 3 times a second to keep CPU load extremely low
        zone_detected_once = False # Only run zone_detector ONCE at startup, not every 90 frames
        frame_count = 0
        last_detections = []
        
        while cap.isOpened() and st.session_state.processing:
            loop_start = time.time()
            
            # Use OpenCV's grab() to skip frames instantly without decoding them into RAM!
            # This is a massive CPU optimization over using cap.read() in a loop.
            for _ in range(video_frame_jump - 1):
                if not cap.grab():
                    break
                    
            ret, frame = cap.read()
            if not ret:
                # For a month-long demo, the video MUST loop endlessly when it finishes!
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                
            if not ret or frame is None:
                break
                
            frame = cv2.resize(frame, (640, 360))
                
            frame_count += 1
            
            if frame_count % frame_skip == 0 or frame_count == 1:
                # Process Heavy AI Models
                raw_detections = tracker.process_frame(frame, conf_threshold=conf_threshold, classes=target_classes)
                
                # Filter out standard YOLO detections that perfectly overlap with known pots
                detections = []
                
                # Filter out obvious inanimate objects that YOLO misclassifies the red pots as.
                ignored_classes = ["bowl", "cup", "vase", "potted plant", "fire hydrant", "bottle", "wine glass", "traffic light", "chair"]
                
                for d in raw_detections:
                    if d.get("class_name", "") in ignored_classes:
                        continue
                        
                    cx, cy = d["center"]
                    is_pot = False
                    for pot in zone_manager.pots["feed"] + zone_manager.pots["water"]:
                        px, py = pot["center"]
                        # If detection center is within 20 pixels of a pot center, it is the pot itself
                        if ((cx - px)**2 + (cy - py)**2)**0.5 < 20:
                            is_pot = True
                            break
                    if not is_pot:
                        detections.append(d)
                
                # Run zone_detector ONCE at startup only - not every 90 frames!
                # Running a 2nd AI model every 3 seconds is a massive hidden CPU spike.
                if not zone_detected_once:
                    world_results = zone_detector(frame, verbose=False)
                    world_dets = []
                    for box in world_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = zone_detector.names[cls_id]
                        if conf > 0.05:
                            world_dets.append({
                                "class_name": cls_name,
                                "center": ((x1+x2)//2, (y1+y2)//2),
                                "box": (x1, y1, x2, y2)
                            })
                    zone_manager.update_pots(world_dets)
                    zone_detected_once = True
                
                # Update analytics with new tracking
                for det in detections:
                    track_id = det["track_id"]
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["box"]
                    box_area = (x2 - x1) * (y2 - y1)
                    class_id = det.get("class_id", 14)
                    class_name = det.get("class_name", "bird")
                    zone = zone_manager.get_zone(cx, cy)
                    analytics.update(track_id, cx, cy, zone, class_id=class_id, class_name=class_name, box_area=box_area)
                    
                # Run flock-level analytics (huddling) at 1 FPS to save massive CPU time
                if frame_count % target_fps == 0:
                    analytics.analyze_flock()
                
                # Cache detections for the next skipped frames
                last_detections = detections
            else:
                # Use cached detections to keep the video looking smooth without running YOLO
                detections = last_detections
                
            # Draw overlays and push to the MJPEG stream server on EVERY frame.
            # This guarantees perfectly smooth 30 FPS video playback even if the AI is skipping frames!
            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics, custom_tags=custom_tags)
            
            # Write frame to shared file for the Flask MJPEG stream server
            _, buffer = cv2.imencode('.jpg', frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            push_video_frame(buffer.tobytes())
            
            if mode != "video_only":
                # Update Heatmap only once every 3 seconds to save CPU and bandwidth
                if frame_count % (target_fps * 3) == 0:
                    heatmap = visualizer.get_heatmap_overlay(frame_disp if 'frame_disp' in dir() else frame)
                    _, heatmap_buffer = cv2.imencode('.jpg', heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    heatmap_placeholder.image(heatmap_buffer.tobytes(), use_container_width=True)
                
                # 3. Update Charts & Metrics at 1 FPS (Every 30 frames)
                if frame_count % 30 == 0:
                    stats = analytics.get_summary_stats()
                    
                    # Update metrics
                    m_total.metric("Total Chickens", stats["total_chickens"])
                    m_active.metric("Active (Moving)", stats["active"])
                    m_humans.metric("Humans Detected", stats.get("total_humans", 0))
                    m_score.metric("Avg Activity Score", f"{stats['avg_activity_score']}%")
                    
                    m_zone.metric("Most Visited Zone", stats["most_visited_zone"].replace(" Zone", ""))
                    m_alerts.metric("Active Alerts", stats["alert_count"])
                    m_uniformity.metric("Size Uniformity", stats["size_uniformity"])
                    
                    # Update Charts
                    if stats["timeline"]["timestamps"]:
                        # Zone utilization chart
                        zone_df = pd.DataFrame({
                            "Feed": stats["timeline"]["feed_zone"],
                            "Water": stats["timeline"]["water_zone"],
                            "Rest": stats["timeline"]["rest_zone"]
                        }, index=stats["timeline"]["timestamps"])
                        zone_chart_placeholder.line_chart(zone_df)
                        
                        # Activity trend chart
                        act_df = pd.DataFrame({
                            "Avg Activity": stats["timeline"]["avg_activity"]
                        }, index=stats["timeline"]["timestamps"])
                        activity_chart_placeholder.line_chart(act_df)
                    
                    # Update Alerts
                    if analytics.alerts:
                        alert_html = "".join([f"<p style='color:red;'>⚠️ {a}</p>" for a in analytics.alerts[-5:]])
                        alert_box.markdown(alert_html, unsafe_allow_html=True)
                        
                # 4. Dispatch Webhook payload every 30 seconds (30 * target_fps frames)
                if webhook_url and frame_count % (target_fps * 30) == 0:
                    stats = analytics.get_summary_stats()
                    payload = {
                        "total_chickens": stats["total_chickens"],
                        "humans_detected": len([t for t in detections if t.get("class_name") in ["person", "human", "worker"]]),
                        "active_chickens": stats["active"],
                        "avg_activity_score": stats["avg_activity_score"],
                        "zone_occupancy": {
                            "Feed Zone": stats["timeline"]["feed_zone"][-1] if stats["timeline"]["feed_zone"] else 0,
                            "Water Zone": stats["timeline"]["water_zone"][-1] if stats["timeline"]["water_zone"] else 0,
                            "Rest Zone": stats["timeline"]["rest_zone"][-1] if stats["timeline"]["rest_zone"] else 0
                        },
                        "active_alerts": analytics.alerts[-5:] if analytics.alerts else []
                    }
                    st.session_state.webhook_dispatcher.dispatch(webhook_url, payload)

            elapsed = time.time() - loop_start
            
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
                
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 Embed Links")
        st.sidebar.write("Copy this code to embed the live annotated video in your own website:")
        embed_url = f"http://{os.environ.get('STREAM_HOST', '4.145.80.121')}:8501/?mode=embed"
        iframe_code = f'<iframe src="{embed_url}" width="100%" height="900" frameborder="0" allowfullscreen></iframe>'
        st.sidebar.code(iframe_code, language="html")
        
        cap.release()
        
    # Generate Reports at the end
    if not st.session_state.processing and mode != "video_only":
        with report_placeholder.container():
            st.subheader("Export Reports")
            rg = ReportGenerator(analytics)
            csv_file = rg.export_csv()
            pdf_file = rg.export_pdf()
            
            with open(csv_file, "rb") as f:
                st.download_button("Download CSV", f, file_name=os.path.basename(csv_file))
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(pdf_file))
