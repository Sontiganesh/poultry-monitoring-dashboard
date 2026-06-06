import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
import torch
import warnings
import pandas as pd
import threading
import uuid

# Restrict PyTorch to a single thread to prevent it from aggressively consuming
# all vCPUs (100%+) during OpenMP matrix multiplications.
torch.set_num_threads(1)

from src.tracker import PoultryTracker
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.visualization import Visualizer
from src.reporting import ReportGenerator
from src.webhook import WebhookDispatcher
from src.event_engine import EventEngine
from src import state_store

# ---------------------------------------------------------------------------
# Session ID — isolates each browser tab's video stream
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id


def push_video_frame(jpeg_bytes: bytes):
    """Write the latest annotated frame to the shared file atomically."""
    frame_path = f"/tmp/poultry_frame_{SESSION_ID}.jpg"
    temp_path = frame_path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(jpeg_bytes)
        os.replace(temp_path, frame_path)
    except Exception:
        pass


def push_heatmap_frame(jpeg_bytes: bytes):
    """Write the latest heatmap frame to the shared file atomically."""
    frame_path = f"/tmp/poultry_heatmap_{SESSION_ID}.jpg"
    temp_path = frame_path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(jpeg_bytes)
        os.replace(temp_path, frame_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Poultry Monitoring", layout="wide")

# ---------------------------------------------------------------------------
# Query Parameters
# ---------------------------------------------------------------------------
query_params = st.query_params
mode = query_params.get("mode")
demo_video = query_params.get("video")
demo_shed = query_params.get("shed")

# ---------------------------------------------------------------------------
# Embed Mode CSS
# ---------------------------------------------------------------------------
if mode in ["embed", "video_only"]:
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
    if "auto_played" not in st.session_state:
        st.session_state.processing = True
        st.session_state.auto_played = True

# ---------------------------------------------------------------------------
# Sidebar & Video Input
# ---------------------------------------------------------------------------
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("Configuration")

is_embedded = False
auto_video_path = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine video path from URL params
if mode in ["embed", "video_only"]:
    is_embedded = True
    auto_video_path = (
        os.path.join(BASE_DIR, "videos", f"{demo_video}.mp4")
        if demo_video
        else os.path.join(BASE_DIR, "videos", "demo1.mp4")
    )
elif demo_video:
    is_embedded = True
    auto_video_path = os.path.join(BASE_DIR, "videos", f"{demo_video}.mp4")
elif demo_shed:
    is_embedded = True
    auto_video_path = os.path.join(BASE_DIR, "videos", f"shed{demo_shed}.mp4")

if is_embedded and "auto_started" not in st.session_state:
    st.session_state.processing = True
    st.session_state.auto_started = True

# Title
if mode != "video_only":
    st.title("🐔 AI Poultry Monitoring Platform")
    st.markdown("Real-time tracking · Zone analytics · Event engine · Webhook integration")

# Sidebar — Video Input
if not is_embedded:
    st.sidebar.header("Video Input")
    input_source = st.sidebar.radio("Select Source", ["Upload Video", "Webcam", "RTSP Stream"])
    video_path = None
    if input_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload MP4", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
    elif input_source == "Webcam":
        video_path = 0
    elif input_source == "RTSP Stream":
        video_path = st.sidebar.text_input("RTSP URL", "rtsp://localhost:8554/stream")
else:
    video_path = auto_video_path

# Sidebar — Detection Settings
st.sidebar.header("Detection Settings")
model_size = st.sidebar.selectbox(
    "Model Size",
    [
        "Nano (yolov8n.pt - Fast)",
        "Small (yolov8s.pt - Better)",
        "Medium (yolov8m.pt - Best/Slow)",
        "World Small (yolov8s-world.pt - Zero-Shot)",
        "World Medium (yolov8m-worldv2.pt - Zero-Shot)",
    ],
    index=0,
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.10, 0.05)
target_classes = None
selected_tracker = "bytetrack"

model_path_map = {
    "Nano (yolov8n.pt - Fast)": "yolov8n.pt",
    "Small (yolov8s.pt - Better)": "yolov8s.pt",
    "Medium (yolov8m.pt - Best/Slow)": "yolov8m.pt",
    "World Small (yolov8s-world.pt - Zero-Shot)": "yolov8s-world.pt",
    "World Medium (yolov8m-worldv2.pt - Zero-Shot)": "yolov8m-worldv2.pt",
}
selected_model_path = model_path_map[model_size]

# Sidebar — Identity Tag Manager
st.sidebar.header("Identity Tag Manager")
st.sidebar.markdown("Map IDs to names (e.g. `5:Worker John`)")
tag_input = st.sidebar.text_area("Custom Tags", "")
custom_tags = {}
for line in tag_input.split("\n"):
    if ":" in line:
        k, v = line.split(":", 1)
        custom_tags[k.strip()] = v.strip()

# Sidebar — Integrations
st.sidebar.header("Integrations")

camera_id = st.sidebar.text_input("Camera ID", os.environ.get("CAMERA_ID", "CAM_01"))
webhook_url = st.sidebar.text_input(
    "Webhook URL",
    os.environ.get("WEBHOOK_URL", ""),
    placeholder="https://your-webhook-endpoint.com",
)

if "webhook_dispatcher" not in st.session_state:
    st.session_state.webhook_dispatcher = WebhookDispatcher(camera_id=camera_id)

# Sidebar — API Info
SERVER_IP = os.environ.get("STREAM_HOST", "4.145.80.121")
st.sidebar.header("REST API Endpoints")
st.sidebar.markdown(f"""
| Endpoint | URL |
|---|---|
| Status | [`/api/status`](http://{SERVER_IP}:8502/api/status) |
| Metrics | [`/api/metrics`](http://{SERVER_IP}:8502/api/metrics) |
| Alerts | [`/api/alerts`](http://{SERVER_IP}:8502/api/alerts) |
| Events | [`/api/events`](http://{SERVER_IP}:8502/api/events) |
""")

# Sidebar — Start/Stop
if not is_embedded:
    start_button = st.sidebar.button("▶ Start Processing")
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if start_button:
        st.session_state.processing = True
    stop_button = st.sidebar.button("⏹ Stop")
    if stop_button:
        st.session_state.processing = False
        state_store.set_status("stopped")
else:
    if "processing" not in st.session_state:
        st.session_state.processing = True

# ---------------------------------------------------------------------------
# Cache Loaders
# ---------------------------------------------------------------------------
from ultralytics import YOLOWorld


@st.cache_resource
def load_zone_detector():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["feeding pot", "water pot"])
    return model


@st.cache_resource
def load_tracker(model_path, tracker_algo):
    return PoultryTracker(model_path, tracker_algo)


# ---------------------------------------------------------------------------
# Session-level Analytics / EventEngine (persists across Streamlit reruns)
# ---------------------------------------------------------------------------
if "analytics" not in st.session_state:
    st.session_state.analytics = PoultryAnalytics()
if "event_engine" not in st.session_state:
    st.session_state.event_engine = EventEngine()

# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------
if mode == "video_only":
    # Pure embed — just the video, nothing else
    video_placeholder = st.empty()
    video_placeholder.markdown(
        f'<div style="width:100vw;height:100vh;display:flex;justify-content:center;'
        f'align-items:center;background:#0E1117;">'
        f'<img src="http://{SERVER_IP}:8502/video_feed/{SESSION_ID}" '
        f'style="max-width:100%;max-height:100%;object-fit:contain;"></div>',
        unsafe_allow_html=True,
    )
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        tab1, tab2 = st.tabs(["🔴 Live Feed", "🔥 Heatmap"])
        with tab1:
            video_placeholder = st.empty()
            video_placeholder.markdown(
                f'<img src="http://{SERVER_IP}:8502/video_feed/{SESSION_ID}" '
                f'style="width:100%;border-radius:8px;">',
                unsafe_allow_html=True,
            )
        with tab2:
            heatmap_placeholder = st.empty()
            heatmap_placeholder.markdown(
                f'<img src="http://{SERVER_IP}:8502/heatmap_feed/{SESSION_ID}" '
                f'style="width:100%;border-radius:8px;">',
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("📊 Analytics Dashboard")
        mc1, mc2, mc3 = st.columns(3)
        m_total = mc1.empty()
        m_active = mc2.empty()
        m_humans = mc3.empty()

        mc4, mc5 = st.columns(2)
        m_score = mc4.empty()
        m_zone = mc5.empty()

        mc6, mc7 = st.columns(2)
        m_alerts = mc6.empty()
        m_uniformity = mc7.empty()

        st.subheader("📈 Zone Occupancy (Live)")
        zone_chart_placeholder = st.empty()

        st.subheader("⚡ Activity Trend")
        activity_chart_placeholder = st.empty()

        st.subheader("🚨 Recent Alerts")
        alert_box = st.empty()

        st.subheader("🗂 Recent Events")
        event_box = st.empty()

        report_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Async Tracker Wrapper — decouples heavy YOLO inference from the video loop
# ---------------------------------------------------------------------------
class AsyncTracker:
    def __init__(self, base_tracker):
        self.tracker = base_tracker
        self.raw_detections = []
        self.lock = threading.Lock()
        self.running = False

    def process_frame(self, frame, conf_threshold, classes):
        if not self.running:
            self.running = True
            frame_copy = frame.copy()

            def worker():
                dets = self.tracker.process_frame(
                    frame_copy, conf_threshold=conf_threshold, classes=classes
                )
                with self.lock:
                    self.raw_detections = dets
                self.running = False

            threading.Thread(target=worker, daemon=True).start()

        with self.lock:
            return list(self.raw_detections)


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------
if st.session_state.processing and video_path is not None:
    base_tracker = load_tracker(selected_model_path, selected_tracker)
    tracker = AsyncTracker(base_tracker)
    zone_detector = load_zone_detector()
    visualizer = Visualizer()
    analytics = st.session_state.analytics
    event_engine = st.session_state.event_engine

    # Update StateStore
    state_store.set_camera_id(camera_id)
    state_store.set_status("running", video_source=os.path.basename(str(video_path)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video source: {video_path}")
        st.session_state.processing = False
    else:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            h, w = frame.shape[:2]
            zone_manager = ZoneManager(w, h)
        else:
            st.error("Failed to read video stream")
            st.session_state.processing = False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        target_fps = 30
        frame_delay = 1.0 / target_fps
        video_frame_jump = max(1, int(fps / target_fps))
        frame_skip = 10  # Run YOLO every 10th frame; video pushed every frame
        zone_detected_once = False
        frame_count = 0
        last_detections = []
        ignored_classes = [
            "bowl", "cup", "vase", "potted plant", "fire hydrant",
            "bottle", "wine glass", "traffic light", "chair",
        ]

        # Sidebar embed links
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 Embed Links")
        embed_url = f"http://{SERVER_IP}:8501/?mode=embed"
        video_only_url = f"http://{SERVER_IP}:8501/?mode=video_only"
        st.sidebar.code(
            f'<iframe src="{embed_url}" width="100%" height="900" frameborder="0" allowfullscreen></iframe>',
            language="html",
        )
        st.sidebar.markdown(f"**Video Only:** `{video_only_url}`")

        while cap.isOpened() and st.session_state.processing:
            loop_start = time.time()

            # Skip frames efficiently without decoding
            for _ in range(video_frame_jump - 1):
                if not cap.grab():
                    break

            ret, frame = cap.read()
            if not ret:
                # Loop video endlessly
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if not ret or frame is None:
                break

            frame = cv2.resize(frame, (640, 360))
            frame_count += 1

            # ----------------------------------------------------------------
            # AI Processing — runs every N frames asynchronously
            # ----------------------------------------------------------------
            if frame_count % frame_skip == 0 or frame_count == 1:
                raw_detections = tracker.process_frame(
                    frame, conf_threshold=conf_threshold, classes=target_classes
                )

                # Filter ignored classes and pot overlaps
                detections = []
                for d in raw_detections:
                    if d.get("class_name", "") in ignored_classes:
                        continue
                    cx, cy = d["center"]
                    is_pot = any(
                        ((cx - pot["center"][0]) ** 2 + (cy - pot["center"][1]) ** 2) ** 0.5 < 20
                        for pot in zone_manager.pots["feed"] + zone_manager.pots["water"]
                    )
                    if not is_pot:
                        detections.append(d)

                # Zone detector — run ONCE at startup only
                if not zone_detected_once:
                    world_results = zone_detector(frame, verbose=False)
                    world_dets = []
                    for box in world_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_val = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = zone_detector.names[cls_id]
                        if conf_val > 0.05:
                            world_dets.append({
                                "class_name": cls_name,
                                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                                "box": (x1, y1, x2, y2),
                            })
                    zone_manager.update_pots(world_dets)
                    zone_detected_once = True

                # Update analytics
                for det in detections:
                    track_id = det["track_id"]
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["box"]
                    box_area = (x2 - x1) * (y2 - y1)
                    class_id = det.get("class_id", 14)
                    class_name = det.get("class_name", "bird")
                    zone = zone_manager.get_zone(cx, cy)
                    analytics.update(
                        track_id, cx, cy, zone,
                        class_id=class_id, class_name=class_name, box_area=box_area
                    )

                # Flock analytics (huddling) at ~1 FPS
                if frame_count % target_fps == 0:
                    analytics.analyze_flock()

                # --------------------------------------------------------
                # Event Engine — fires zone entry/exit, inactivity, crowd
                # --------------------------------------------------------
                new_events = event_engine.process(detections, analytics, zone_manager)
                for evt in new_events:
                    state_store.add_event(evt)

                # Push alerts from analytics into StateStore
                if analytics.alerts:
                    # Only add alerts that haven't been added yet
                    if not hasattr(st.session_state, "_last_alert_count"):
                        st.session_state._last_alert_count = 0
                    new_alert_count = len(analytics.alerts)
                    for alert in analytics.alerts[st.session_state._last_alert_count:]:
                        state_store.add_alert(alert)
                    st.session_state._last_alert_count = new_alert_count

                # --------------------------------------------------------
                # StateStore — update metrics + zone occupancy
                # --------------------------------------------------------
                stats = analytics.get_summary_stats()
                zone_occupancy = zone_manager.get_zone_occupancy(detections)
                state_store.update_metrics(stats, zone_occupancy)

                last_detections = detections
            else:
                detections = last_detections

            # ----------------------------------------------------------------
            # Video Rendering — EVERY frame for smooth playback
            # ----------------------------------------------------------------
            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics, custom_tags=custom_tags)

            _, buffer = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            push_video_frame(buffer.tobytes())

            if mode != "video_only":
                # Heatmap — every 5 seconds
                if frame_count % (target_fps * 5) == 0:
                    heatmap = visualizer.get_heatmap_overlay(frame_disp)
                    _, hm_buf = cv2.imencode(".jpg", heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    push_heatmap_frame(hm_buf.tobytes())

                # Dashboard metrics + charts — every 1 second
                if frame_count % target_fps == 0:
                    stats = analytics.get_summary_stats()

                    m_total.metric("🐔 Total Chickens", stats["total_chickens"])
                    m_active.metric("✅ Active", stats["active"])
                    m_humans.metric("👤 Humans", stats.get("total_humans", 0))
                    m_score.metric("⚡ Avg Activity", f"{stats['avg_activity_score']}%")
                    m_zone.metric("📍 Top Zone", stats["most_visited_zone"].replace(" Zone", ""))
                    m_alerts.metric("🚨 Alerts", stats["alert_count"])
                    m_uniformity.metric("📐 Uniformity", stats["size_uniformity"])

                    # Zone occupancy chart
                    if stats["timeline"]["timestamps"]:
                        zone_df = pd.DataFrame(
                            {
                                "Feed": stats["timeline"]["feed_zone"],
                                "Water": stats["timeline"]["water_zone"],
                                "Rest": stats["timeline"]["rest_zone"],
                            },
                            index=stats["timeline"]["timestamps"],
                        )
                        zone_chart_placeholder.line_chart(zone_df)

                        act_df = pd.DataFrame(
                            {"Avg Activity": stats["timeline"]["avg_activity"]},
                            index=stats["timeline"]["timestamps"],
                        )
                        activity_chart_placeholder.line_chart(act_df)

                    # Alerts panel
                    if analytics.alerts:
                        alert_html = "".join(
                            [f"<p style='color:#ff4b4b;margin:2px 0'>⚠️ {a}</p>"
                             for a in analytics.alerts[-5:]]
                        )
                        alert_box.markdown(alert_html, unsafe_allow_html=True)

                    # Events panel (last 5 events from StateStore)
                    recent_events = state_store.get_events(limit=5)
                    if recent_events:
                        evt_html = "".join(
                            [f"<p style='color:#ffa500;margin:2px 0;font-size:0.85em'>"
                             f"🔔 <b>{e['event_type']}</b> — {e.get('details','')}</p>"
                             for e in reversed(recent_events)]
                        )
                        event_box.markdown(evt_html, unsafe_allow_html=True)

                # --------------------------------------------------------
                # Webhook — every 30 seconds
                # --------------------------------------------------------
                if webhook_url and frame_count % (target_fps * 30) == 0:
                    stats = analytics.get_summary_stats()
                    zone_occ = zone_manager.get_zone_occupancy(last_detections)
                    events_since = state_store.flush_events_since_ping()
                    alerts_list = state_store.get_alerts(limit=10)

                    payload = st.session_state.webhook_dispatcher.build_payload(
                        analytics_stats=stats,
                        zone_occupancy=zone_occ,
                        events_since_last_ping=events_since,
                        alerts=alerts_list,
                    )
                    st.session_state.webhook_dispatcher.dispatch(webhook_url, payload)

            # Frame rate limiter
            elapsed = time.time() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

        state_store.set_status("stopped")
        cap.release()

    # Report generation after processing ends
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
