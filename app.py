import streamlit as st
import streamlit.components.v1 as components
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

# Restrict OpenCV to a single thread to prevent thread thrashing with PyTorch
cv2.setNumThreads(1)

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
    """Write the latest annotated frame to the shared RAM disk atomically."""
    frame_path = f"/dev/shm/poultry_frame_{SESSION_ID}.jpg"
    temp_path = frame_path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(jpeg_bytes)
        os.replace(temp_path, frame_path)
    except Exception:
        pass


def push_heatmap_frame(jpeg_bytes: bytes):
    """Write the latest heatmap frame to the shared RAM disk atomically."""
    frame_path = f"/dev/shm/poultry_heatmap_{SESSION_ID}.jpg"
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
url_webhook = query_params.get("webhook", "")
url_camera = query_params.get("camera", "")

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

# Sidebar — Video Input
if not is_embedded:
    st.sidebar.header("Video Input")
    input_source = st.sidebar.radio("Select Source", ["Demo Videos", "Upload Video", "Webcam", "RTSP Stream"], index=0)
    video_path = None
    if input_source == "Demo Videos":
        demo_choice = st.sidebar.selectbox("Select Demo Video", ["demo1", "demo2", "demo3", "demo4"], index=0)
        video_path = os.path.join(BASE_DIR, "videos", f"{demo_choice}.mp4")
    elif input_source == "Upload Video":
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

# Determine if this is a poultry video or restaurant/hotel video
is_poultry = True
if video_path is not None:
    video_name_lower = str(video_path).lower()
    if any(term in video_name_lower for term in ["demo3", "demo4", "shed3", "shed4", "restaurant", "hotel", "people"]):
        is_poultry = False

# Title
if mode != "video_only":
    if is_poultry:
        st.title("🐔 AI Poultry Monitoring Platform")
        st.markdown("Real-time tracking · Zone analytics · Event engine · Webhook integration")
    else:
        st.title("🏨 AI Smart Space Platform")
        st.markdown("Real-time occupancy tracking · Zone analytics · Event engine · Webhook integration")

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
    index=1,
)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.15, 0.05)
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

from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ["http", "https"]
    except Exception:
        return False

# Sidebar — Integrations
st.sidebar.header("Integrations")

camera_id = st.sidebar.text_input("Camera ID", url_camera or os.environ.get("CAMERA_ID", "CAM_01"))
webhook_url = st.sidebar.text_input(
    "Webhook URL",
    url_webhook or os.environ.get("WEBHOOK_URL", ""),
    placeholder="https://your-webhook-endpoint.com",
)

if "webhook_dispatcher" not in st.session_state:
    st.session_state.webhook_dispatcher = WebhookDispatcher(camera_id=camera_id)
else:
    st.session_state.webhook_dispatcher.camera_id = camera_id

is_url_valid = True
if webhook_url:
    is_url_valid = is_valid_url(webhook_url)
    if not is_url_valid:
        st.sidebar.error("⚠️ Invalid Webhook URL. Must be a valid HTTP/HTTPS URL.")

# Test Webhook Button
if webhook_url and is_url_valid:
    if st.sidebar.button("🧪 Test Webhook"):
        with st.sidebar.spinner("Sending sample payload..."):
            if "analytics" in st.session_state:
                stats = st.session_state.analytics.get_summary_stats()
            else:
                stats = {
                    "total_chickens": 10,
                    "total_humans": 1,
                    "active": 6,
                    "avg_activity_score": 45,
                    "alert_count": 0,
                    "size_uniformity": "High",
                    "most_visited_zone": "Feed Zone",
                    "timeline": {"timestamps": []}
                }
            
            # mock zone occupancy or current occupancy
            zone_occ = {"Feed Zone": 3, "Water Zone": 2, "Rest Zone": 5}
            
            payload = st.session_state.webhook_dispatcher.build_payload(
                analytics_stats=stats,
                zone_occupancy=zone_occ,
                events_since_last_ping=[],
                alerts=[],
                is_poultry=is_poultry
            )
            
            success, status_code, status_desc = st.session_state.webhook_dispatcher.dispatch(
                webhook_url, payload, sync=True
            )
            
            if success:
                st.sidebar.success(f"✅ Sent! Status: {status_code}")
            else:
                st.sidebar.error(f"❌ Failed: {status_desc}")

# Sidebar — Webhook Monitor
st.sidebar.subheader("📈 Webhook Monitor")
if webhook_url and is_url_valid:
    dispatcher = st.session_state.webhook_dispatcher
    st.sidebar.markdown(f"""
    * **Successes:** `{dispatcher.total_success}`
    * **Failures:** `{dispatcher.total_failed}`
    * **Last Sent:** `{dispatcher.last_sent_time or 'Never'}`
    * **Last Status:** `{dispatcher._last_status_code or 'N/A'}`
    """)
    if dispatcher.last_response_body:
        body_show = dispatcher.last_response_body[:500]
        if len(dispatcher.last_response_body) > 500:
            body_show += "\n... (truncated)"
        st.sidebar.text_area("Last Response Body", body_show, height=80, key="webhook_monitor_last_body")
else:
    st.sidebar.info("Configure a valid Webhook URL to monitor delivery.")

# Sidebar — API Info
try:
    from streamlit.web.server.websocket_headers import _get_websocket_headers
    headers = _get_websocket_headers()
    host_header = headers.get("Host", "") if headers else ""
    if host_header:
        SERVER_IP = host_header.split(":")[0]
    else:
        SERVER_IP = os.environ.get("STREAM_HOST", "20.219.17.164")
except Exception:
    SERVER_IP = os.environ.get("STREAM_HOST", "20.219.17.164")
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
        st.session_state.processing = True
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
    # Use components.html so JavaScript executes (st.markdown strips <script> tags)
    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0E1117; display:flex; justify-content:center; align-items:center;
        width:100vw; height:100vh; overflow:hidden; }}
img {{ max-width:100%; max-height:100%; object-fit:contain; }}
</style>
</head>
<body>
<img id="feed" src="" style="max-width:100%;max-height:100%;object-fit:contain;">
<script>
(function() {{
  try {{
    var host = window.parent.location.hostname;
  }} catch(e) {{
    var host = window.location.hostname;
  }}
  document.getElementById("feed").src =
    "http://" + host + ":8502/video_feed/{SESSION_ID}";
}})();
</script>
</body>
</html>""",
        height=720,
        scrolling=False,
    )
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        tab1, tab2 = st.tabs(["🔴 Live Feed", "🔥 Heatmap"])
        with tab1:
            video_placeholder = st.empty()
            video_placeholder.markdown(
                f'<img id="live_feed_video" src="http://{SERVER_IP}:8502/video_feed/{SESSION_ID}" '
                f'style="width:100%;border-radius:8px;">'
                f'<script>document.getElementById("live_feed_video").src = "http://" + window.location.hostname + ":8502/video_feed/{SESSION_ID}";</script>',
                unsafe_allow_html=True,
            )
        with tab2:
            heatmap_placeholder = st.empty()
            heatmap_placeholder.markdown(
                f'<img id="heatmap_feed_video" src="http://{SERVER_IP}:8502/heatmap_feed/{SESSION_ID}" '
                f'style="width:100%;border-radius:8px;">'
                f'<script>document.getElementById("heatmap_feed_video").src = "http://" + window.location.hostname + ":8502/heatmap_feed/{SESSION_ID}";</script>',
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

    def process_frame(self, frame, conf_threshold, classes, is_poultry=True):
        if not self.running:
            self.running = True
            frame_copy = frame.copy()

            def worker(f_cpy, c_thr, cls, is_p):
                dets = self.tracker.process_frame(
                    f_cpy, conf_threshold=c_thr, classes=cls, is_poultry=is_p
                )
                with self.lock:
                    self.raw_detections = dets
                self.running = False

            threading.Thread(target=worker, args=(frame_copy, conf_threshold, classes, is_poultry), daemon=True).start()

        with self.lock:
            return list(self.raw_detections)


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------
if st.session_state.processing and video_path is not None:
    base_tracker = load_tracker(selected_model_path, selected_tracker)
    tracker = base_tracker
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
            frame = cv2.resize(frame, (480, 270))  # Smaller resolution = less CPU + bandwidth
            h, w = frame.shape[:2]
            zone_manager = ZoneManager(w, h, is_poultry=is_poultry)
        else:
            st.error("Failed to read video stream")
            st.session_state.processing = False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
        target_fps = 8    # 8 FPS is smooth enough for CCTV-style monitoring and saves CPU
        frame_delay = 1.0 / target_fps
        video_frame_jump = max(1, int(fps / target_fps))
        frame_skip = 3    # Run YOLO every 3rd frame — reduces CPU by ~66%, detections interpolated
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
        
        extra_params = ""
        if demo_video:
            extra_params = f"&video={demo_video}"
        elif demo_shed:
            extra_params = f"&shed={demo_shed}"
            
        embed_url = f"http://{SERVER_IP}:8501/?mode=embed{extra_params}"
        video_only_url = f"http://{SERVER_IP}:8501/?mode=video_only{extra_params}"
        st.sidebar.code(
            f'<iframe src="{embed_url}" width="100%" height="900" frameborder="0" allowfullscreen></iframe>',
            language="html",
        )
        st.sidebar.markdown(f"**Video Only:** `{video_only_url}`")

        video_start_time = time.time()
        last_webhook_time = time.time() - 30.0  # Force immediate first dispatch
        while cap.isOpened() and st.session_state.processing:
            loop_start = time.time()

            # Dynamic frame dropping to maintain normal 1x playback speed
            elapsed_real_time = time.time() - video_start_time
            target_frame_index = int(elapsed_real_time * fps)
            
            if target_frame_index >= total_frames:
                video_start_time = time.time()
                target_frame_index = 0
                
            current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            ret = False
            frame = None
            if target_frame_index > current_frame_index:
                diff = target_frame_index - current_frame_index
                if diff < 10:
                    for _ in range(diff - 1):
                        cap.grab()
                    ret, frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                    ret, frame = cap.read()
            elif target_frame_index < current_frame_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                ret, frame = cap.read()
            else:
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
                    frame, conf_threshold=conf_threshold, classes=target_classes, is_poultry=is_poultry
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
                if is_poultry and not zone_detected_once:
                    world_results = zone_detector(frame, verbose=False)
                    world_dets = []
                    for box in world_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_val = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = zone_detector.names.get(cls_id, f"object_{cls_id}")
                        if conf_val > 0.05:
                            world_dets.append({
                                "class_name": cls_name,
                                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                                "box": (x1, y1, x2, y2),
                            })
                    zone_manager.update_pots(world_dets)
                    zone_detected_once = True

                # Update analytics
                analytics.is_poultry = is_poultry
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
                new_events = event_engine.process(detections, analytics, zone_manager, is_poultry=is_poultry)
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
                # Strip the 'timeline' key — it's large and only needed for UI charts,
                # not for the REST API state file
                api_stats = {k: v for k, v in stats.items() if k != "timeline"}
                state_store.update_metrics(api_stats, zone_occupancy)

                last_detections = detections
            else:
                detections = last_detections

            # ----------------------------------------------------------------
            # Video Rendering — Render and encode on EVERY frame at full 15 FPS
            # using the latest detections. This provides smooth video playback.
            # ----------------------------------------------------------------
            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics, custom_tags=custom_tags)
            _, buffer = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
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

                    if is_poultry:
                        m_total.metric("🐔 Total Chickens", stats["total_chickens"])
                        m_active.metric("✅ Active", stats["active"])
                        m_humans.metric("👤 Humans", stats.get("total_humans", 0))
                        m_score.metric("⚡ Avg Activity", f"{stats['avg_activity_score']}%")
                        m_zone.metric("📍 Top Zone", stats["most_visited_zone"].replace(" Zone", ""))
                        m_alerts.metric("🚨 Alerts", stats["alert_count"])
                        m_uniformity.metric("📐 Uniformity", stats["size_uniformity"])
                    else:
                        m_total.metric("👤 Total People", stats["total_humans"])
                        m_active.metric("✅ Active People", stats["active_humans"])
                        m_humans.metric("👤 Inactive People", stats["inactive_humans"])
                        m_score.metric("⚡ Avg Activity", f"{stats['avg_human_activity_score']}%")
                        m_zone.metric("📍 Top Zone", stats["most_visited_zone"].replace(" Zone", ""))
                        m_alerts.metric("🚨 Alerts", stats["alert_count"])
                        m_uniformity.metric("📐 Uniformity", "N/A")

                    # Update heavy UI charts every 5 seconds to prevent browser freeze/lag
                    if frame_count % (target_fps * 5) == 0 and stats["timeline"]["timestamps"]:
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
            # Webhook — every 30 seconds (real-world time)
            # Runs in ALL modes including video_only
            # --------------------------------------------------------
            current_time = time.time()
            if webhook_url and is_url_valid and (current_time - last_webhook_time >= 30.0):
                print(f"[WEBHOOK] Firing webhook at frame {frame_count}, elapsed={current_time - last_webhook_time:.1f}s")
                stats = analytics.get_summary_stats()
                zone_occ = zone_manager.get_zone_occupancy(last_detections)
                events_since = state_store.flush_events_since_ping()
                alerts_list = state_store.get_alerts(limit=10)

                payload = st.session_state.webhook_dispatcher.build_payload(
                    analytics_stats=stats,
                    zone_occupancy=zone_occ,
                    events_since_last_ping=events_since,
                    alerts=alerts_list,
                    is_poultry=is_poultry,
                )
                success, status_code, desc = st.session_state.webhook_dispatcher.dispatch(webhook_url, payload)
                print(f"[WEBHOOK] Result: success={success}, status={status_code}, desc={desc}")
                last_webhook_time = current_time

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
