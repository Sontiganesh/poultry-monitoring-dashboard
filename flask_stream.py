"""
Unified MJPEG Streaming + REST API Server + In-Memory AI Workers
==================================================================
Port 8502 — Standalone service running Flask, In-Memory AI Video Workers, and REST APIs.

Features:
  - 100% In-Memory JPEG Frame Caching (Zero disk I/O, zero file locks)
  - 25 FPS smooth natural playback with live YOLOv8 + ByteTrack AI tracking
  - Built-in automatic video workers for demo1, demo2, demo3, demo4
  - Automatic 30-second Webhook Dispatcher
  - REST API Endpoints (/api/status, /api/metrics, /api/alerts, /api/events)
"""

from flask import Flask, Response, jsonify, request
import time
import os
import sys
import datetime
import threading
import cv2

# Add the project root to the Python path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import PoultryTracker
from src.visualization import Visualizer
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.event_engine import EventEngine
from src.webhook import WebhookDispatcher
from src import state_store

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-Memory Frame Cache & Global State
# ---------------------------------------------------------------------------
FRAME_CACHE = {}
CACHE_LOCK = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEMO_CONFIGS = {
    "demo1": {"path": os.path.join(BASE_DIR, "videos", "demo1.mp4"), "is_poultry": True,  "camera": "CAM_01"},
    "demo2": {"path": os.path.join(BASE_DIR, "videos", "demo2.mp4"), "is_poultry": True,  "camera": "CAM_02"},
    "demo3": {"path": os.path.join(BASE_DIR, "videos", "demo3.mp4"), "is_poultry": False, "camera": "CAM_03"},
    "demo4": {"path": os.path.join(BASE_DIR, "videos", "demo4.mp4"), "is_poultry": False, "camera": "CAM_04"},
}

IGNORED_CLASSES = {
    "bowl", "cup", "vase", "potted plant", "fire hydrant",
    "bottle", "wine glass", "traffic light", "chair",
}

RESOLUTION = (480, 270)
JPEG_QUALITY = 60
TRACK_EVERY_N = 2


def _make_black_jpeg(width=640, height=360):
    try:
        import numpy as np
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        return buf.tobytes()
    except Exception:
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\1\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
            b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
        )

_BLACK_FRAME = _make_black_jpeg()


# ---------------------------------------------------------------------------
# Background In-Memory AI Video Workers
# ---------------------------------------------------------------------------
def _in_memory_video_worker(demo_name: str, config: dict):
    video_path = config["path"]
    is_poultry = config["is_poultry"]
    camera_id = config["camera"]

    print(f"[WORKER:{demo_name}] Loading YOLO tracker for {camera_id}...")
    tracker = PoultryTracker(model_path="yolov8n.pt", tracker_algo="bytetrack")
    visualizer = Visualizer()
    webhook_dispatcher = WebhookDispatcher(camera_id=camera_id)
    last_webhook_time = time.time() - 30.0

    print(f"[WORKER:{demo_name}] Active — In-memory live streaming.")

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            time.sleep(3)
            continue

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(2)
            continue

        first_frame = cv2.resize(first_frame, RESOLUTION)
        h, w = first_frame.shape[:2]
        zone_manager = ZoneManager(w, h, is_poultry=is_poultry)
        analytics = PoultryAnalytics()
        analytics.is_poultry = is_poultry
        event_engine = EventEngine()

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_delay = 1.0 / source_fps
        frame_count = 0
        current_dets = []

        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                tracker = PoultryTracker(model_path="yolov8n.pt", tracker_algo="bytetrack")
                frame_count = 0

            frame = cv2.resize(frame, RESOLUTION)
            frame_count += 1

            if frame_count % TRACK_EVERY_N == 0 or frame_count == 1:
                raw_dets = tracker.process_frame(frame, is_poultry=is_poultry)
                current_dets = [
                    d for d in raw_dets
                    if d.get("class_name", "") not in IGNORED_CLASSES
                ]
                for det in current_dets:
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["box"]
                    analytics.update(
                        det["track_id"], cx, cy,
                        zone_manager.get_zone(cx, cy),
                        class_id=det.get("class_id", 14),
                        class_name=det.get("class_name", "bird"),
                        box_area=(x2 - x1) * (y2 - y1),
                    )
                new_events = event_engine.process(current_dets, analytics, zone_manager, is_poultry=is_poultry)
                for evt in new_events:
                    state_store.add_event(evt)

            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, current_dets, analytics)
            _, buf = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            # Store in RAM cache thread-safely (No disk I/O!)
            jpeg_bytes = buf.tobytes()
            with CACHE_LOCK:
                FRAME_CACHE[demo_name] = jpeg_bytes
                FRAME_CACHE[f"poultry_frame_{demo_name}"] = jpeg_bytes

            # Webhook dispatch every 30 seconds
            current_time = time.time()
            if current_time - last_webhook_time >= 30.0:
                webhook_url = os.environ.get("WEBHOOK_URL", "https://webhook.site/7e5a79b4-1918-41a5-b090-50d74ed643c9")
                if webhook_url:
                    stats = analytics.get_summary_stats()
                    zone_occ = zone_manager.get_zone_occupancy(current_dets)
                    events_since = state_store.flush_events_since_ping()
                    alerts_list = state_store.get_alerts(limit=10)

                    payload = webhook_dispatcher.build_payload(
                        analytics_stats=stats,
                        zone_occupancy=zone_occ,
                        events_since_last_ping=events_since,
                        alerts=alerts_list,
                        is_poultry=is_poultry,
                    )
                    webhook_dispatcher.dispatch(webhook_url, payload)
                    last_webhook_time = current_time

            # Precision 1.0x FPS Limiter
            elapsed = time.time() - loop_start
            sleep_t = target_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()
        time.sleep(0.05)


def start_background_workers():
    """Start in-memory AI streaming workers for all 4 demo videos."""
    for demo_name, config in DEMO_CONFIGS.items():
        if os.path.exists(config["path"]):
            t = threading.Thread(
                target=_in_memory_video_worker,
                args=(demo_name, config),
                daemon=True,
                name=f"worker-{demo_name}"
            )
            t.start()
            time.sleep(1.5)


# Start background workers automatically when server starts
start_background_workers()


# ---------------------------------------------------------------------------
# CORS helper
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ---------------------------------------------------------------------------
# MJPEG Streams (100% In-Memory RAM Streaming)
# ---------------------------------------------------------------------------
ALLOWED_DEMOS = {"demo1", "demo2", "demo3", "demo4"}

@app.route("/demo_feed/<demo_name>")
def demo_feed(demo_name: str):
    if demo_name not in ALLOWED_DEMOS:
        return "Not found", 404

    def generate():
        last_bytes = None
        while True:
            with CACHE_LOCK:
                frame_data = FRAME_CACHE.get(demo_name)
            if frame_data:
                last_bytes = frame_data

            if last_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n")
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
            time.sleep(0.04)  # ~25 FPS stream rate

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    safe_session = "".join(c for c in session_id if c.isalnum() or c == "-")

    def generate():
        last_bytes = None
        while True:
            with CACHE_LOCK:
                # Check for session frame or fall back to demo1
                frame_data = FRAME_CACHE.get(safe_session) or FRAME_CACHE.get("demo1")
            if frame_data:
                last_bytes = frame_data

            if last_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n")
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/heatmap_feed/<session_id>")
def heatmap_feed(session_id: str):
    def generate():
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
            time.sleep(0.2)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}), 200

@app.route("/api/status")
def api_status():
    return jsonify(state_store.get_status()), 200

@app.route("/api/metrics")
def api_metrics():
    return jsonify(state_store.get_metrics()), 200

@app.route("/api/alerts")
def api_alerts():
    limit = request.args.get("limit", 10, type=int)
    return jsonify({"alerts": state_store.get_alerts(limit=limit)}), 200

@app.route("/api/events")
def api_events():
    limit = request.args.get("limit", 20, type=int)
    return jsonify({"events": state_store.get_events(limit=limit)}), 200


if __name__ == "__main__":
    print("[SERVER] Starting unified MJPEG + REST API server on port 8502...")
    app.run(host="0.0.0.0", port=8502, threaded=True)
