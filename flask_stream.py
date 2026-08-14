"""
Unified MJPEG Streaming + REST API Server + On-Demand 30 FPS AI Workers
========================================================================
Port 8502 — High-performance Flask Server with On-Demand Live AI Workers.

Features:
  - On-Demand Streamer Power Management (Saves CPU when streams aren't viewed)
  - 30 FPS smooth video streaming with non-blocking async YOLOv8 + ByteTrack tracking
  - 100% In-Memory RAM Caching (Zero disk I/O, zero file locks)
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
import torch

# Restrict PyTorch & OpenCV CPU thread pools to prevent CPU starvation
torch.set_num_threads(1)
cv2.setNumThreads(1)

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
# In-Memory Frame Cache & Activity Tracking
# ---------------------------------------------------------------------------
FRAME_CACHE = {}
LAST_VIEWED = {}
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

RESOLUTION = (640, 360)
JPEG_QUALITY = 70


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
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
            b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
        )

_BLACK_FRAME = _make_black_jpeg()


# ---------------------------------------------------------------------------
# Smooth Asynchronous Video & AI Worker
# ---------------------------------------------------------------------------
class SmoothLiveStreamer:
    def __init__(self, demo_name: str, config: dict):
        self.demo_name = demo_name
        self.video_path = config["path"]
        self.is_poultry = config["is_poultry"]
        self.camera_id = config["camera"]

        self.latest_dets = []
        self.lock = threading.Lock()
        self.ai_busy = False

    def _async_ai_worker(self, frame_copy, tracker, zone_manager, analytics, event_engine):
        """Runs fast YOLO tracking in background without blocking video stream."""
        try:
            raw_dets = tracker.process_frame(frame_copy, is_poultry=self.is_poultry)
            filtered_dets = [
                d for d in raw_dets
                if d.get("class_name", "") not in IGNORED_CLASSES
            ]
            for det in filtered_dets:
                cx, cy = det["center"]
                x1, y1, x2, y2 = det["box"]
                analytics.update(
                    det["track_id"], cx, cy,
                    zone_manager.get_zone(cx, cy),
                    class_id=det.get("class_id", 14),
                    class_name=det.get("class_name", "bird"),
                    box_area=(x2 - x1) * (y2 - y1),
                )
            new_events = event_engine.process(filtered_dets, analytics, zone_manager, is_poultry=self.is_poultry)
            for evt in new_events:
                state_store.add_event(evt)

            with self.lock:
                self.latest_dets = filtered_dets
        except Exception:
            pass
        finally:
            self.ai_busy = False

    def run(self):
        print(f"[STREAMER:{self.demo_name}] Starting 30 FPS live worker for {self.camera_id}...")
        tracker = PoultryTracker(model_path="yolov8n.pt", tracker_algo="bytetrack")
        visualizer = Visualizer()
        webhook_dispatcher = WebhookDispatcher(camera_id=self.camera_id)
        last_webhook_time = time.time() - 30.0

        # Mark initially viewed so first frames build immediately
        with CACHE_LOCK:
            LAST_VIEWED[self.demo_name] = time.time()

        while True:
            # Standby mode if stream hasn't been viewed in 15 seconds (Saves CPU)
            with CACHE_LOCK:
                time_since_viewed = time.time() - LAST_VIEWED.get(self.demo_name, time.time())

            if time_since_viewed > 15.0:
                time.sleep(0.3)
                continue

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                time.sleep(2)
                continue

            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1)
                continue

            first_frame = cv2.resize(first_frame, RESOLUTION)
            h, w = first_frame.shape[:2]
            zone_manager = ZoneManager(w, h, is_poultry=self.is_poultry)
            analytics = PoultryAnalytics()
            analytics.is_poultry = self.is_poultry
            event_engine = EventEngine()

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / min(source_fps, 30.0)

            while cap.isOpened():
                # Check standby condition inside video loop
                with CACHE_LOCK:
                    if time.time() - LAST_VIEWED.get(self.demo_name, time.time()) > 15.0:
                        break

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame = cv2.resize(frame, RESOLUTION)

                # Trigger AI tracking in background whenever free
                if not self.ai_busy:
                    self.ai_busy = True
                    threading.Thread(
                        target=self._async_ai_worker,
                        args=(frame.copy(), tracker, zone_manager, analytics, event_engine),
                        daemon=True
                    ).start()

                # Get latest available detections thread-safely
                with self.lock:
                    dets = list(self.latest_dets)

                # Draw overlays and cache JPEG frame in RAM at smooth 30 FPS
                frame_disp = visualizer.draw_zones(frame, zone_manager)
                frame_disp = visualizer.draw_tracking(frame_disp, dets, analytics)
                _, buf = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                jpeg_bytes = buf.tobytes()
                with CACHE_LOCK:
                    FRAME_CACHE[self.demo_name] = jpeg_bytes
                    FRAME_CACHE[f"poultry_frame_{self.demo_name}"] = jpeg_bytes

                # Webhook dispatch every 30s
                curr_t = time.time()
                if curr_t - last_webhook_time >= 30.0:
                    webhook_url = os.environ.get("WEBHOOK_URL", "https://webhook.site/7e5a79b4-1918-41a5-b090-50d74ed643c9")
                    if webhook_url:
                        stats = analytics.get_summary_stats()
                        zone_occ = zone_manager.get_zone_occupancy(dets)
                        events_since = state_store.flush_events_since_ping()
                        alerts_list = state_store.get_alerts(limit=10)
                        payload = webhook_dispatcher.build_payload(
                            analytics_stats=stats,
                            zone_occupancy=zone_occ,
                            events_since_last_ping=events_since,
                            alerts=alerts_list,
                            is_poultry=self.is_poultry,
                        )
                        webhook_dispatcher.dispatch(webhook_url, payload)
                        last_webhook_time = curr_t

                # Exact 30 FPS stream pacing — smooth continuous playback
                elapsed = time.time() - t0
                sleep_t = frame_delay - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            cap.release()
            time.sleep(0.05)


def start_background_workers():
    """Start live video stream workers for all demo videos."""
    for demo_name, config in DEMO_CONFIGS.items():
        if os.path.exists(config["path"]):
            streamer = SmoothLiveStreamer(demo_name, config)
            t = threading.Thread(
                target=streamer.run,
                daemon=True,
                name=f"streamer-{demo_name}"
            )
            t.start()
            time.sleep(1.0)


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
# MJPEG Streams (100% Smooth In-Memory RAM Streaming)
# ---------------------------------------------------------------------------
ALLOWED_DEMOS = {"demo1", "demo2", "demo3", "demo4"}

@app.route("/demo_feed/<demo_name>")
def demo_feed(demo_name: str):
    if demo_name not in ALLOWED_DEMOS:
        return "Not found", 404

    def generate():
        last_sent_bytes = None
        while True:
            with CACHE_LOCK:
                LAST_VIEWED[demo_name] = time.time()
                frame_data = FRAME_CACHE.get(demo_name)

            if frame_data and frame_data is not last_sent_bytes:
                last_sent_bytes = frame_data
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
                time.sleep(0.015)
            elif last_sent_bytes:
                time.sleep(0.01)
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
                time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    safe_session = "".join(c for c in session_id if c.isalnum() or c == "-")

    def generate():
        last_sent_bytes = None
        while True:
            with CACHE_LOCK:
                LAST_VIEWED["demo1"] = time.time()
                frame_data = FRAME_CACHE.get(safe_session) or FRAME_CACHE.get("demo1")

            if frame_data and frame_data is not last_sent_bytes:
                last_sent_bytes = frame_data
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
                time.sleep(0.015)
            elif last_sent_bytes:
                time.sleep(0.01)
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
                time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
