"""
Unified MJPEG Streaming + REST API Server
=========================================
Port 8502 — runs as a standalone systemd service alongside Streamlit.

MJPEG Endpoints:
  GET /video_feed/<session_id>       — Live annotated video stream
  GET /heatmap_feed/<session_id>     — Live heatmap overlay stream

REST API Endpoints:
  GET  /api/status                   — System status + uptime
  GET  /api/metrics                  — Full analytics snapshot
  GET  /api/alerts                   — Last N alerts
  GET  /api/events                   — Last N events
  POST /api/analyze                  — Trigger one-shot analysis (future use)

Health:
  GET  /health                       — Simple health check
"""

from flask import Flask, Response, jsonify, request
import time
import os
import sys
import datetime
import struct

# Add the project root to the Python path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import state_store

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Black placeholder JPEG — served while AI processing loop starts up
# Prevents broken image icon on first page load
# ---------------------------------------------------------------------------
def _make_black_jpeg(width=640, height=360):
    """Generate a minimal black JPEG frame without PIL dependency."""
    try:
        import numpy as np
        import cv2
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        return buf.tobytes()
    except Exception:
        # Minimal 1x1 black JPEG bytes as fallback
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
# CORS helper — allow cross-origin requests for embedded product integration
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ---------------------------------------------------------------------------
# MJPEG Stream
# ---------------------------------------------------------------------------

@app.route("/video_feed/<session_id>")
def video_feed(session_id: str):
    safe_session = "".join(c for c in session_id if c.isalnum() or c == "-")
    frame_path = f"/dev/shm/poultry_frame_{safe_session}.jpg"

    def generate():
        last_bytes = None
        while True:
            if os.path.exists(frame_path):
                try:
                    with open(frame_path, "rb") as f:
                        buf = f.read()
                    if buf and len(buf) > 1000:
                        last_bytes = buf
                except Exception:
                    pass

            if last_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n")
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
            time.sleep(0.04)  # ~25 FPS smooth stream

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Shared Demo Feed — served by video_worker.py (no per-session YOLO)
# ---------------------------------------------------------------------------
ALLOWED_DEMOS = {"demo1", "demo2", "demo3", "demo4"}

@app.route("/demo_feed/<demo_name>")
def demo_feed(demo_name: str):
    """
    MJPEG stream backed by video_worker.py shared frame files.
    All viewers of the same demo share one YOLO loop — no lag from multiple viewers.
    """
    if demo_name not in ALLOWED_DEMOS:
        return "Not found", 404

    frame_path = f"/dev/shm/poultry_frame_{demo_name}.jpg"

    def generate():
        last_bytes = None
        while True:
            if os.path.exists(frame_path):
                try:
                    with open(frame_path, "rb") as f:
                        buf = f.read()
                    if buf and len(buf) > 1000:
                        last_bytes = buf
                except Exception:
                    pass

            if last_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n")
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + _BLACK_FRAME + b"\r\n")
            time.sleep(0.04)  # ~25 FPS smooth stream

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Smooth Pre-rendered Feed — streams offline-processed MP4s as MJPEG loops
# Zero real-time YOLO. Perfectly smooth regardless of viewer count.
# Pre-render once with: python run_offline_tracking.py
# ---------------------------------------------------------------------------
@app.route("/smooth_feed/<demo_name>")
def smooth_feed(demo_name: str):
    """
    Streams a pre-rendered annotated MP4 as a looping MJPEG stream.
    No live inference — just serves frames from disk at target FPS.
    """
    if demo_name not in ALLOWED_DEMOS:
        return "Not found", 404

    video_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output", f"{demo_name}_tracked.mp4"
    )

    if not os.path.exists(video_path):
        return f"Pre-rendered video not found. Run: python run_offline_tracking.py", 503

    def generate():
        import cv2 as _cv2
        while True:  # loop forever
            cap = _cv2.VideoCapture(video_path)
            source_fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
            frame_delay = 1.0 / min(source_fps, 15)  # stream at up to 15 FPS

            while cap.isOpened():
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break  # video ended — restart outer loop
                _, buf = _cv2.imencode(
                    ".jpg", frame, [int(_cv2.IMWRITE_JPEG_QUALITY), 60]
                )
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                elapsed = time.time() - t0
                sleep_t = frame_delay - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def heatmap_feed(session_id: str):
    safe_session = "".join(c for c in session_id if c.isalnum() or c == "-")
    frame_path = f"/dev/shm/poultry_heatmap_{safe_session}.jpg"

    def generate():
        while True:
            if os.path.exists(frame_path):
                try:
                    with open(frame_path, "rb") as f:
                        jpeg_bytes = f.read()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
                except Exception:
                    pass
            time.sleep(0.2)  # ~5 FPS

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/api/status")
def api_status():
    """
    Returns current system status, uptime, and camera info.

    Example response:
    {
      "camera_id": "CAM_01",
      "status": "running",
      "video_source": "demo1.mp4",
      "started_at": "2026-06-06T12:00:00Z",
      "uptime": "0h 5m 23s",
      "last_updated": "2026-06-06T12:05:23Z"
    }
    """
    return jsonify(state_store.get_status()), 200


@app.route("/api/metrics")
def api_metrics():
    """
    Returns a full analytics snapshot including zone occupancy.

    Example response:
    {
      "camera_id": "CAM_01",
      "status": "running",
      "last_updated": "...",
      "metrics": {
        "total_chickens": 118,
        "total_humans": 2,
        "active": 90,
        "inactive": 28,
        "avg_activity_score": 74,
        "most_visited_zone": "Feed Zone",
        "size_uniformity": "Good",
        "alert_count": 3
      },
      "zone_occupancy": {
        "Feed Zone": 45,
        "Water Zone": 20,
        "Rest Zone": 53,
        "Entry Zone": 0
      }
    }
    """
    return jsonify(state_store.get_metrics()), 200


@app.route("/api/alerts")
def api_alerts():
    """
    Returns the last N alerts.
    Query param: ?limit=50 (default 50)

    Example response:
    [
      { "message": "Chicken #5 inactive for 15 seconds.", "timestamp": "..." },
      ...
    ]
    """
    limit = request.args.get("limit", 50, type=int)
    return jsonify(state_store.get_alerts(limit=limit)), 200


@app.route("/api/events")
def api_events():
    """
    Returns the last N events (zone entry/exit, inactivity, crowd, etc.).
    Query param: ?limit=100 (default 100)

    Example response:
    [
      {
        "event_type": "zone_entry",
        "track_id": 12,
        "zone": "Feed Zone",
        "timestamp": "...",
        "details": "Chicken #12 entered Feed Zone"
      },
      ...
    ]
    """
    limit = request.args.get("limit", 100, type=int)
    return jsonify(state_store.get_events(limit=limit)), 200


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def api_analyze():
    """
    POST /api/analyze
    Returns the current metrics snapshot immediately.
    Future: accept an image upload for one-shot analysis.

    Response is identical to GET /api/metrics.
    """
    if request.method == "OPTIONS":
        return "", 204
    snapshot = state_store.get_metrics()
    snapshot["events"] = state_store.get_events(limit=20)
    snapshot["alerts"] = state_store.get_alerts(limit=10)
    return jsonify(snapshot), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[SERVER] Starting unified MJPEG + REST API server on port 8502...")
    app.run(host="0.0.0.0", port=8502, threaded=True, debug=False)
