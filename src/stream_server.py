"""
MJPEG Streaming Server
======================
Runs a lightweight Flask server on port 8502 that streams video frames as
a true MJPEG stream. This bypasses Streamlit's slow WebSocket→React pipeline
entirely. The browser renders MJPEG natively at the OS level — zero JS overhead.

This is the same protocol used by real IP security cameras (e.g. Hikvision, Dahua).
"""

from flask import Flask, Response
import threading
import time

# Shared state between the Streamlit app and the Flask server
_latest_frame = None   # raw JPEG bytes (already encoded by OpenCV)
_frame_lock = threading.Lock()

app = Flask(__name__)

def push_frame(jpeg_bytes: bytes):
    """Called by app.py to push the latest encoded frame into the stream."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = jpeg_bytes

def _generate():
    """MJPEG generator: yields frames as multipart HTTP response."""
    boundary = b"--frame"
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame + b"\r\n"
        )
        # ~15 FPS cap to prevent the stream from burning bandwidth
        time.sleep(0.066)

@app.route("/video_feed")
def video_feed():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/health")
def health():
    return "OK", 200

def start_stream_server(port: int = 8502):
    """Start the Flask MJPEG server in a background daemon thread."""
    server_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    return server_thread

# --- AUTO-START AT IMPORT TIME ---
# Use a module-level flag (not st.session_state) so Flask starts exactly ONCE
# the moment this module is imported by app.py — regardless of browser visits.
_server_started = False

def ensure_started(port: int = 8502):
    global _server_started
    if not _server_started:
        _server_started = True
        start_stream_server(port=port)

# Start immediately on import
ensure_started()
