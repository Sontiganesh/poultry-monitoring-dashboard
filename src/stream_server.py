"""
MJPEG Streaming Server
======================
Runs a lightweight Flask server on port 8502 that streams video frames as
a true MJPEG stream. This bypasses Streamlit's slow WebSocket->React pipeline
entirely. The browser renders MJPEG natively at the OS level - zero JS overhead.
"""

from flask import Flask, Response
import threading
import time

# Shared state between the Streamlit app and the Flask server
_latest_frame = None
_frame_lock = threading.Lock()

app = Flask(__name__)

def push_frame(jpeg_bytes: bytes):
    """Called by app.py to push the latest encoded frame into the stream."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = jpeg_bytes

def _generate():
    """MJPEG generator: yields frames as multipart HTTP response."""
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame + b"\r\n"
        )
        time.sleep(0.066)  # ~15 FPS cap

@app.route("/video_feed")
def video_feed():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/health")
def health():
    return "OK", 200

# Use a process-level singleton via a port-binding check
# This survives Streamlit's module reloads
def start_stream_server(port: int = 8502):
    """Start the Flask MJPEG server. Safe to call multiple times - only binds once."""
    import socket
    
    # Check if something is already listening on this port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            # Port is free - we can start Flask
            s.close()
            server_thread = threading.Thread(
                target=lambda: app.run(
                    host="0.0.0.0",
                    port=port,
                    threaded=True,
                    use_reloader=False,
                    debug=False
                ),
                daemon=True,
                name="mjpeg-server"
            )
            server_thread.start()
            time.sleep(0.5)  # Give Flask time to bind
            print(f"[MJPEG] Streaming server started on port {port}")
        except OSError:
            # Port already in use - Flask is already running, do nothing
            print(f"[MJPEG] Server already running on port {port}")

# Auto-start when imported - the socket check prevents double-binding
start_stream_server(port=8502)
