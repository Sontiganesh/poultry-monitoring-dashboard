"""
Standalone MJPEG Streaming Server
Runs independently from Streamlit as its own systemd service on port 8502.
Reads frames from a shared memory file that app.py writes to.
"""

from flask import Flask, Response
import time
import os
import sys

app = Flask(__name__)

# Path to the shared frame file (app.py writes JPEGs here, Flask reads them)
FRAME_PATH = "/tmp/poultry_latest_frame.jpg"

def _generate():
    """MJPEG generator that reads the latest frame from disk."""
    while True:
        try:
            if os.path.exists(FRAME_PATH):
                with open(FRAME_PATH, "rb") as f:
                    frame = f.read()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        frame + b"\r\n"
                    )
        except Exception:
            pass
        time.sleep(0.066)  # ~15 FPS

@app.route("/video_feed")
def video_feed():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    print("[MJPEG] Starting stream server on port 8502...")
    app.run(host="0.0.0.0", port=8502, threaded=True, debug=False)
