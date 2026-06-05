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

def _generate(frame_path):
    """MJPEG generator that reads the latest frame from disk."""
    while True:
        try:
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
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

@app.route("/video_feed/<session_id>")
def video_feed(session_id):
    # Prevent path traversal attacks
    safe_session = "".join(c for c in session_id if c.isalnum() or c == '-')
    frame_path = f"/tmp/poultry_frame_{safe_session}.jpg"
    
    return Response(
        _generate(frame_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    print("[MJPEG] Starting stream server on port 8502...")
    app.run(host="0.0.0.0", port=8502, threaded=True, debug=False)
