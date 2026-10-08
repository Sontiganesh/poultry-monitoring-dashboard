"""Small read-only API for the standalone drone traffic web dashboard."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory


ROOT = Path(os.environ.get("DRONE_RESULTS_DIR", Path(__file__).resolve().parents[1] / "results" / "drone_traffic")).resolve()
VIDEO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")
app = Flask(__name__, static_folder="static", static_url_path="")


def _paths(video_id: str) -> tuple[Path, Path]:
    if not VIDEO_ID_RE.fullmatch(video_id):
        abort(404)
    video = (ROOT / f"{video_id}_traffic.mp4").resolve()
    summary = (ROOT / f"{video_id}_traffic_summary.json").resolve()
    if video.parent != ROOT or summary.parent != ROOT or not video.is_file() or not summary.is_file():
        abort(404)
    return video, summary


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/videos")
def videos():
    output = []
    for summary in sorted(ROOT.glob("*_traffic_summary.json")):
        video_id = summary.name.removesuffix("_traffic_summary.json")
        try:
            video, _ = _paths(video_id)
            data = json.loads(summary.read_text(encoding="utf-8"))
            details = data.get("full_inference_summary") or {}
            final = data.get("final_frame_analytics") or {}
            source = details.get("video") or data.get("source") or video_id.replace("_", " ").title()
            source_path = Path(str(source))
            if str(source).startswith("/tmp/"):
                source = source_path.stem
            elif str(source).lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                source = source_path.stem
            output.append({
                "id": video_id,
                "title": source,
                "video_url": f"/media/{video_id}",
                "duration_sec": details.get("duration_sec"),
                "frames_processed": data.get("frames_processed") or final.get("frames_processed") or details.get("frames_processed"),
                "vehicle_observations": final.get("observation_count") or details.get("detection_observations_total") or final.get("vehicle_count_final_frame"),
            })
        except (OSError, ValueError):
            continue
    return jsonify({"videos": output})


@app.get("/api/videos/<video_id>")
def video_summary(video_id):
    _, summary = _paths(video_id)
    data = json.loads(summary.read_text(encoding="utf-8"))
    cache = ROOT / f"{video_id}_playback_frames.json"
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            signature = []
            for suffix in ("_traffic_detections.csv", "_direction_tracks.csv"):
                source = ROOT / f"{video_id}{suffix}"
                if source.is_file():
                    stat = source.stat()
                    signature.append([source.name, stat.st_size, stat.st_mtime_ns])
            if cached.get("source_signature") == signature:
                data["playback_frames"] = cached.get("frames", [])
                data["playback_fps"] = cached.get("fps", 30)
        except (OSError, ValueError):
            pass
    response = jsonify(data)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/media/<video_id>")
def media(video_id):
    video, _ = _paths(video_id)
    response = send_from_directory(video.parent, video.name, mimetype="video/mp4", conditional=True, max_age=3600)
    response.headers["Accept-Ranges"] = "bytes"
    return response


@app.get("/health")
def health():
    return jsonify({"status": "ok", "results_dir": str(ROOT)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DRONE_WEB_PORT", "8503")), threaded=True)
