import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PLATFORM_ROOT / "src" / "modules" / "drone_traffic"
SCRIPT = PACKAGE / "runner.py"
MODEL = Path(os.environ.get("DRONE_TRAFFIC_WEIGHTS", PLATFORM_ROOT / "models" / "yolo" / "drone_vehicles.pt"))
OUTPUTS = PLATFORM_ROOT / "results" / "drone_traffic"

CONFIGS = {
    "Configured camera (two-direction road layout)": PLATFORM_ROOT / "cfg" / "drone_traffic" / "traffic.yaml",
}


def render_drone_traffic():
    st.title("🚁 Drone Traffic Analytics")
    st.caption("Vehicle tracking, direction, speed, and road-segment congestion")
    st.info(
        "Road segments use the Manesar camera calibration (two directions on one road). "
        "Use it only for matching footage; other views need road polygons calibrated first."
    )

    if not SCRIPT.is_file() or not MODEL.is_file():
        st.error("The drone traffic engine or vehicle model is missing.")
        st.caption("Set DRONE_TRAFFIC_WEIGHTS or place drone_vehicles.pt under models/yolo/.")
        return

    config_label = st.selectbox("Road layout", list(CONFIGS))
    uploaded = st.file_uploader("Choose a drone video", type=["mp4", "mov", "avi", "mkv"])
    device = st.selectbox("Processing device", ["Automatic", "CPU"], index=0)
    max_frames = st.number_input("Optional frame limit (0 = full video)", min_value=0, value=0, step=300)

    if st.button("Analyze video", type="primary", disabled=uploaded is None):
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="drone_traffic_") as tmp:
            safe_name = Path(uploaded.name).name
            source = Path(tmp) / safe_name
            source.write_bytes(uploaded.getvalue())
            args = [
                sys.executable, str(SCRIPT),
                "--source", str(source),
                "--weights", str(MODEL),
                "--traffic-config", str(CONFIGS[config_label]),
                "--direction-config", str(PLATFORM_ROOT / "cfg" / "drone_traffic" / "direction.yaml"),
                "--tracker-config", str(PLATFORM_ROOT / "cfg" / "drone_traffic" / "bytetrack.yaml"),
                "--out-dir", str(OUTPUTS),
                "--device", "cpu" if device == "CPU" else "auto",
            ]
            if max_frames:
                args += ["--max-frames", str(int(max_frames))]
            with st.spinner("Analyzing video. Longer clips can take several minutes on CPU."):
                result = subprocess.run(args, cwd=PLATFORM_ROOT, capture_output=True, text=True)
            if result.returncode:
                st.error("Video analysis failed.")
                st.code((result.stderr or result.stdout)[-12000:])
                return
            st.session_state["drone_traffic_last_stem"] = Path(safe_name).stem
            st.success("Analysis complete. Results are saved in the platform results folder.")
            if result.stdout.strip():
                st.caption(result.stdout.strip().splitlines()[-1])

    stem = st.session_state.get("drone_traffic_last_stem")
    if not stem:
        return
    video = OUTPUTS / f"{stem}_traffic.mp4"
    summary = OUTPUTS / f"{stem}_traffic_summary.json"
    intervals = OUTPUTS / f"{stem}_traffic_summary.csv"
    if video.is_file():
        st.subheader("Annotated video")
        st.video(str(video))
    if summary.is_file():
        st.subheader("Analysis summary")
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            st.json(payload)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read the summary: {exc}")
        st.download_button("Download JSON summary", summary.read_bytes(), file_name=summary.name, mime="application/json")
    if intervals.is_file():
        st.download_button("Download segment data (CSV)", intervals.read_bytes(), file_name=intervals.name, mime="text/csv")
