import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PLATFORM_ROOT / "src" / "modules" / "drone_traffic"
SCRIPT = PACKAGE / "runner.py"
MODEL = Path(os.environ.get("DRONE_TRAFFIC_WEIGHTS", PLATFORM_ROOT / "models" / "yolo" / "drone_vehicles.pt"))
OUTPUTS = PLATFORM_ROOT / "results" / "drone_traffic"

CONFIGS = {
    "Configured camera (two-direction road layout)": PLATFORM_ROOT / "cfg" / "drone_traffic" / "traffic.yaml",
}


def _public_app_base_url():
    """Return a shareable app URL, respecting reverse-proxy headers/config."""
    configured = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured

    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers() or {}
        host = headers.get("X-Forwarded-Host") or headers.get("Host") or ""
        scheme = headers.get("X-Forwarded-Proto") or "http"
        host = host.split(",", 1)[0].strip()
        scheme = scheme.split(",", 1)[0].strip().lower()
        if host and scheme in {"http", "https"}:
            return f"{scheme}://{host}".rstrip("/")
    except Exception:
        pass

    stream_host = os.environ.get("STREAM_HOST", "").strip()
    if stream_host:
        return f"http://{stream_host}:8501"
    return None


def _result_paths(video_id):
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", video_id):
        return None, None
    root = OUTPUTS.resolve()
    video = (root / f"{video_id}_traffic.mp4").resolve()
    summary = (root / f"{video_id}_traffic_summary.json").resolve()
    if video.parent != root or summary.parent != root:
        return None, None
    return video, summary


def _saved_videos():
    """Return completed analyses whose IDs are safe for share links."""
    videos = {}
    for video in OUTPUTS.glob("*_traffic.mp4"):
        video_id = video.name[:-len("_traffic.mp4")]
        paths = _result_paths(video_id)
        if paths[0] == video.resolve() and paths[1].is_file():
            videos[video_id] = video
    return videos


def _render_saved_analytics(summary):
    """Show the summary shape available for either legacy or drone runs."""
    if not summary.is_file():
        st.caption("No analytics summary is saved for this clip.")
        return
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load the analysis summary: {exc}")
        return

    snapshot = payload.get("final_frame_analytics", {})
    if snapshot:
        st.caption("Counts below describe the final processed frame; stationary is a motion estimate, not proof that a vehicle is parked.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vehicles in final frame", snapshot.get("vehicle_count_final_frame", 0))
        c2.metric("Moving", snapshot.get("moving_count", 0))
        c3.metric("Stationary", snapshot.get("stationary_count", 0))
        st.subheader("Direction and segment analytics")
        st.json(snapshot.get("directions", {}))
    else:
        states = payload.get("final_segment_states", {})
        if states:
            st.subheader("Final segment analytics")
            st.caption("Final-frame snapshot. Track IDs can fragment, so these values are not unique trip totals.")
            rows = []
            for segment, state in states.items():
                rows.append({
                    "Segment / direction": segment,
                    "Vehicles": state.get("vehicle_count", 0),
                    "Classes": json.dumps(state.get("class_counts", {}), sort_keys=True),
                    "Moving / stationary": json.dumps(state.get("motion_counts", {}), sort_keys=True),
                    "Speed km/h": state.get("speed_kmh"),
                    "Occupancy %": state.get("occupancy_pct"),
                    "Congestion": state.get("level", "unknown"),
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
        counting = payload.get("counting_lines", {})
        if counting:
            with st.expander("Direction counts and flow"):
                st.json(counting)

    st.download_button(
        "Download full analysis JSON",
        summary.read_bytes(),
        file_name=summary.name,
        mime="application/json",
        key=f"download-{summary.stem}",
    )


def render_drone_embed(video_id):
    """Read-only embed view for one completed drone analysis."""
    st.markdown("""<style>
    [data-testid="stSidebar"] { display: none !important; }
    header, footer { display: none !important; }
    .block-container { padding-top: 0.75rem !important; }
    </style>""", unsafe_allow_html=True)
    video, summary = _result_paths(video_id)
    if video is None or not video.is_file():
        st.error("This drone traffic video is unavailable or the embed link is invalid.")
        return

    st.title("🚁 Drone Traffic Analysis")
    st.video(str(video), loop=True)
    _render_saved_analytics(summary)


def render_drone_traffic(initial_webhook_url=""):
    st.title("🚁 Drone Traffic Analytics")
    st.caption("Vehicle tracking, direction, speed, and road-segment congestion")
    st.info(
        "The bundled road polygons define two directions for one camera view. "
        "Use them only when they match your footage; calibrate other camera views first."
    )

    saved_videos = _saved_videos()
    if saved_videos:
        st.subheader("Saved traffic videos")
        saved_id = st.selectbox(
            "Choose a video to play with its analytics below",
            options=list(saved_videos),
            format_func=lambda video_id: video_id.replace("_", " ").title(),
            index=(list(saved_videos).index("test1") if "test1" in saved_videos else 0),
            key="drone_saved_video_choice",
        )
        saved_video, saved_summary = _result_paths(saved_id)
        st.video(str(saved_video), loop=True)
        _render_saved_analytics(saved_summary)
        app_base = _public_app_base_url()
        if app_base:
            query = urlencode({"mode": "drone_embed", "traffic_video": saved_id})
            st.markdown(f"[Open or embed this looping video]({app_base}/?{query})")
        st.divider()

    if not SCRIPT.is_file() or not MODEL.is_file():
        st.error("The drone traffic engine or vehicle model is missing.")
        st.caption("Set DRONE_TRAFFIC_WEIGHTS or place drone_vehicles.pt under models/yolo/.")
        return

    config_label = st.selectbox("Road layout", list(CONFIGS))
    uploaded = st.file_uploader("Choose a drone video", type=["mp4", "mov", "avi", "mkv"])
    device = st.selectbox("Processing device", ["Automatic", "CPU"], index=0)
    max_frames = st.number_input("Optional frame limit (0 = full video)", min_value=0, value=0, step=300)
    webhook_default = (
        initial_webhook_url
        or os.environ.get("DRONE_TRAFFIC_WEBHOOK_URL", "")
        or os.environ.get("WEBHOOK_URL", "")
    )
    webhook_url = st.text_input(
        "Analytics webhook URL",
        value=webhook_default,
        type="password",
        help="After analysis, the platform POSTs vehicle, direction, motion, and segment analytics to this endpoint.",
    ).strip()
    if not webhook_url:
        st.caption("No webhook configured. Results will still be available in the dashboard and JSON download.")

    if st.button("Analyze video", type="primary", disabled=uploaded is None):
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="drone_traffic_") as tmp:
            original_name = Path(uploaded.name).name
            source_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_name).stem).strip("-")[:80] or "drone-video"
            video_id = f"{source_stem}-{uuid.uuid4().hex[:8]}"
            source = Path(tmp) / f"{video_id}{Path(original_name).suffix.lower()}"
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
            run_env = os.environ.copy()
            run_env["DRONE_SOURCE_FILENAME"] = original_name
            if webhook_url:
                run_env["DRONE_TRAFFIC_WEBHOOK_URL"] = webhook_url
            else:
                run_env.pop("DRONE_TRAFFIC_WEBHOOK_URL", None)
                run_env.pop("WEBHOOK_URL", None)
            with st.spinner("Analyzing video. Longer clips can take several minutes on CPU."):
                result = subprocess.run(args, cwd=PLATFORM_ROOT, capture_output=True, text=True, env=run_env)
            if result.returncode:
                st.error("Video analysis failed.")
                diagnostic = result.stderr or result.stdout
                if webhook_url:
                    diagnostic = diagnostic.replace(webhook_url, "[redacted webhook URL]")
                st.code(diagnostic[-12000:])
                return
            st.session_state["drone_traffic_last_stem"] = video_id
            webhook_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("[DRONE WEBHOOK]")),
                None,
            )
            st.session_state["drone_traffic_last_webhook"] = webhook_line
            st.success("Analysis complete. Results are saved in the platform results folder.")
            if result.stdout.strip():
                st.caption(result.stdout.strip().splitlines()[-1])
            if webhook_line:
                if "sent successfully" in webhook_line:
                    st.success(webhook_line)
                else:
                    st.warning(webhook_line)

    stem = st.session_state.get("drone_traffic_last_stem")
    if not stem:
        return
    video, summary = _result_paths(stem)
    intervals = OUTPUTS / f"{stem}_traffic_summary.csv"
    if video and video.is_file():
        st.subheader("Annotated video")
        st.video(str(video), loop=True)

        app_base = _public_app_base_url()
        if app_base:
            query = urlencode({"mode": "drone_embed", "traffic_video": stem})
            embed_url = f"{app_base}/?{query}"
            st.subheader("Share or embed this analysis")
            st.markdown(f"[Open video and analytics]({embed_url})")
            st.code(embed_url, language="text")
            st.code(
                f'<iframe src="{embed_url}" width="100%" height="760" frameborder="0" allowfullscreen></iframe>',
                language="html",
            )
        else:
            st.info("Set PUBLIC_BASE_URL, or open the app using its public host, to create an embed link.")

    if summary and summary.is_file():
        st.subheader("Analysis summary")
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            snapshot = payload.get("final_frame_analytics", {})
            if snapshot:
                st.caption("Final-frame snapshot; stationary vehicles are motion estimates, not verified parked vehicles.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Vehicles", snapshot.get("vehicle_count_final_frame", 0))
                c2.metric("Moving", snapshot.get("moving_count", 0))
                c3.metric("Stationary", snapshot.get("stationary_count", 0))
                st.json(snapshot.get("directions", {}))
            else:
                st.json(payload)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read the summary: {exc}")
        st.download_button("Download JSON summary", summary.read_bytes(), file_name=summary.name, mime="application/json")
    if intervals.is_file():
        st.download_button("Download segment data (CSV)", intervals.read_bytes(), file_name=intervals.name, mime="text/csv")
