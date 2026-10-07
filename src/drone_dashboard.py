import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PLATFORM_ROOT / "results" / "drone_traffic"


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
    """Return completed, browser-playable analyses with safe share-link IDs."""
    videos = {}
    for video in OUTPUTS.glob("*_traffic.mp4"):
        video_id = video.name[:-len("_traffic.mp4")]
        paths = _result_paths(video_id)
        if paths[0] != video.resolve() or not paths[1].is_file():
            continue
        capture = None
        try:
            import cv2
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                continue
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = bytes((fourcc >> (8 * index)) & 0xFF for index in range(4)).decode("ascii", "ignore").lower()
            if codec in {"h264", "avc1"}:
                videos[video_id] = video
        except (ImportError, OSError, ValueError):
            continue
        finally:
            if capture is not None:
                capture.release()
    return videos


def _render_saved_analytics(summary):
    """Render saved traffic analytics as readable KPI and segment cards."""
    if not summary.is_file():
        st.caption("No analytics summary is saved for this clip.")
        return
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load the analysis summary: {exc}")
        return

    snapshot = payload.get("final_frame_analytics", {})
    states = payload.get("final_segment_states", {})
    counts = {}
    moving = stationary = 0
    for state in states.values():
        for vehicle_class, count in state.get("class_counts", {}).items():
            counts[vehicle_class] = counts.get(vehicle_class, 0) + int(count or 0)
        motion = state.get("motion_counts", {})
        moving += sum(int(n or 0) for k, n in motion.items() if k not in {"stationary", "unknown"})
        stationary += int(motion.get("stationary", 0) or 0)
    if not states and snapshot:
        counts = snapshot.get("class_counts", snapshot.get("vehicle_class_counts", {}))
        moving = snapshot.get("moving_count", 0)
        stationary = snapshot.get("stationary_count", 0)
    vehicle_count = sum(int(n or 0) for n in counts.values())
    is_track_summary = snapshot.get("count_basis") == "filtered_track_ids"
    is_detection_summary = snapshot.get("count_basis") == "detection_observations"
    if is_detection_summary:
        vehicle_count = int(snapshot.get("observation_count", vehicle_count) or 0)
        video_id = summary.name.removesuffix("_traffic_summary.json")
        st.subheader("Traffic over time")
        st.caption(payload.get("analytics_note", "These are per-frame detections, not unique vehicle totals."))
        kpis = st.columns(4)
        kpis[0].metric("Vehicle detections", vehicle_count)
        kpis[1].metric("Frames analyzed", snapshot.get("frames_processed", "—"))
        kpis[2].metric("Frames with detections", snapshot.get("frames_with_detections", "—"))
        frame_count = int(snapshot.get("frames_processed", 0) or 0)
        kpis[3].metric("Average per frame", f"{vehicle_count / frame_count:.1f}" if frame_count else "—")
        if counts:
            st.markdown("**Vehicle-class detections across all frames**")
            class_cols = st.columns(min(5, max(1, len(counts))))
            for col, (name, value) in zip(class_cols, sorted(counts.items())):
                col.metric(f"{name.replace('_', ' ').title()} detections", value)
        detections_csv = OUTPUTS / f"{video_id}_traffic_detections.csv"
        if detections_csv.is_file():
            try:
                import pandas as pd
                frame_data = pd.read_csv(detections_csv, usecols=["timestamp", "class_name"])
                if not frame_data.empty:
                    frame_data["second"] = frame_data["timestamp"].astype(float).floordiv(1).astype(int)
                    per_second = frame_data.groupby(["second", "class_name"]).size().unstack(fill_value=0)
                    st.markdown("**Vehicles detected in each second**")
                    st.caption("These are detections visible in sampled frames; the same vehicle can be counted again in later seconds.")
                    st.line_chart(per_second, height=260)

                    st.markdown("**Traffic states and segment outlook**")
                    status = st.columns(4)
                    status[0].metric("Moving vehicles", "Needs tracking")
                    status[1].metric("Stationary / parked", "Not verified")
                    status[2].metric("Travel direction", "Not measured")
                    status[3].metric("Blockage forecast", "Not available")
                    st.info("This full-clip run detects vehicles but does not track them across frames. The 45–60 s pilot is not reliable for motion or segments: the camera pans, its road sections drift off the road, and its IDs fragment. A 30 s / 1 min segment warning would be a guess with this map. Calibrate fixed road sections for a stable view, then use tracked movement and occupancy history to issue a forecast.")
            except (ImportError, OSError, ValueError, KeyError) as exc:
                st.warning(f"Could not build the per-second chart from the saved detections: {exc}")
        else:
            st.info("Direction, road-segment, speed, and parked-versus-moving analytics were not measured in this detection-only run.")
        st.download_button(
            "Download full analysis JSON",
            summary.read_bytes(),
            file_name=summary.name,
            mime="application/json",
            key=f"download-{summary.stem}",
        )
        return

    st.subheader("Vehicles tracked in this clip" if is_track_summary else "Traffic at a glance")
    caption = payload.get("analytics_note") or (
        "Track IDs are filtered to reduce short false detections; camera movement and shot changes can still split one vehicle into multiple tracks."
        if is_track_summary else
        "Counts are a snapshot from the final processed frame; stationary is a motion estimate, not a verified parked state."
    )
    st.caption(caption)
    kpis = st.columns(4)
    kpis[0].metric("Vehicle tracks" if is_track_summary else "Vehicles", vehicle_count)
    kpis[1].metric("Moving", "Not measured" if snapshot.get("moving_count") is None else moving)
    kpis[2].metric("Stationary", "Not measured" if snapshot.get("stationary_count") is None else stationary)
    avg_speed_values = [state.get("speed_kmh") for state in states.values() if isinstance(state.get("speed_kmh"), (int, float))]
    kpis[3].metric("Average speed", f"{sum(avg_speed_values) / len(avg_speed_values):.1f} km/h" if avg_speed_values else "—")

    if counts:
        st.markdown("**Vehicle mix**")
        class_cols = st.columns(min(5, max(1, len(counts))))
        for col, (name, value) in zip(class_cols, sorted(counts.items())):
            col.metric(name.replace("_", " ").title(), value)

    if states:
        st.subheader("Road segments by direction")
        st.caption("Each card summarizes the final-frame vehicles detected inside that road segment.")
        directions = {}
        for segment, state in states.items():
            direction = str(state.get("direction") or segment.rsplit(":", 1)[-1] or "Other")
            directions.setdefault(direction, []).append((segment, state))
        for direction, entries in sorted(directions.items()):
            st.markdown(f"#### Direction {direction}")
            cards = st.columns(min(4, len(entries)))
            for index, (segment, state) in enumerate(entries):
                with cards[index % len(cards)]:
                    level = str(state.get("level", "unknown")).replace("_", " ").title()
                    st.markdown(f"**{segment.split(':', 1)[0]}** · {level}")
                    st.metric("Vehicles", state.get("vehicle_count", 0))
                    speed = state.get("speed_kmh")
                    st.caption(f"Speed: {speed:.1f} km/h" if isinstance(speed, (int, float)) else "Speed: unavailable")
                    occupancy = state.get("occupancy_pct")
                    st.caption(f"Road occupancy: {occupancy:.0f}%" if isinstance(occupancy, (int, float)) else "Road occupancy: unavailable")
                    segment_classes = state.get("class_counts", {})
                    if segment_classes:
                        st.caption(" · ".join(f"{label.title()} {value}" for label, value in sorted(segment_classes.items())))
                    motion = state.get("motion_counts", {})
                    if motion:
                        st.caption(" · ".join(f"{label.title()} {value}" for label, value in sorted(motion.items())))

    counting = payload.get("counting_lines", {})
    if counting:
        with st.expander("Traffic flow details"):
            crossings = counting.get("crossings", {})
            per_hour = counting.get("veh_per_h", {})
            for gate, amount in sorted(crossings.items()):
                hourly = per_hour.get(gate)
                rate = f" · {hourly:.0f} vehicles/hour" if isinstance(hourly, (int, float)) else ""
                st.write(f"**{gate}** — {amount} crossings{rate}")

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


def render_drone_traffic():
    st.title("🚁 Saved Drone Traffic Results")
    st.caption("Choose a processed video to replay it and review its saved traffic analytics.")

    saved_videos = _saved_videos()
    if saved_videos:
        st.subheader("Select a video")
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
        detections_csv = OUTPUTS / f"{saved_id}_traffic_detections.csv"
        if detections_csv.is_file():
            st.download_button(
                "Download per-frame detections CSV",
                detections_csv.read_bytes(),
                file_name=detections_csv.name,
                mime="text/csv",
                key=f"detections-{saved_id}",
            )
        app_base = _public_app_base_url()
        if app_base:
            query = urlencode({"mode": "drone_embed", "traffic_video": saved_id})
            st.markdown(f"[Open or embed this looping video]({app_base}/?{query})")
        return

    st.info("No completed drone traffic videos with saved analytics are available yet.")
