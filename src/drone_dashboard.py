import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import streamlit.components.v1 as components

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PLATFORM_ROOT / "results" / "drone_traffic"
_video_with_analytics = components.declare_component(
    "drone_video_with_analytics_v2",
    path=str(Path(__file__).with_name("drone_video_component")),
)


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


def _streamlit_media_url(video):
    """Register a local video with Streamlit and return its browser URL."""
    try:
        import streamlit.runtime as runtime
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        context = get_script_run_ctx()
        if context is None or not runtime.exists():
            return None
        coordinates = f"drone-traffic:{video.resolve()}"
        return runtime.get_instance().media_file_mgr.add(str(video), "video/mp4", coordinates)
    except (ImportError, RuntimeError, AttributeError):
        return None


def _render_saved_analytics(summary, video=None, component_key=None):
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
    video_id = summary.name.removesuffix("_traffic_summary.json")
    detections_csv = OUTPUTS / f"{video_id}_traffic_detections.csv"
    tracks_csv = OUTPUTS / f"{video_id}_direction_tracks.csv"
    has_frame_data = detections_csv.is_file() or tracks_csv.is_file()
    if is_detection_summary or has_frame_data:
        vehicle_count = int(snapshot.get("observation_count", vehicle_count) or 0)
        st.subheader("Per-frame traffic")
        st.caption("Inspect the vehicle count and class mix for each video frame. Motion and direction are estimated separately from tracked detections; direction is relative to the screen.")
        frame_count = int(snapshot.get("frames_processed", payload.get("frames_processed", 0)) or 0)
        if has_frame_data:
            try:
                import pandas as pd
                if detections_csv.is_file():
                    frame_data = pd.read_csv(detections_csv, usecols=["frame", "timestamp", "class_name"])
                else:
                    # Older runs saved ByteTrack rows but not the detector CSV.
                    # Each track row is one vehicle observed in that frame.
                    frame_data = pd.read_csv(tracks_csv, usecols=["frame", "timestamp", "track_id", "class_name"])
                    if not frame_data.empty:
                        frame_data["frame"] = frame_data["frame"].astype(int) - int(frame_data["frame"].min())
                if not frame_data.empty:
                    frame_data["frame"] = frame_data["frame"].astype(int)
                    if not detections_csv.is_file():
                        frame_count = int(frame_data["frame"].nunique())
                        sampled_times = frame_data.drop_duplicates("frame").sort_values("frame")["timestamp"].astype(float)
                        sample_step = sampled_times.diff().median() if len(sampled_times) > 1 else 0
                        if sample_step and sample_step > 0:
                            fps = round(1.0 / float(sample_step), 2)
                        else:
                            fps = float(payload.get("source_fps", 30) or 30)
                    else:
                        fps = float(payload.get("source_fps", 30) or 30)
                    per_frame = frame_data.groupby(["frame", "class_name"]).size().unstack(fill_value=0)
                    max_frame = max(frame_count - 1, int(per_frame.index.max()))
                    per_frame = per_frame.reindex(range(max_frame + 1), fill_value=0)
                    per_frame.index.name = "Frame"
                    per_frame["Total vehicles in frame"] = per_frame.sum(axis=1)
                    if fps <= 2:
                        st.caption("Test1 analytics are sampled about once per second; the cards follow the playing video.")
                    else:
                        st.caption("Counts are detections visible in that individual frame. A vehicle can appear again in the next frame.")
                    tracks_by_frame = {}
                    tracked = None
                    if tracks_csv.is_file():
                        try:
                            track_columns = set(pd.read_csv(tracks_csv, nrows=0).columns)
                            usecols = [name for name in ("frame", "track_id", "direction", "class_name", "camera_compensation_reliable") if name in track_columns]
                            tracked = pd.read_csv(tracks_csv, usecols=usecols)
                            if not detections_csv.is_file() and not tracked.empty:
                                tracked["frame"] = tracked["frame"].astype(int) - int(tracked["frame"].min())
                            span_by_track = tracked.groupby("track_id")["frame"].agg(lambda values: values.max() - values.min())
                            median_track_seconds = float(span_by_track.median()) / fps if not span_by_track.empty else 0.0
                            has_reliability = "camera_compensation_reliable" in tracked.columns
                            comp_values = tracked["camera_compensation_reliable"].astype(str).str.lower().isin({"true", "1"}) if has_reliability else None
                            comp_pct = 100.0 * float(comp_values.mean()) if comp_values is not None and len(comp_values) else None
                            for frame, current in tracked.groupby("frame", sort=False):
                                reliable = current["camera_compensation_reliable"].astype(str).str.lower().isin({"true", "1"}) if has_reliability else pd.Series(True, index=current.index)
                                visible = current.loc[reliable]
                                directions = visible["direction"].fillna("unknown").str.lower().value_counts()
                                tracks_by_frame[int(frame)] = {
                                    "tracked": int(visible["track_id"].nunique()),
                                    "moving": sum(int(directions.get(name, 0)) for name in ("left", "right", "up", "down")),
                                    "stationary": int(directions.get("stationary", 0)),
                                    "uncertain": int(directions.get("unknown", 0)),
                                    "directions": {name: int(directions.get(name, 0)) for name in ("left", "right", "up", "down") if directions.get(name, 0)},
                                    "reliable": int(reliable.sum()),
                                    "visible": int(len(reliable)),
                                }
                        except (ImportError, OSError, ValueError, KeyError) as exc:
                            st.warning(f"Could not load the saved motion estimates: {exc}")
                            tracked = None

                    per_frame = per_frame.reindex(range(max_frame + 1), fill_value=0)
                    classes = [name for name in per_frame.columns if name != "Total vehicles in frame"]
                    frame_payload = []
                    for frame in range(max_frame + 1):
                        class_counts = {str(name): int(per_frame.iloc[frame][name]) for name in classes if per_frame.iloc[frame][name]}
                        motion = tracks_by_frame.get(frame, {})
                        frame_payload.append({
                            "count": int(per_frame.iloc[frame]["Total vehicles in frame"]),
                            "classes": class_counts,
                            "tracked": motion.get("tracked"),
                            "moving": motion.get("moving"),
                            "stationary": motion.get("stationary"),
                            "uncertain": motion.get("uncertain"),
                            "directions": motion.get("directions", {}),
                        })

                    video_url = _streamlit_media_url(video) if video else None
                    if video_url:
                        _video_with_analytics(
                            video_url=video_url,
                            frames=frame_payload,
                            fps=fps,
                            duration=max_frame / fps,
                            key=component_key or f"drone-video-{video_id}",
                            loop=True,
                        )
                    else:
                        selected_frame = st.slider(
                            "Inspect frame",
                            min_value=0,
                            max_value=max_frame,
                            value=min(2500, max_frame),
                            key=f"frame-inspector-{video_id}",
                        )
                        st.video(str(video), loop=True) if video else None
                    if video_url:
                        st.caption("The player shows saved detections and motion estimates for its current playback second.")
                    else:
                        class_row = per_frame.loc[selected_frame].drop(labels=["Total vehicles in frame"])
                        detected_classes = [(name, int(value)) for name, value in class_row.items() if value]
                        frame_cols = st.columns(min(6, max(1, len(detected_classes) + 1)))
                        frame_cols[0].metric("Vehicles in this frame", int(per_frame.loc[selected_frame, "Total vehicles in frame"]))
                        for col, (name, value) in zip(frame_cols[1:], detected_classes):
                            col.metric(name.replace("_", " ").title(), value)

                    if tracks_csv.is_file() and not video_url:
                        try:
                            tracked = pd.read_csv(
                                tracks_csv,
                                usecols=["frame", "track_id", "direction", "camera_compensation_reliable"],
                            )
                            span_by_track = tracked.groupby("track_id")["frame"].agg(lambda values: values.max() - values.min())
                            median_track_seconds = float(span_by_track.median()) / fps if not span_by_track.empty else 0.0
                            comp_values = tracked["camera_compensation_reliable"].astype(str).str.lower().isin({"true", "1"})
                            comp_pct = 100.0 * float(comp_values.mean()) if len(comp_values) else 0.0
                            current = tracked.loc[tracked["frame"].astype(int) == selected_frame]
                            reliable = current["camera_compensation_reliable"].astype(str).str.lower().isin({"true", "1"})
                            current = current.loc[reliable]
                            directions = current["direction"].fillna("unknown").str.lower().value_counts()
                            stationary = int(directions.get("stationary", 0))
                            unknown = int(directions.get("unknown", 0))
                            moving = sum(int(directions.get(name, 0)) for name in ("left", "right", "up", "down"))
                            st.caption(f"Motion snapshot at {selected_frame / fps:.1f} seconds · {current['track_id'].nunique()} tracked vehicles · camera compensation reliable for {int(reliable.sum())}/{len(reliable)} visible tracks")
                            motion_cols = st.columns(4)
                            motion_cols[0].metric("Moving", moving)
                            motion_cols[1].metric("Stationary in frame", stationary)
                            motion_cols[2].metric("Parked", "Cannot tell")
                            motion_cols[3].metric("Uncertain", unknown)
                            directional = [f"{label.title()} {int(directions.get(label, 0))}" for label in ("left", "right", "up", "down") if directions.get(label, 0)]
                            st.caption("Screen direction: " + (" · ".join(directional) if directional else "no confident direction in this frame"))
                            reliability_note = f"Camera compensation was marked reliable on {comp_pct:.1f}% of tracked observations. " if comp_pct is not None else "Camera compensation reliability was not saved for this run. "
                            st.info(f"{reliability_note}Track IDs last a median {median_track_seconds:.2f} seconds, so counts are only a frame snapshot—not unique trips. ‘Stationary’ means low movement; a queue stop cannot be confirmed as parked.")
                        except (ImportError, OSError, ValueError, KeyError) as exc:
                            st.warning(f"Could not load the saved motion estimates: {exc}")
                    elif not tracks_csv.is_file() and not video_url:
                        st.info("Moving/stationary and direction snapshots need the saved ByteTrack motion results.")

                    if not video_url:
                        st.markdown("**Segment blockage outlook**")
                        st.caption("No 30-second or 1-minute forecast yet. The video changes camera views and the old road sections drifted off the roadway. Calibrated segment boundaries and sustained occupancy history are needed to estimate a time to blockage.")

                    with st.expander("Whole-video detection totals"):
                        st.caption("A sum of detections across the entire clip; repeated sightings of the same vehicle are included.")
                        totals = st.columns(4)
                        totals[0].metric("Detections across clip", vehicle_count)
                        totals[1].metric("Frames analyzed", frame_count)
                        totals[2].metric("Frames with detections", snapshot.get("frames_with_detections", "—"))
                        totals[3].metric("Average detections per frame", f"{vehicle_count / frame_count:.1f}" if frame_count else "—")
                        if counts:
                            class_cols = st.columns(min(5, max(1, len(counts))))
                            for col, (name, value) in zip(class_cols, sorted(counts.items())):
                                col.metric(f"{name.replace('_', ' ').title()} detections", value)
            except (ImportError, OSError, ValueError, KeyError) as exc:
                st.warning(f"Could not build per-frame analytics from the saved detections and tracks: {exc}")
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
    _render_saved_analytics(summary, video=video, component_key=f"drone-embed-{video_id}")


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
            index=(
                list(saved_videos).index("himachal_kullu_bypass")
                if "himachal_kullu_bypass" in saved_videos
                else list(saved_videos).index("test1") if "test1" in saved_videos else 0
            ),
            key="drone_saved_video_choice",
        )
        saved_video, saved_summary = _result_paths(saved_id)
        _render_saved_analytics(saved_summary, video=saved_video, component_key=f"drone-video-{saved_id}")
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
