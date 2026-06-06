"""
State Store — Thread-safe singleton that acts as the shared memory
between the Streamlit processing loop and the Flask REST API server.

Both processes write/read through this module so they always share
the exact same real-time snapshot without any race conditions.
"""
import threading
import datetime
import time
import json
import os

_lock = threading.Lock()
SHM_FILE = "/dev/shm/poultry_state.json"

_state = {
    "camera_id": "CAM_01",
    "status": "idle",          # idle | running | stopped
    "started_at": None,
    "last_updated": None,
    "video_source": None,

    # Live metrics snapshot (written every frame by app.py)
    "metrics": {
        "total_chickens": 0,
        "total_humans": 0,
        "active": 0,
        "inactive": 0,
        "avg_activity_score": 0,
        "most_visited_zone": "None",
        "size_uniformity": "N/A",
        "alert_count": 0,
    },

    # Zone occupancy — real-time counts per zone
    "zone_occupancy": {
        "Feed Zone": 0,
        "Water Zone": 0,
        "Rest Zone": 0,
        "Entry Zone": 0,
    },

    # Rolling event log — last 200 events
    "events": [],

    # Rolling alert log — last 100 alerts
    "alerts": [],

    # Events collected since the last webhook ping (cleared after each dispatch)
    "events_since_last_ping": [],
}


def _save_data():
    try:
        with open(SHM_FILE, "w", encoding="utf-8") as f:
            json.dump(_state, f)
    except Exception:
        pass


def _load_data():
    try:
        if os.path.exists(SHM_FILE):
            with open(SHM_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _state.update(data)
    except Exception:
        pass


def set_camera_id(camera_id: str):
    with _lock:
        _state["camera_id"] = camera_id
        _save_data()


def set_status(status: str, video_source: str = None):
    with _lock:
        _state["status"] = status
        if status == "running" and _state["started_at"] is None:
            _state["started_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        if video_source is not None:
            _state["video_source"] = video_source
        _save_data()


def update_metrics(metrics: dict, zone_occupancy: dict = None):
    with _lock:
        _state["metrics"].update(metrics)
        if zone_occupancy:
            _state["zone_occupancy"].update(zone_occupancy)
        _state["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        _save_data()


def add_event(event: dict):
    """Add a single event dict to the rolling log and the since-last-ping buffer."""
    with _lock:
        _state["events"].append(event)
        if len(_state["events"]) > 200:
            _state["events"] = _state["events"][-200:]
        _state["events_since_last_ping"].append(event)
        _save_data()


def add_alert(alert: str):
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    with _lock:
        _state["alerts"].append({"message": alert, "timestamp": ts})
        if len(_state["alerts"]) > 100:
            _state["alerts"] = _state["alerts"][-100:]
        _save_data()


def get_snapshot() -> dict:
    """Return a deep-copy snapshot of the entire state (safe to serialise to JSON)."""
    with _lock:
        _load_data()
        import copy
        return copy.deepcopy(_state)


def get_metrics() -> dict:
    with _lock:
        _load_data()
        import copy
        return {
            "camera_id": _state["camera_id"],
            "status": _state["status"],
            "last_updated": _state["last_updated"],
            "metrics": copy.deepcopy(_state["metrics"]),
            "zone_occupancy": copy.deepcopy(_state["zone_occupancy"]),
        }


def get_alerts(limit: int = 50) -> list:
    with _lock:
        _load_data()
        return list(_state["alerts"][-limit:])


def get_events(limit: int = 100) -> list:
    with _lock:
        _load_data()
        return list(_state["events"][-limit:])


def flush_events_since_ping() -> list:
    """Return and clear the events accumulated since last webhook dispatch."""
    with _lock:
        events = list(_state["events_since_last_ping"])
        _state["events_since_last_ping"] = []
        _save_data()
        return events


def get_status() -> dict:
    with _lock:
        _load_data()
        uptime = None
        if _state["started_at"]:
            started = datetime.datetime.fromisoformat(_state["started_at"].replace("Z", "+00:00"))
            uptime_secs = int((datetime.datetime.now(datetime.timezone.utc) - started).total_seconds())
            uptime = f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m {uptime_secs % 60}s"
        return {
            "camera_id": _state["camera_id"],
            "status": _state["status"],
            "video_source": _state["video_source"],
            "started_at": _state["started_at"],
            "uptime": uptime,
            "last_updated": _state["last_updated"],
        }
