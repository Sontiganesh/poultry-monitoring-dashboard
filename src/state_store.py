"""
State Store — Multi-process shared memory between the Streamlit processing loop 
and the Flask REST API server via atomic JSON file serialization.

The Streamlit process (single writer) updates the state and flushes it atomically to disk.
The Flask process (reader) loads the latest state from disk on every API call.
"""
import threading
import datetime
import time
import json
import os
import copy
import tempfile

_lock = threading.Lock()
STATE_FILE = "/tmp/poultry_state.json"

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


def _load_state():
    """Load state from disk if it exists, to sync between Streamlit and Flask."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                _state.update(data)
        except Exception:
            pass


def _save_state():
    """Atomically save the state to disk so the other process can read it safely."""
    try:
        # Write to a temporary file on the same filesystem to guarantee atomic rename
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE), prefix="state_", suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(_state, f)
        
        # Atomically replace the target file (avoids partial reads by Flask)
        os.replace(temp_path, STATE_FILE)
    except Exception:
        pass


def set_camera_id(camera_id: str):
    with _lock:
        _load_state()
        _state["camera_id"] = camera_id
        _save_state()


def set_status(status: str, video_source: str = None):
    with _lock:
        _load_state()
        _state["status"] = status
        if status == "running" and _state["started_at"] is None:
            _state["started_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        if video_source is not None:
            _state["video_source"] = video_source
        _save_state()


def update_metrics(metrics: dict, zone_occupancy: dict = None):
    with _lock:
        _load_state()
        _state["metrics"].update(metrics)
        if zone_occupancy:
            _state["zone_occupancy"].update(zone_occupancy)
        _state["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        _save_state()


def add_event(event: dict):
    with _lock:
        _load_state()
        _state["events"].append(event)
        if len(_state["events"]) > 200:
            _state["events"] = _state["events"][-200:]
        _state["events_since_last_ping"].append(event)
        _save_state()


def add_alert(alert: str):
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    with _lock:
        _load_state()
        _state["alerts"].append({"message": alert, "timestamp": ts})
        if len(_state["alerts"]) > 100:
            _state["alerts"] = _state["alerts"][-100:]
        _save_state()


def get_snapshot() -> dict:
    with _lock:
        _load_state()
        return copy.deepcopy(_state)


def get_metrics() -> dict:
    with _lock:
        _load_state()
        return {
            "camera_id": _state.get("camera_id", "CAM_01"),
            "status": _state.get("status", "idle"),
            "last_updated": _state.get("last_updated"),
            "metrics": copy.deepcopy(_state.get("metrics", {})),
            "zone_occupancy": copy.deepcopy(_state.get("zone_occupancy", {})),
        }


def get_alerts(limit: int = 50) -> list:
    with _lock:
        _load_state()
        return list(_state.get("alerts", [])[-limit:])


def get_events(limit: int = 100) -> list:
    with _lock:
        _load_state()
        return list(_state.get("events", [])[-limit:])


def flush_events_since_ping() -> list:
    with _lock:
        _load_state()
        events = list(_state.get("events_since_last_ping", []))
        _state["events_since_last_ping"] = []
        _save_state()
        return events


def get_status() -> dict:
    with _lock:
        _load_state()
        uptime = None
        if _state.get("started_at"):
            try:
                started = datetime.datetime.fromisoformat(_state["started_at"].replace("Z", "+00:00"))
                uptime_secs = int((datetime.datetime.now(datetime.timezone.utc) - started).total_seconds())
                uptime = f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m {uptime_secs % 60}s"
            except Exception:
                pass
        return {
            "camera_id": _state.get("camera_id", "CAM_01"),
            "status": _state.get("status", "idle"),
            "video_source": _state.get("video_source"),
            "started_at": _state.get("started_at"),
            "uptime": uptime,
            "last_updated": _state.get("last_updated"),
        }
