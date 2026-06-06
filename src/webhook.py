import json
import threading
import datetime

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
    _HAS_REQUESTS = False


class WebhookDispatcher:
    def __init__(self, camera_id: str = "CAM_01"):
        self.camera_id = camera_id
        self.timeout = 5.0
        self._last_dispatch_time = None
        self._dispatch_count = 0
        self._last_status_code = None

    def build_payload(self, analytics_stats: dict, zone_occupancy: dict,
                      events_since_last_ping: list, alerts: list) -> dict:
        """
        Build the full structured analytics payload matching the spec.
        """
        # Friendly zone keys (lowercase, no spaces)
        zones = {
            "feeding": zone_occupancy.get("Feed Zone", 0),
            "water": zone_occupancy.get("Water Zone", 0),
            "resting": zone_occupancy.get("Rest Zone", 0),
            "entry": zone_occupancy.get("Entry Zone", 0),
        }

        # Summarise recent alert messages
        recent_alerts = [a["message"] if isinstance(a, dict) else str(a) for a in alerts[-10:]]

        payload = {
            "camera_id": self.camera_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "total_objects": analytics_stats.get("total_chickens", 0) + analytics_stats.get("total_humans", 0),
            "total_chickens": analytics_stats.get("total_chickens", 0),
            "total_humans": analytics_stats.get("total_humans", 0),
            "zones": zones,
            "activity": {
                "active": analytics_stats.get("active", 0),
                "inactive": analytics_stats.get("inactive", 0),
                "avg_score": analytics_stats.get("avg_activity_score", 0),
                "most_visited_zone": analytics_stats.get("most_visited_zone", "None"),
            },
            "events_since_last_ping": events_since_last_ping[-20:],
            "alerts": recent_alerts,
            "size_uniformity": analytics_stats.get("size_uniformity", "N/A"),
        }
        return payload

    def dispatch(self, url: str, payload: dict):
        """
        Fire payload to webhook URL asynchronously (non-blocking).
        """
        if not url or not url.startswith("http"):
            return
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        thread = threading.Thread(
            target=self._send_payload,
            args=(url, payload),
            daemon=True,
        )
        thread.start()
        self._dispatch_count += 1
        self._last_dispatch_time = payload["timestamp"]

    def _send_payload(self, url: str, payload: dict):
        try:
            headers = {"Content-Type": "application/json"}
            data = json.dumps(payload, default=str).encode("utf-8")

            if _HAS_REQUESTS:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                self._last_status_code = resp.status_code
            else:
                req = _urllib_request.Request(url, data=data, headers=headers, method="POST")
                with _urllib_request.urlopen(req, timeout=self.timeout) as response:
                    self._last_status_code = response.status
        except Exception as e:
            # Silent failure — never crash the main loop
            self._last_status_code = None
