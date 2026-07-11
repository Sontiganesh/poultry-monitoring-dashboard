import json
import threading
import datetime
import time

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
        
        # Delivery metrics tracking
        self.total_success = 0
        self.total_failed = 0
        self.last_sent_time = None
        self.last_response_body = None

    def build_payload(self, analytics_stats: dict, zone_occupancy: dict,
                      events_since_last_ping: list = None, alerts: list = None,
                      is_poultry: bool = True) -> dict:
        """
        Build the structured analytics payload matching the required JSON schema.
        """
        # Summarise recent alert messages
        alerts_src = alerts if alerts is not None else []
        recent_alerts = [a["message"] if isinstance(a, dict) else str(a) for a in alerts_src[-10:]]

        # Resolve zone names dynamically based on mode
        feed_key = "Feed Zone" if is_poultry else "Checkout"
        water_key = "Water Zone" if is_poultry else "Aisle 1"
        rest_key = "Rest Zone" if is_poultry else "Aisle 2"

        payload = {
            "camera_id": self.camera_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "total_count": analytics_stats.get("total_chickens", 0) + analytics_stats.get("total_humans", 0),
            "feed_zone_count": zone_occupancy.get(feed_key, 0),
            "water_zone_count": zone_occupancy.get(water_key, 0),
            "rest_zone_count": zone_occupancy.get(rest_key, 0),
            "activity_score": analytics_stats.get("avg_activity_score", 0) if is_poultry else analytics_stats.get("avg_human_activity_score", 0),
            "alerts": recent_alerts,
            "events": events_since_last_ping if events_since_last_ping is not None else [],
        }
        return payload

    def _log_delivery(self, url: str, payload: dict, status_code: int, status_desc: str, response_body: str):
        """
        Append webhook delivery attempt details to a log file.
        """
        log_line = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "url": url,
            "method": "POST",
            "status_code": status_code,
            "status_description": status_desc,
            "response_body": response_body[:200] if response_body else "", # truncate response body
            "payload": payload
        }
        try:
            with open("webhook_delivery.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_line) + "\n")
        except Exception:
            pass

    def dispatch(self, url: str, payload: dict, sync: bool = False):
        """
        Fire payload to webhook URL.
        """
        print(f"[WEBHOOK DEBUG] dispatch() called for URL: {url}")
        if not url or not url.startswith("http"):
            print(f"[WEBHOOK DEBUG] dispatch failed: invalid URL: {url}")
            return False, None, "Invalid URL"

        if "timestamp" not in payload:
            payload["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        self.last_sent_time = payload["timestamp"]
        self._last_dispatch_time = payload["timestamp"]
        self._dispatch_count += 1

        if sync:
            return self._send_payload(url, payload)
        else:
            print(f"[WEBHOOK DEBUG] Spawning background thread for dispatch...")
            thread = threading.Thread(
                target=self._send_payload,
                args=(url, payload),
                daemon=True,
            )
            thread.start()
            return True, None, "Dispatched in background"

    def _send_payload(self, url: str, payload: dict):
        max_retries = 3
        backoff = 1.0  # seconds
        success = False
        last_error = None
        status_code = None
        response_text = ""

        print(f"[WEBHOOK DEBUG] sending payload to: {url} | total_count: {payload.get('total_count')}")
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload, default=str).encode("utf-8")

        for attempt in range(max_retries):
            try:
                print(f"[WEBHOOK DEBUG] Attempt {attempt+1}/{max_retries}...")
                if _HAS_REQUESTS:
                    resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                    status_code = resp.status_code
                    response_text = resp.text
                else:
                    req = _urllib_request.Request(url, data=data, headers=headers, method="POST")
                    with _urllib_request.urlopen(req, timeout=self.timeout) as response:
                        status_code = response.status
                        response_text = response.read().decode("utf-8", errors="ignore")

                self._last_status_code = status_code
                self.last_response_body = response_text

                if status_code is not None and 200 <= status_code < 300:
                    self.total_success += 1
                    self._log_delivery(url, payload, status_code, "Success", response_text)
                    success = True
                    print(f"[WEBHOOK DEBUG] Dispatch success! Status code: {status_code}")
                    break
                else:
                    self._log_delivery(url, payload, status_code, f"HTTP Error {status_code}", response_text)
                    last_error = f"HTTP Error {status_code}"
                    print(f"[WEBHOOK DEBUG] HTTP error: {status_code} | response: {response_text[:100]}")

            except Exception as e:
                status_code = None
                response_text = str(e)
                self._last_status_code = None
                self.last_response_body = response_text
                self._log_delivery(url, payload, None, f"Connection Failed: {e}", response_text)
                last_error = f"Connection Failed: {e}"
                print(f"[WEBHOOK DEBUG] Connection error: {e}")

            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff retries: 1s, 2s, 4s

        if not success:
            self.total_failed += 1
            print(f"[WEBHOOK DEBUG] Dispatch failed after {max_retries} attempts. Last error: {last_error}")
            return False, status_code, last_error

        return True, status_code, "Success"
