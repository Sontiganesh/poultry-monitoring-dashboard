import json
import threading
import datetime
try:
    import requests
except ImportError:
    # Fallback to urllib if requests isn't installed to avoid breaking the dashboard
    import urllib.request as urllib_request
    import urllib.error as urllib_error
    requests = None

class WebhookDispatcher:
    def __init__(self):
        # We can enforce a timeout so we never hang threads forever
        self.timeout = 5.0
        
    def dispatch(self, url: str, payload: dict):
        """
        Dispatches a payload to the webhook URL in a background thread 
        so it doesn't block the video processing loop.
        """
        if not url or not url.startswith("http"):
            return
            
        # Ensure timestamp is injected if not already present
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
            
        thread = threading.Thread(target=self._send_payload, args=(url, payload), daemon=True)
        thread.start()
        
    def _send_payload(self, url: str, payload: dict):
        try:
            headers = {"Content-Type": "application/json"}
            data = json.dumps(payload).encode('utf-8')
            
            if requests:
                requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            else:
                # Fallback to urllib
                req = urllib_request.Request(url, data=data, headers=headers, method='POST')
                with urllib_request.urlopen(req, timeout=self.timeout) as response:
                    _ = response.read()
        except Exception as e:
            # We silently ignore webhook errors so it doesn't crash the server,
            # but in a production environment we'd log this.
            pass
