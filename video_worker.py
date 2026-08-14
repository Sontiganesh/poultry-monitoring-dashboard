"""
Strict Sequential Live Video & AI Worker
=========================================
Runs live YOLO inference strictly in sequential frame order.
Guarantees forward-only smooth video playback with zero box jumping or backward jitter.
"""

import cv2
import time
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import PoultryTracker
from src.visualization import Visualizer
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.event_engine import EventEngine
from src.webhook import WebhookDispatcher
from src import state_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEOS = {
    "demo1": {"path": os.path.join(BASE_DIR, "videos", "demo1.mp4"), "is_poultry": True,  "camera": "CAM_01"},
    "demo2": {"path": os.path.join(BASE_DIR, "videos", "demo2.mp4"), "is_poultry": True,  "camera": "CAM_02"},
    "demo3": {"path": os.path.join(BASE_DIR, "videos", "demo3.mp4"), "is_poultry": False, "camera": "CAM_03"},
    "demo4": {"path": os.path.join(BASE_DIR, "videos", "demo4.mp4"), "is_poultry": False, "camera": "CAM_04"},
}

IGNORED_CLASSES = {
    "bowl", "cup", "vase", "potted plant", "fire hydrant",
    "bottle", "wine glass", "traffic light", "chair",
}

RESOLUTION   = (480, 270)
JPEG_QUALITY = 60
TRACK_EVERY_N = 2   # Run ByteTrack every 2 frames for smooth tracking & optimal CPU usage


class LiveStreamWorker:
    def __init__(self, video_name: str, config: dict):
        self.video_name = video_name
        self.video_path = config["path"]
        self.is_poultry = config["is_poultry"]
        self.camera_id = config["camera"]
        
        self.frame_path = f"/dev/shm/poultry_frame_{video_name}.jpg"
        self.tmp_path   = self.frame_path + ".tmp"
        
        self.visualizer = Visualizer()
        self.webhook_dispatcher = WebhookDispatcher(camera_id=self.camera_id)
        self.last_webhook_time = time.time() - 30.0

    def run(self):
        """Sequential Processing Loop: 100% in-sync forward playback."""
        print(f"[WORKER:{self.video_name}] Loading YOLO tracker...")
        tracker = PoultryTracker(model_path="yolov8n.pt", tracker_algo="bytetrack")
        print(f"[WORKER:{self.video_name}] Sequential live pipeline active.")

        while True:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                time.sleep(3)
                continue

            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(2)
                continue

            first_frame = cv2.resize(first_frame, RESOLUTION)
            h, w = first_frame.shape[:2]
            zone_manager = ZoneManager(w, h, is_poultry=self.is_poultry)
            analytics    = PoultryAnalytics()
            analytics.is_poultry = self.is_poultry
            event_engine = EventEngine()

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            current_dets = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video -> restart loop cleanly

                frame = cv2.resize(frame, RESOLUTION)
                frame_count += 1

                # Live ByteTrack AI inference on exact frame in sequence
                if frame_count % TRACK_EVERY_N == 0 or frame_count == 1:
                    raw_dets = tracker.process_frame(frame, is_poultry=self.is_poultry)
                    current_dets = [
                        d for d in raw_dets
                        if d.get("class_name", "") not in IGNORED_CLASSES
                    ]
                    
                    # Analytics & events update
                    for det in current_dets:
                        cx, cy = det["center"]
                        x1, y1, x2, y2 = det["box"]
                        analytics.update(
                            det["track_id"], cx, cy,
                            zone_manager.get_zone(cx, cy),
                            class_id=det.get("class_id", 14),
                            class_name=det.get("class_name", "bird"),
                            box_area=(x2 - x1) * (y2 - y1),
                        )
                    
                    new_events = event_engine.process(current_dets, analytics, zone_manager, is_poultry=self.is_poultry)
                    for evt in new_events:
                        state_store.add_event(evt)

                # Render overlays synchronously on current frame
                frame_disp = self.visualizer.draw_zones(frame, zone_manager)
                frame_disp = self.visualizer.draw_tracking(frame_disp, current_dets, analytics)
                _, buf = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                # Atomic write to shared memory
                try:
                    with open(self.tmp_path, "wb") as f:
                        f.write(buf.tobytes())
                    os.rename(self.tmp_path, self.frame_path)
                except Exception:
                    pass

                # Webhook dispatch every 30 seconds
                current_time = time.time()
                if current_time - self.last_webhook_time >= 30.0:
                    webhook_url = os.environ.get("WEBHOOK_URL", "https://webhook.site/7e5a79b4-1918-41a5-b090-50d74ed643c9")
                    if webhook_url:
                        stats = analytics.get_summary_stats()
                        zone_occ = zone_manager.get_zone_occupancy(current_dets)
                        events_since = state_store.flush_events_since_ping()
                        alerts_list = state_store.get_alerts(limit=10)

                        payload = self.webhook_dispatcher.build_payload(
                            analytics_stats=stats,
                            zone_occupancy=zone_occ,
                            events_since_last_ping=events_since,
                            alerts=alerts_list,
                            is_poultry=self.is_poultry,
                        )
                        self.webhook_dispatcher.dispatch(webhook_url, payload)
                        self.last_webhook_time = current_time

            cap.release()
            time.sleep(0.05)


def main():
    threads = []
    for video_name, config in VIDEOS.items():
        worker = LiveStreamWorker(video_name, config)
        t = threading.Thread(target=worker.run, daemon=True, name=f"seq-worker-{video_name}")
        t.start()
        threads.append(t)
        time.sleep(2)

    print("\n[SEQUENTIAL WORKER] All live video streams running in strict forward order.\n")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[SEQUENTIAL WORKER] Exiting...")


if __name__ == "__main__":
    main()
