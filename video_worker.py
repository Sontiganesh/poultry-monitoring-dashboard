"""
Shared Video Processing Worker
================================
Runs ONE YOLO loop per demo video in background threads.
Writes annotated frames to /dev/shm/ so ALL viewers share the same stream.

This eliminates the problem where N iframes = N YOLO loops.
With this worker: always exactly 4 YOLO loops total, regardless of viewer count.

Usage:
    python video_worker.py
    # or via systemd: poultry-worker.service
"""

import threading
import cv2
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import PoultryTracker
from src.visualization import Visualizer
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
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

TARGET_FPS    = 8     # Output frame rate
YOLO_EVERY_N  = 3     # Run YOLO every Nth frame (others reuse last detections)
RESOLUTION    = (480, 270)
JPEG_QUALITY  = 35


def process_video_loop(video_name: str, config: dict):
    """Continuously processes a demo video and writes frames to /dev/shm/."""
    frame_path = f"/dev/shm/poultry_frame_{video_name}.jpg"
    tmp_path   = frame_path + ".tmp"
    video_path = config["path"]
    is_poultry = config["is_poultry"]

    print(f"[WORKER:{video_name}] Starting — model=yolov8n.pt, is_poultry={is_poultry}")

    # Load tracker once — reuse across video loops
    tracker   = PoultryTracker(model_path="yolov8n.pt", tracker_algo="bytetrack")
    visualizer = Visualizer()

    while True:  # outer loop: restart video when it ends
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WORKER:{video_name}] Cannot open video, retrying in 5s...")
            time.sleep(5)
            continue

        # Read first frame to init zone manager
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(5)
            continue

        first_frame = cv2.resize(first_frame, RESOLUTION)
        h, w = first_frame.shape[:2]
        zone_manager = ZoneManager(w, h, is_poultry=is_poultry)
        analytics    = PoultryAnalytics()
        analytics.is_poultry = is_poultry

        source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_jump   = max(1, int(source_fps / TARGET_FPS))  # frames to skip per cycle
        frame_delay  = 1.0 / TARGET_FPS
        frame_count  = 0
        detections   = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"[WORKER:{video_name}] Processing at {TARGET_FPS} FPS (source={source_fps:.1f}, jump={frame_jump})")

        while cap.isOpened():
            loop_start = time.time()

            # Skip ahead to maintain target FPS relative to source
            for _ in range(frame_jump - 1):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                break  # video ended — restart outer loop

            frame = cv2.resize(frame, RESOLUTION)
            frame_count += 1

            # YOLO inference every Nth frame
            if frame_count % YOLO_EVERY_N == 0 or frame_count == 1:
                raw_dets   = tracker.process_frame(frame, is_poultry=is_poultry)
                detections = [
                    d for d in raw_dets
                    if d.get("class_name", "") not in IGNORED_CLASSES
                ]
                # Update analytics
                for det in detections:
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["box"]
                    analytics.update(
                        det["track_id"], cx, cy,
                        zone_manager.get_zone(cx, cy),
                        class_id=det.get("class_id", 14),
                        class_name=det.get("class_name", "bird"),
                        box_area=(x2 - x1) * (y2 - y1),
                    )

            # Draw and encode
            frame_disp = visualizer.draw_zones(frame, zone_manager)
            frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics)
            _, buf = cv2.imencode(".jpg", frame_disp, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            # Atomic write: temp file → rename (prevents partial reads by Flask)
            with open(tmp_path, "wb") as f:
                f.write(buf.tobytes())
            os.rename(tmp_path, frame_path)

            # Frame rate limiter
            elapsed = time.time() - loop_start
            sleep_t = frame_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()
        print(f"[WORKER:{video_name}] Video ended — restarting loop")
        time.sleep(0.5)


def main():
    threads = []
    for i, (video_name, config) in enumerate(VIDEOS.items()):
        t = threading.Thread(
            target=process_video_loop,
            args=(video_name, config),
            daemon=True,
            name=f"worker-{video_name}",
        )
        t.start()
        threads.append(t)
        time.sleep(3)  # stagger startup to avoid simultaneous model loads

    print(f"\n[WORKER] All {len(VIDEOS)} video workers running. Press Ctrl+C to stop.\n")
    try:
        while True:
            # Health check — log if a thread died
            for t in threads:
                if not t.is_alive():
                    print(f"[WORKER] WARNING: thread {t.name} died!")
            time.sleep(30)
    except KeyboardInterrupt:
        print("[WORKER] Shutting down...")


if __name__ == "__main__":
    main()
