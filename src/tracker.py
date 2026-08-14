import cv2
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils import ROOT
import yaml
import os


class PoultryTracker:
    def __init__(self, model_path='yolov8n.pt', tracker_algo='bytetrack'):
        # Track whether this is a YOLOWorld model explicitly
        self.is_world_model = 'world' in model_path.lower()

        # Dynamically generate custom tracker config
        if tracker_algo == 'botsort':
            default_yaml_path = ROOT / 'cfg' / 'trackers' / 'botsort.yaml'
            custom_yaml_path = 'custom_botsort.yaml'
        else:
            default_yaml_path = ROOT / 'cfg' / 'trackers' / 'bytetrack.yaml'
            custom_yaml_path = 'custom_bytetrack.yaml'

        try:
            with open(default_yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            config['track_high_thresh'] = 0.10
            config['track_low_thresh'] = 0.05
            config['new_track_thresh'] = 0.10
            config['track_buffer'] = 150  # Keep IDs alive for 150 frames if lost

            with open(custom_yaml_path, 'w') as f:
                yaml.dump(config, f)
            self.tracker_type = custom_yaml_path
        except Exception as e:
            print(f"Failed to generate custom tracker config, falling back to default: {e}")
            self.tracker_type = f"{tracker_algo}.yaml"

        # Set imgsz to 320 for fast real-time inference on CPU (~35ms per frame)
        self.default_imgsz = 320

        # Initialize model
        if self.is_world_model:
            self.model = YOLOWorld(model_path)
            self.model.set_classes(["white broiler chicken", "chicken", "poultry", "white bird", "person"])
        else:
            self.model = YOLO(model_path)

    def process_frame(self, frame, conf_threshold=0.15, classes=None, is_poultry=True):
        # For standard YOLO, restrict classes based on mode.
        # YOLOWorld uses text prompts so class filtering is not needed.
        effective_classes = classes
        if not self.is_world_model and effective_classes is None:
            if not is_poultry:
                effective_classes = [0]       # person only for restaurant/hotel settings

        results = self.model.track(
            frame,
            persist=True,
            classes=effective_classes,
            conf=conf_threshold,
            tracker=self.tracker_type,
            verbose=False,
            iou=0.5,      # lower IoU so nearby objects don't suppress each other
            imgsz=self.default_imgsz,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0]) if box.cls is not None else 14
            class_name = self.model.names[class_id]
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue

            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            aspect_ratio = h / w

            if is_poultry:
                # --- Aspect Ratio Classifier ---
                # Bird detected but shape is very tall/thin → reclassify as person
                if class_name in ("bird", "chicken", "poultry", "white broiler chicken", "white bird"):
                    if aspect_ratio > 1.8:  # Raised from 1.5 to prevent tall chickens being marked as human
                        class_id = 0
                        class_name = "person"

                # Person detected but shape is very flat/wide → reclassify as bird
                if class_id == 0 and aspect_ratio < 0.7:
                    class_id = 14
                    class_name = "bird"
            else:
                # In hotel/restaurant, we only want humans, no bird conversions
                if class_id != 0:
                    continue

            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })

        return detections
