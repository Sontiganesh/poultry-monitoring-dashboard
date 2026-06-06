import cv2
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils import ROOT
import yaml
import os

class PoultryTracker:
    def __init__(self, model_path='yolov8n.pt', tracker_algo='bytetrack'):
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

        # Initialize YOLOv8 model
        if 'world' in model_path.lower():
            self.model = YOLOWorld(model_path)
            # Use highly descriptive semantic text prompts to achieve fine-tuned accuracy without actually fine-tuning
            self.model.set_classes(["white broiler chicken", "chicken", "poultry", "white bird", "person"])
        else:
            self.model = YOLO(model_path)

        
    def process_frame(self, frame, conf_threshold=0.15, classes=None):
        is_world_model = 'world' in str(getattr(self.model, 'model_name', '')).lower() or hasattr(self.model, 'set_classes')
        
        # For standard YOLO, only detect class 0 (person) and class 14 (bird) — nothing else!
        if not is_world_model and classes is None:
            classes = [0, 14]
        
        results = self.model.track(frame, persist=True, classes=classes, conf=conf_threshold, tracker=self.tracker_type, verbose=False, iou=0.85, imgsz=320)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0]) if box.cls is not None else 14
                class_name = self.model.names[class_id]
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue
                
                # --- Smart Human vs Chicken Classifier ---
                # Use bounding box aspect ratio as a secondary signal.
                # Humans are TALL and THIN (height >> width). Chickens are SHORT and WIDE.
                h = max(1, y2 - y1)
                w = max(1, x2 - x1)
                aspect_ratio = h / w  # > 1.5 strongly indicates a standing human
                
                # If YOLO said "bird" but the box looks very tall and thin, it's almost certainly a human
                if class_name in ("bird", "chicken", "poultry", "white broiler chicken", "white bird") and aspect_ratio > 1.6:
                    class_id = 0
                    class_name = "person"
                
                # If YOLO said "person" but the box is very wide/flat, it could be a chicken — keep as chicken
                if class_id == 0 and aspect_ratio < 0.8:
                    class_id = 14
                    class_name = "bird"
                
                detections.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return detections
