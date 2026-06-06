import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add AsyncTracker class
async_code = '''
import threading

class AsyncTracker:
    def __init__(self, base_tracker):
        self.tracker = base_tracker
        self.last_detections = []
        self.raw_detections = []
        self.lock = threading.Lock()
        self.running = False
        
    def process_frame(self, frame, conf_threshold, classes):
        if not self.running:
            self.running = True
            frame_copy = frame.copy()
            def worker():
                dets = self.tracker.process_frame(frame_copy, conf_threshold=conf_threshold, classes=classes)
                with self.lock:
                    self.raw_detections = dets
                self.running = False
            threading.Thread(target=worker, daemon=True).start()
        
        with self.lock:
            return self.raw_detections

if st.session_state.processing'''

content = content.replace('if st.session_state.processing', async_code, 1)

# 2. Wrap tracker
content = content.replace('tracker = load_tracker(selected_model_path, selected_tracker)', 'base_tracker = load_tracker(selected_model_path, selected_tracker)\n    tracker = AsyncTracker(base_tracker)')

# 3. Remove frame_skip variable
content = re.sub(r'\s*frame_skip = 10[^\n]*\n', '\n', content)

# 4. Remove if frame_count % frame_skip block and fix indentation
# Find the exact text to replace
old_loop = """            if frame_count % frame_skip == 0 or frame_count == 1:
                # Process Heavy AI Models
                raw_detections = tracker.process_frame(frame, conf_threshold=conf_threshold, classes=target_classes)
                
                # Filter out standard YOLO detections that perfectly overlap with known pots
                detections = []
                
                # Filter out obvious inanimate objects that YOLO misclassifies the red pots as.
                ignored_classes = ["bowl", "cup", "vase", "potted plant", "fire hydrant", "bottle", "wine glass", "traffic light", "chair"]
                
                for d in raw_detections:
                    if d.get("class_name", "") in ignored_classes:
                        continue
                        
                    detections.append(d)
                
                # Update analytics ONLY when AI runs
                for det in detections:
                    track_id = det["track_id"]
                    cx, cy = det["center"]
                    x1, y1, x2, y2 = det["box"]
                    box_area = (x2 - x1) * (y2 - y1)
                    class_id = det.get("class_id", 14)
                    class_name = det.get("class_name", "bird")
                    zone = zone_manager.get_zone(cx, cy)
                    analytics.update(track_id, cx, cy, zone, class_id=class_id, class_name=class_name, box_area=box_area)
                    
                # Run flock-level analytics (huddling) at 1 FPS to save massive CPU time
                if frame_count % target_fps == 0:
                    analytics.analyze_flock()
                
                # Cache detections for the next skipped frames
                last_detections = detections
            else:
                # Use cached detections to keep the video looking smooth without running YOLO
                detections = last_detections"""

new_loop = """            # Process Heavy AI Models (Asynchronously!)
            raw_detections = tracker.process_frame(frame, conf_threshold=conf_threshold, classes=target_classes)
            
            # Filter out standard YOLO detections that perfectly overlap with known pots
            detections = []
            
            # Filter out obvious inanimate objects that YOLO misclassifies the red pots as.
            ignored_classes = ["bowl", "cup", "vase", "potted plant", "fire hydrant", "bottle", "wine glass", "traffic light", "chair"]
            
            for d in raw_detections:
                if d.get("class_name", "") in ignored_classes:
                    continue
                    
                detections.append(d)
            
            # Update analytics
            for det in detections:
                track_id = det["track_id"]
                cx, cy = det["center"]
                x1, y1, x2, y2 = det["box"]
                box_area = (x2 - x1) * (y2 - y1)
                class_id = det.get("class_id", 14)
                class_name = det.get("class_name", "bird")
                zone = zone_manager.get_zone(cx, cy)
                analytics.update(track_id, cx, cy, zone, class_id=class_id, class_name=class_name, box_area=box_area)
                
            # Run flock-level analytics (huddling) at 1 FPS to save massive CPU time
            if frame_count % target_fps == 0:
                analytics.analyze_flock()"""

content = content.replace(old_loop, new_loop)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
