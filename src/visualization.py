import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Accumulated heatmap layer
        self.heatmap_layer = None
        
    def draw_zones(self, frame, zone_manager):
        # Draw dynamic zones as transparent overlays
        overlay = frame.copy()
        
        for pot in zone_manager.pots["feed"]:
            px, py = pot["center"]
            cv2.circle(overlay, (px, py), zone_manager.ZONE_RADIUS, (255, 0, 255), -1)
            cv2.putText(overlay, "Feed Zone", (px - 40, py - zone_manager.ZONE_RADIUS - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            if pot.get("box"):
                x1, y1, x2, y2 = pot["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, "Feeding Pot", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        for pot in zone_manager.pots["water"]:
            px, py = pot["center"]
            cv2.circle(overlay, (px, py), zone_manager.ZONE_RADIUS, (255, 255, 0), -1)
            cv2.putText(overlay, "Water Zone", (px - 40, py - zone_manager.ZONE_RADIUS - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if pot.get("box"):
                x1, y1, x2, y2 = pot["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Water Pot", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame
        
    def draw_tracking(self, frame, detections, analytics, custom_tags=None):
        if custom_tags is None:
            custom_tags = {}
            
        # Initialize heatmap if needed
        if self.heatmap_layer is None or self.heatmap_layer.shape[:2] != frame.shape[:2]:
            self.heatmap_layer = np.zeros(frame.shape[:2], dtype=np.float32)
            
        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            conf = det["conf"]
            class_id = det.get("class_id", 14)
            class_name = det.get("class_name", "bird")
            is_human = (class_name == "person")
            is_pot = ("pot" in class_name)
            is_chicken = not (is_human or is_pot)
            
            # Update heatmap
            # Add a Gaussian blob
            cv2.circle(self.heatmap_layer, (cx, cy), 15, 1.0, -1)
            
            # Draw box
            status = "Active"
            if is_human:
                color = (0, 165, 255) # BGR Orange for human
                status = "Human"
            elif is_pot:
                if "water" in class_name:
                    color = (255, 255, 0) # BGR Cyan for water pot
                else:
                    color = (255, 0, 255) # BGR Magenta for feeding pot
                status = "Pot"
            else:
                color = (0, 255, 0)
                if track_id in analytics.stats and analytics.stats[track_id]["is_inactive"]:
                    status = "Inactive"
                    color = (0, 0, 255) # Red for inactive
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if is_chicken:
                display_class = "Chicken"
            else:
                display_class = class_name.capitalize()
                
            display_name = custom_tags.get(str(track_id), custom_tags.get(track_id, f"{display_class}:{track_id}"))
            label = f"{display_name} {conf:.2f} ({status})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trail
            if track_id in analytics.history:
                history = analytics.history[track_id]
                pts = [ (int(p["x"]), int(p["y"])) for p in history[-30:] ] # last 30 points
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], color, 2)
                    
        return frame
        
    def get_heatmap_overlay(self, frame):
        if self.heatmap_layer is None:
            return frame
            
        # Normalize and apply colormap
        norm_heat = cv2.normalize(self.heatmap_layer, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Optional: blur it to make it smoother
        norm_heat = cv2.GaussianBlur(norm_heat, (15, 15), 0)
        heatmap_color = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        
        # Overlay only where heat > 0
        mask = norm_heat > 10
        overlay = frame.copy()
        # Add a dimension to mask to broadcast across color channels
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # Where mask is true, apply heatmap
        np.putmask(overlay, mask_3d, cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0))
        
        return overlay
