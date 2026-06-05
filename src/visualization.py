import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Accumulated heatmap layer
        self.heatmap_layer = None
        
    def draw_zones(self, frame, zone_manager):
        overlay = frame.copy()
        for name, bounds in zone_manager.zones.items():
            color = bounds["color"]
            cv2.rectangle(overlay, (bounds["x_min"], bounds["y_min"]), 
                          (bounds["x_max"], bounds["y_max"]), color, -1)
            cv2.putText(frame, name, (bounds["x_min"] + 10, bounds["y_min"] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # alpha blend
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame
        
    def draw_tracking(self, frame, detections, analytics):
        # Initialize heatmap if needed
        if self.heatmap_layer is None or self.heatmap_layer.shape[:2] != frame.shape[:2]:
            self.heatmap_layer = np.zeros(frame.shape[:2], dtype=np.float32)
            
        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            conf = det["conf"]
            
            # Update heatmap
            # Add a Gaussian blob
            cv2.circle(self.heatmap_layer, (cx, cy), 15, 1.0, -1)
            
            # Draw box
            status = "Active"
            color = (0, 255, 0)
            if track_id in analytics.stats and analytics.stats[track_id]["is_inactive"]:
                status = "Inactive"
                color = (0, 0, 255) # Red for inactive
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {conf:.2f} ({status})"
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
