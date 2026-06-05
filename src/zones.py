import numpy as np

class ZoneManager:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        
        # Define relative zones (for demo purposes)
        # Feed Zone: Left 25%
        # Water Zone: Right 25%
        # Rest Zone: Center Bottom (30% to 70% width, 60% to 100% height)
        self.zones = {
            "Feed Zone": {
                "x_min": 0, "x_max": int(self.width * 0.25),
                "y_min": 0, "y_max": self.height,
                "color": (0, 255, 0) # Green
            },
            "Water Zone": {
                "x_min": int(self.width * 0.75), "x_max": self.width,
                "y_min": 0, "y_max": self.height,
                "color": (255, 0, 0) # Blue
            },
            "Rest Zone": {
                "x_min": int(self.width * 0.3), "x_max": int(self.width * 0.7),
                "y_min": int(self.height * 0.6), "y_max": self.height,
                "color": (0, 165, 255) # Orange
            }
        }
        
    def get_zone(self, x, y):
        for name, bounds in self.zones.items():
            if (bounds["x_min"] <= x <= bounds["x_max"] and 
                bounds["y_min"] <= y <= bounds["y_max"]):
                return name
        return None
