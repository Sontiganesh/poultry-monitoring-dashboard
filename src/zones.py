import numpy as np

class ZoneManager:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.pots = {"feed": [], "water": []}
        self.ZONE_RADIUS = 150 # pixels radius around a pot to count as a zone
        
    def update_pots(self, detections):
        self.pots["feed"] = []
        self.pots["water"] = []
        for det in detections:
            cname = det.get("class_name", "")
            if "pot" in cname:
                cx, cy = det["center"]
                if "water" in cname:
                    self.pots["water"].append((cx, cy))
                else:
                    self.pots["feed"].append((cx, cy))

    def get_zone(self, x, y):
        # Find nearest pot
        min_dist_feed = float('inf')
        for px, py in self.pots["feed"]:
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < min_dist_feed: min_dist_feed = dist
            
        min_dist_water = float('inf')
        for px, py in self.pots["water"]:
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < min_dist_water: min_dist_water = dist
            
        if min_dist_feed < self.ZONE_RADIUS and min_dist_feed <= min_dist_water:
            return "Feed Zone"
        if min_dist_water < self.ZONE_RADIUS:
            return "Water Zone"
            
        # If no pots are near, or standard YOLO model is used (no pots detected), default to Rest Zone
        return "Rest Zone"
