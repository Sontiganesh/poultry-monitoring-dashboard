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
                box = det.get("box", None)
                pot_data = {"center": (cx, cy), "box": box}
                if "water" in cname:
                    self.pots["water"].append(pot_data)
                else:
                    self.pots["feed"].append(pot_data)

    def get_zone(self, x, y):
        # Find nearest pot
        min_dist_feed = float('inf')
        for pot in self.pots["feed"]:
            px, py = pot["center"]
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < min_dist_feed: min_dist_feed = dist
            
        min_dist_water = float('inf')
        for pot in self.pots["water"]:
            px, py = pot["center"]
            dist = np.sqrt((x-px)**2 + (y-py)**2)
            if dist < min_dist_water: min_dist_water = dist
            
        if min_dist_feed < self.ZONE_RADIUS and min_dist_feed <= min_dist_water:
            return "Feed Zone"
        if min_dist_water < self.ZONE_RADIUS:
            return "Water Zone"
            
        # If no pots are near, or standard YOLO model is used (no pots detected), default to Rest Zone
        return "Rest Zone"
