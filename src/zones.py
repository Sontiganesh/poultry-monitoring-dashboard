import numpy as np


class ZoneManager:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.pots = {"feed": [], "water": []}
        self.ZONE_RADIUS = 150  # pixels radius around a pot to count as a zone

        # Entry Zone: a configurable rectangular strip (default: bottom 15% of frame)
        self.entry_zone = self._default_entry_zone(frame_width, frame_height)

    def _default_entry_zone(self, w, h):
        """Default entry zone is the bottom 15% of the frame — where workers typically walk in."""
        return {
            "x1": 0,
            "y1": int(h * 0.85),
            "x2": w,
            "y2": h,
        }

    def set_entry_zone(self, x1: int, y1: int, x2: int, y2: int):
        """Override the entry zone rectangle."""
        self.entry_zone = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

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

    def _in_entry_zone(self, x: int, y: int) -> bool:
        ez = self.entry_zone
        return ez["x1"] <= x <= ez["x2"] and ez["y1"] <= y <= ez["y2"]

    def get_zone(self, x, y) -> str:
        # Entry zone takes priority
        if self._in_entry_zone(x, y):
            return "Entry Zone"

        # Find nearest feed pot
        min_dist_feed = float("inf")
        for pot in self.pots["feed"]:
            px, py = pot["center"]
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist_feed:
                min_dist_feed = dist

        # Find nearest water pot
        min_dist_water = float("inf")
        for pot in self.pots["water"]:
            px, py = pot["center"]
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist_water:
                min_dist_water = dist

        if min_dist_feed < self.ZONE_RADIUS and min_dist_feed <= min_dist_water:
            return "Feed Zone"
        if min_dist_water < self.ZONE_RADIUS:
            return "Water Zone"

        # Default
        return "Rest Zone"

    def get_zone_occupancy(self, detections: list) -> dict:
        """
        Calculate real-time zone occupancy from the latest frame's detections.
        Returns a dict: { zone_name: count }
        """
        occupancy = {
            "Feed Zone": 0,
            "Water Zone": 0,
            "Rest Zone": 0,
            "Entry Zone": 0,
        }
        for det in detections:
            cx, cy = det["center"]
            zone = self.get_zone(cx, cy)
            if zone in occupancy:
                occupancy[zone] += 1
        return occupancy
