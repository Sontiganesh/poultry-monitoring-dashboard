import time
import numpy as np

class PoultryAnalytics:
    def __init__(self):
        # track_id -> list of dicts {"x", "y", "timestamp", "zone"}
        self.history = {}
        # track_id -> dict of stats
        self.stats = {}
        # Alerts history: list of strings
        self.alerts = []
        
        # Timeline data for charts
        self.timeline = {
            "timestamps": [],
            "feed_zone": [],
            "water_zone": [],
            "rest_zone": [],
            "avg_activity": []
        }
        self.last_timeline_update = 0
        
        self.INACTIVITY_THRESHOLD_SECONDS = 15
        self.INACTIVITY_DISTANCE_THRESHOLD = 20 # pixels
        
        # Advanced Behavior Thresholds
        self.SEVERE_LETHARGY_THRESHOLD_SECONDS = 45
        self.ERRATIC_SPEED_THRESHOLD = 150 # px/s
        self.HUDDLE_DISTANCE = 50
        self.HUDDLE_MIN_BIRDS = 4
        self.last_huddle_alert = 0
        
    def update(self, track_id, x, y, zone, class_id=14, box_area=0):
        current_time = time.time()
        
        if track_id not in self.history:
            self.history[track_id] = []
            self.stats[track_id] = {
                "total_distance": 0.0,
                "zone_visits": {"Feed Zone": 0, "Water Zone": 0, "Rest Zone": 0},
                "zone_times": {"Feed Zone": 0.0, "Water Zone": 0.0, "Rest Zone": 0.0},
                "last_zone": None,
                "active_time": 0.0,
                "resting_time": 0.0,
                "is_inactive": False,
                "inactivity_alert_triggered": False,
                "activity_score": 100,
                "is_erratic": False,
                "severe_lethargy_triggered": False,
                "avg_box_area": box_area,
                "class_id": class_id
            }
            
        history = self.history[track_id]
        stats = self.stats[track_id]
        
        if box_area > 0:
            stats["avg_box_area"] = stats["avg_box_area"] * 0.9 + box_area * 0.1
        
        # Calculate distance from last point
        distance = 0.0
        if history:
            last_pt = history[-1]
            dt = current_time - last_pt["timestamp"]
            distance = np.sqrt((x - last_pt["x"])**2 + (y - last_pt["y"])**2)
            stats["total_distance"] += distance
            
            # Update zone stats
            if zone:
                stats["zone_times"][zone] += dt
                if zone != stats["last_zone"]:
                    stats["zone_visits"][zone] += 1
            
            stats["last_zone"] = zone
            
            # Erratic Movement Check
            if dt > 0:
                speed = distance / dt
                if speed > self.ERRATIC_SPEED_THRESHOLD:
                    if not stats["is_erratic"]:
                        stats["is_erratic"] = True
                        self.alerts.append(f"⚡ Chicken #{track_id} erratic movement (Panic/Stress)!")
                else:
                    stats["is_erratic"] = False
            
        history.append({"x": x, "y": y, "timestamp": current_time, "zone": zone})
        
        # Trim history to last 500 points to save memory
        if len(history) > 500:
            history.pop(0)
            
        self._check_inactivity(track_id)
        self._calculate_activity_score(track_id)
        
    def _check_inactivity(self, track_id):
        history = self.history[track_id]
        stats = self.stats[track_id]
        
        # Check if chicken hasn't moved much in the last INACTIVITY_THRESHOLD_SECONDS
        current_time = time.time()
        recent_pts = [p for p in history if current_time - p["timestamp"] <= self.INACTIVITY_THRESHOLD_SECONDS]
        
        if len(recent_pts) > 10: # Need enough data
            xs = [p["x"] for p in recent_pts]
            ys = [p["y"] for p in recent_pts]
            max_dist = np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
            
            if max_dist < self.INACTIVITY_DISTANCE_THRESHOLD:
                if not stats["is_inactive"]:
                    stats["is_inactive"] = True
                    if not stats["inactivity_alert_triggered"]:
                        alert = f"Chicken #{track_id} inactive for {self.INACTIVITY_THRESHOLD_SECONDS} seconds."
                        self.alerts.append(alert)
                        stats["inactivity_alert_triggered"] = True
                        
                # Severe Lethargy Check
                severe_pts = [p for p in history if current_time - p["timestamp"] <= self.SEVERE_LETHARGY_THRESHOLD_SECONDS]
                if len(severe_pts) > 20:
                    s_xs = [p["x"] for p in severe_pts]
                    s_ys = [p["y"] for p in severe_pts]
                    s_max_dist = np.sqrt((max(s_xs) - min(s_xs))**2 + (max(s_ys) - min(s_ys))**2)
                    if s_max_dist < self.INACTIVITY_DISTANCE_THRESHOLD:
                        if not stats["severe_lethargy_triggered"]:
                            self.alerts.append(f"⚠️ SEVERE LETHARGY: Chicken #{track_id} inactive for {self.SEVERE_LETHARGY_THRESHOLD_SECONDS}s. Potential disease!")
                            stats["severe_lethargy_triggered"] = True
            else:
                stats["is_inactive"] = False
                stats["inactivity_alert_triggered"] = False # Reset if they move
                stats["severe_lethargy_triggered"] = False
                
    def analyze_flock(self):
        # Called once per frame to detect group behaviors
        current_time = time.time()
        active_positions = []
        for tid, history in self.history.items():
            if history and (current_time - history[-1]["timestamp"] < 1.0):
                active_positions.append((tid, history[-1]["x"], history[-1]["y"]))
                
        # Check Huddling
        huddle_groups = []
        for i, (t1, x1, y1) in enumerate(active_positions):
            close_birds = 1
            for j, (t2, x2, y2) in enumerate(active_positions):
                if i != j:
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < self.HUDDLE_DISTANCE:
                        close_birds += 1
            if close_birds >= self.HUDDLE_MIN_BIRDS:
                huddle_groups.append(t1)
                
        if len(huddle_groups) >= self.HUDDLE_MIN_BIRDS:
            if (current_time - self.last_huddle_alert > 30): # Rate limit alert to every 30s
                self.alerts.append(f"❄️ HUDDLING DETECTED: {len(huddle_groups)} birds clustered tightly. Check temperature!")
                self.last_huddle_alert = current_time
                
        # Update Timeline (every 1 second)
        if current_time - self.last_timeline_update >= 1.0:
            zone_counts = {"Feed Zone": 0, "Water Zone": 0, "Rest Zone": 0, None: 0}
            active_scores = []
            
            for tid, history in self.history.items():
                if history and (current_time - history[-1]["timestamp"] < 2.0):
                    z = history[-1]["zone"]
                    if z in zone_counts:
                        zone_counts[z] += 1
                    active_scores.append(self.stats[tid]["activity_score"])
                    
            self.timeline["timestamps"].append(time.strftime("%H:%M:%S"))
            self.timeline["feed_zone"].append(zone_counts["Feed Zone"])
            self.timeline["water_zone"].append(zone_counts["Water Zone"])
            self.timeline["rest_zone"].append(zone_counts["Rest Zone"])
            self.timeline["avg_activity"].append(int(np.mean(active_scores)) if active_scores else 0)
            
            # Keep timeline to last 60 points (1 minute) for UI display
            for key in self.timeline:
                if len(self.timeline[key]) > 60:
                    self.timeline[key].pop(0)
                    
            self.last_timeline_update = current_time
                
    def _calculate_activity_score(self, track_id):
        stats = self.stats[track_id]
        # simple heuristic: 100 - (inactive time penalty) + (distance bonus)
        # For demo: just map distance to a score 0-100
        score = min(100, int((stats["total_distance"] / 200) * 10) + (80 if not stats["is_inactive"] else 40))
        stats["activity_score"] = min(100, max(0, score))
        
    def get_summary_stats(self):
        total_chickens = sum(1 for s in self.stats.values() if s.get("class_id", 14) == 14)
        total_humans = sum(1 for s in self.stats.values() if s.get("class_id", 14) == 0)
        
        active = sum(1 for s in self.stats.values() if not s["is_inactive"] and s.get("class_id", 14) == 14)
        inactive = total_chickens - active
        
        chicken_scores = [s["activity_score"] for s in self.stats.values() if s.get("class_id", 14) == 14]
        avg_score = int(np.mean(chicken_scores)) if chicken_scores else 0
        
        # most visited zone (chickens only)
        zone_counts = {"Feed Zone": 0, "Water Zone": 0, "Rest Zone": 0}
        for s in self.stats.values():
            if s.get("class_id", 14) == 14:
                for z, count in s["zone_visits"].items():
                    zone_counts[z] += count
        most_visited = max(zone_counts, key=zone_counts.get) if total_chickens > 0 and sum(zone_counts.values()) > 0 else "None"
        
        # Size Uniformity (chickens only)
        areas = [s.get("avg_box_area", 0) for s in self.stats.values() if s.get("avg_box_area", 0) > 0 and s.get("class_id", 14) == 14]
        uniformity = "N/A"
        if len(areas) > 1:
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv = (std_area / mean_area) * 100 if mean_area > 0 else 0
            if cv < 10: uniformity = "Excellent"
            elif cv < 15: uniformity = "Good"
            else: uniformity = "Poor (Check Feeding)"
        
        return {
            "total_chickens": total_chickens,
            "total_humans": total_humans,
            "active": active,
            "inactive": inactive,
            "avg_activity_score": avg_score,
            "most_visited_zone": most_visited,
            "alert_count": len(self.alerts),
            "size_uniformity": uniformity,
            "timeline": self.timeline
        }
