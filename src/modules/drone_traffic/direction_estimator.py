"""Trajectory-based image and road-relative direction estimation."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Observation:
    frame: int
    timestamp: float
    x: float
    y: float
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str


def unit(vec):
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def inside(point, polygon):
    if not polygon:
        return False
    import cv2
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32),
                                (float(point[0]), float(point[1])), False) >= 0


class DirectionEstimator:
    """Keeps bounded track histories and emits smoothed directional labels."""
    def __init__(self, config):
        self.cfg = config
        self.window = int(config.get("observation_window", 12))
        self.min_len = int(config.get("minimum_trajectory_length", 5))
        self.min_disp = float(config.get("minimum_displacement_px", 18))
        self.min_conf = float(config.get("minimum_detection_confidence", 0.15))
        self.max_gap = int(config.get("max_occlusion_frames", 45))
        self.smooth_n = int(config.get("smoothing_observations", 5))
        self.alpha = float(config.get("direction_smoothing", 0.35))
        self.tracks = defaultdict(lambda: deque(maxlen=max(self.window * 3, self.min_len + 2)))
        self.states = {}
        self.last_seen = {}
        self.previous_zones = {}
        self.events = []
        self.road_segments = config.get("road_segments", [])
        self.intersections = config.get("intersections", [])

    def observe(self, track_id, obs, point=None, reliable=True):
        p = point if point is not None else (obs.x, obs.y)
        history = self.tracks[int(track_id)]
        history.append((obs, float(p[0]), float(p[1]), bool(reliable)))
        self.last_seen[int(track_id)] = obs.frame
        while history and obs.frame - history[0][0].frame > self.window:
            history.popleft()
        if len(history) < self.min_len:
            return self._empty()
        reliable_history = [h for h in history if h[3]]
        recent = reliable_history[-self.smooth_n:]
        reliable_points = reliable_history
        if len(reliable_points) < min(3, self.min_len):
            return self._empty("unreliable")
        start, end = reliable_points[0], reliable_points[-1]
        dx, dy = end[1] - start[1], end[2] - start[2]
        mag = math.hypot(dx, dy)
        if mag < self.min_disp:
            return {**self._empty(), "direction": "stationary", "movement_type": "unknown",
                    "dx": dx, "dy": dy, "magnitude": mag, "angle_deg": math.degrees(math.atan2(dy, dx))}
        vec = unit((dx, dy))
        previous_vector = self.states.get(int(track_id), {}).get("vector")
        if previous_vector is not None:
            vec = unit(self.alpha * vec + (1.0 - self.alpha) * np.asarray(previous_vector, dtype=float))
        # Stability confidence combines distance above the jitter threshold and
        # agreement among recent per-observation displacements.
        deltas = np.asarray([[h[1], h[2]] for h in recent])
        if len(deltas) > 2:
            steps = np.diff(deltas, axis=0)
            moving = [unit(v) for v in steps if np.linalg.norm(v) > 1]
            coherence = max(0.0, float(np.mean([np.dot(v, vec) for v in moving]))) if moving else 0.0
        else:
            coherence = 0.5
        confidence = min(1.0, mag / max(self.min_disp * 3, 1)) * max(0.0, min(1.0, coherence))
        image_dir = self._image_direction(vec)
        road_id, road_dir = self._road_direction(end[1:3], vec, confidence)
        direction = road_dir or image_dir
        movement = self._intersection_movement(int(track_id), end[1:3], vec, obs.frame)
        if movement in ("left_turn", "right_turn", "u_turn", "straight"):
            direction = movement
        if confidence < float(self.cfg.get("minimum_direction_confidence", 0.25)):
            direction = "unknown"
        prev = self.states.get(int(track_id), {}).get("direction", "unknown")
        # Keep established direction through a short ambiguous interval.
        if direction == "unknown" and prev not in ("unknown", "stationary"):
            direction = prev
        state = {"direction": direction, "movement_type": movement if movement != "unknown" else "unknown",
                 "direction_confidence": round(confidence, 4), "road_segment_id": road_id or "",
                 "dx": round(dx, 2), "dy": round(dy, 2), "magnitude": round(mag, 2),
                 "angle_deg": round(math.degrees(math.atan2(dy, dx)), 2), "vector": vec}
        self.states[int(track_id)] = state
        return state

    @staticmethod
    def _empty(reason=""):
        return {"direction": "unknown", "movement_type": "unknown", "direction_confidence": 0.0,
                "road_segment_id": "", "dx": 0.0, "dy": 0.0, "magnitude": 0.0,
                "angle_deg": 0.0, "vector": np.zeros(2), "unreliable": reason == "unreliable"}

    def _image_direction(self, v):
        deg = math.degrees(math.atan2(v[1], v[0]))
        if abs(deg) >= 135: return "left"
        if abs(deg) <= 45: return "right"
        return "down" if deg > 0 else "up"

    def _road_direction(self, point, vec, confidence):
        for road in self.road_segments:
            if road.get("bounds") and not inside(point, road["bounds"]):
                continue
            if confidence < float(road.get("minimum_confidence", self.min_conf)):
                return road.get("id", ""), None
            candidates = []
            for key in ("direction_a", "direction_b"):
                spec = road.get(key)
                if spec:
                    candidates.append((float(np.dot(vec, unit(spec["vector"]))), spec.get("label", key[-1].upper())))
            if candidates:
                score, label = max(candidates)
                if score >= float(road.get("minimum_similarity", 0.65)):
                    return road.get("id", ""), label
                return road.get("id", ""), "unknown"
        return "", None

    def segment_for_point(self, point):
        """Return configured polygon membership without requiring a mature track."""
        for road in self.road_segments:
            bounds = road.get("bounds")
            if bounds and inside(point, bounds):
                return road.get("id", "")
        return ""

    def _intersection_movement(self, tid, point, vec, frame):
        for inter in self.intersections:
            entry_name = next((z.get("id") for z in inter.get("entry_zones", []) if inside(point, z.get("polygon", []))), None)
            exit_name = next((z.get("id") for z in inter.get("exit_zones", []) if inside(point, z.get("polygon", []))), None)
            prev = self.previous_zones.get(tid, {})
            if entry_name:
                prev["entry"] = entry_name
            if exit_name and prev.get("entry") and exit_name != prev.get("entry"):
                a = next((z.get("vector") for z in inter.get("zone_vectors", []) if z.get("id") == prev["entry"]), None)
                b = next((z.get("vector") for z in inter.get("zone_vectors", []) if z.get("id") == exit_name), None)
                if a and b:
                    ua, ub = unit(a), unit(b)
                    dot = float(np.dot(ua, ub))
                    cross = float(ua[0]*ub[1] - ua[1]*ub[0])
                    # Image coordinates have +Y downward, so positive cross
                    # product corresponds to a clockwise/right turn.
                    result = "u_turn" if dot < -0.75 else ("straight" if dot > 0.75 else ("right_turn" if cross > 0 else "left_turn"))
                    prev["entry"] = exit_name
                    self.previous_zones[tid] = prev
                    self.events.append({"track_id": tid, "frame": frame, "intersection_id": inter.get("id", ""), "movement_type": result})
                    return result
            self.previous_zones[tid] = prev
        return "unknown"

    def prune(self, frame):
        expired = [tid for tid, last in self.last_seen.items() if frame - last > self.max_gap]
        for tid in expired:
            self.last_seen.pop(tid, None); self.tracks.pop(tid, None); self.states.pop(tid, None); self.previous_zones.pop(tid, None)


def point_in_polygon(point, polygon):
    return inside(point, polygon)
