"""
Event Engine — Detects and fires business-level events from raw tracking data.

Events are stateful: each event fires ONCE per transition, not every frame.
Results are pushed into the StateStore for the REST API and webhook to consume.
"""
import time
import datetime
import numpy as np


class EventEngine:
    def __init__(self):
        # track_id -> last known zone
        self._last_zone: dict = {}
        # track_id -> whether inactivity alert already fired
        self._inactivity_fired: dict = {}
        # track_id -> whether erratic alert already fired
        self._erratic_fired: dict = {}
        # track_id -> timestamp first seen (for new-entry events)
        self._first_seen: dict = {}
        # track_id -> whether human-detected alert fired
        self._human_alerted: set = set()
        # Crowd detection cooldown
        self._last_crowd_event: float = 0.0
        self._crowd_cooldown: float = 30.0

    def _now_iso(self) -> str:
        return datetime.datetime.utcnow().isoformat() + "Z"

    def _make_event(self, event_type: str, track_id=None, zone: str = None, details: str = "") -> dict:
        return {
            "event_type": event_type,
            "track_id": track_id,
            "zone": zone,
            "timestamp": self._now_iso(),
            "details": details,
        }

    def process(self, detections: list, analytics, zone_manager, is_poultry=True) -> list:
        """
        Called once per AI inference cycle.

        Args:
            detections: list of detection dicts from tracker
            analytics: PoultryAnalytics instance
            zone_manager: ZoneManager instance
            is_poultry: bool, True if poultry mode, False for restaurant/hotel

        Returns:
            list of new event dicts fired this cycle
        """
        events = []
        current_time = time.time()
        current_track_ids = set()

        for det in detections:
            track_id = det["track_id"]
            cx, cy = det["center"]
            class_name = det.get("class_name", "bird")
            is_human = class_id = det.get("class_id", 14) == 0 or class_name in ("person", "human", "worker")

            current_track_ids.add(track_id)
            current_zone = zone_manager.get_zone(cx, cy)

            # --- 1. Zone Entry / Exit Events ---
            last_zone = self._last_zone.get(track_id)
            subject = "Human" if is_human else ("Chicken" if is_poultry else "Person")
            
            if last_zone is None:
                # First time we see this track — it's a zone entry
                events.append(self._make_event(
                    "zone_entry",
                    track_id=track_id,
                    zone=current_zone,
                    details=f"{subject} #{track_id} entered {current_zone}"
                ))
                self._first_seen[track_id] = current_time
            elif last_zone != current_zone:
                # Zone changed — fire exit + entry
                events.append(self._make_event(
                    "zone_exit",
                    track_id=track_id,
                    zone=last_zone,
                    details=f"{subject} #{track_id} left {last_zone}"
                ))
                events.append(self._make_event(
                    "zone_entry",
                    track_id=track_id,
                    zone=current_zone,
                    details=f"{subject} #{track_id} entered {current_zone}"
                ))

            self._last_zone[track_id] = current_zone

            # --- 2. Human Detected Event (only in poultry mode) ---
            if is_poultry and is_human and track_id not in self._human_alerted:
                events.append(self._make_event(
                    "human_detected",
                    track_id=track_id,
                    zone=current_zone,
                    details=f"Human worker #{track_id} detected in {current_zone}"
                ))
                self._human_alerted.add(track_id)

            # --- 3. Inactivity Threshold Exceeded (only in poultry mode) ---
            if is_poultry and track_id in analytics.stats:
                stats = analytics.stats[track_id]

                if stats.get("is_inactive") and not self._inactivity_fired.get(track_id):
                    events.append(self._make_event(
                        "inactivity_threshold_exceeded",
                        track_id=track_id,
                        zone=current_zone,
                        details=f"Chicken #{track_id} inactive for >15s in {current_zone}"
                    ))
                    self._inactivity_fired[track_id] = True

                elif not stats.get("is_inactive"):
                    # Reset so it can fire again if they stop again
                    self._inactivity_fired[track_id] = False

            # --- 4. Abnormal Movement / Erratic ---
            if track_id in analytics.stats:
                stats = analytics.stats[track_id]
                if stats.get("is_erratic") and not self._erratic_fired.get(track_id):
                    details = (
                        f"Chicken #{track_id} showing erratic high-speed movement — possible panic/stress"
                        if is_poultry else
                        f"Person #{track_id} showing erratic high-speed movement — possible stress/panic"
                    )
                    events.append(self._make_event(
                        "abnormal_movement",
                        track_id=track_id,
                        zone=current_zone,
                        details=details
                    ))
                    self._erratic_fired[track_id] = True
                elif not stats.get("is_erratic"):
                    self._erratic_fired[track_id] = False

        # --- 5. Crowd / Cluster Detection ---
        if current_time - self._last_crowd_event > self._crowd_cooldown:
            if is_poultry:
                positions = [
                    (det["track_id"], det["center"][0], det["center"][1])
                    for det in detections
                    if det.get("class_id", 14) != 0 and det.get("class_name", "bird") not in ("person", "human", "worker")
                ]
            else:
                positions = [
                    (det["track_id"], det["center"][0], det["center"][1])
                    for det in detections
                    if det.get("class_id", 14) == 0 or det.get("class_name", "bird") in ("person", "human", "worker")
                ]

            HUDDLE_DISTANCE = 50
            HUDDLE_MIN = 4
            huddle_count = 0

            for i, (t1, x1, y1) in enumerate(positions):
                close = sum(
                    1 for j, (t2, x2, y2) in enumerate(positions)
                    if i != j and np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < HUDDLE_DISTANCE
                )
                if close >= HUDDLE_MIN - 1:
                    huddle_count += 1

            if huddle_count >= HUDDLE_MIN:
                details = (
                    f"{huddle_count} chickens tightly clustered — check temperature/disease"
                    if is_poultry else
                    f"{huddle_count} people closely gathered."
                )
                events.append(self._make_event(
                    "crowd_cluster_detected",
                    zone=None,
                    details=details
                ))
                self._last_crowd_event = current_time

        # --- 6. Clean up stale track IDs ---
        stale = set(self._last_zone.keys()) - current_track_ids
        for tid in stale:
            last_z = self._last_zone.pop(tid, None)
            self._inactivity_fired.pop(tid, None)
            self._erratic_fired.pop(tid, None)
            self._human_alerted.discard(tid)

        return events
