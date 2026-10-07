"""Road geometry, image-based occupancy, calibrated speed and stable traffic states."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import math

import cv2
import numpy as np


TRAFFIC_BGR = {"green": (40, 180, 40), "yellow": (0, 210, 255),
               "red": (30, 30, 230), "unknown": (135, 135, 135)}


class RoadSegmentManager:
    """Pixel-space road polygons and direction-specific occupancy raster masks."""
    def __init__(self, config, frame_width, frame_height):
        self.config = config
        ref_w = int(config.get("reference_width", frame_width))
        ref_h = int(config.get("reference_height", frame_height))
        self.sx, self.sy = frame_width / ref_w, frame_height / ref_h
        self.frame_width, self.frame_height = int(frame_width), int(frame_height)
        occ_w = int(config.get("occupancy_resolution_width", 480))
        occ_scale = occ_w / frame_width
        self.width = max(1, occ_w)
        self.height = max(1, int(round(frame_height * occ_scale)))
        self.segments = []
        self.base_roads = []
        # Ground scale. The view is oblique, so one metres-per-pixel constant
        # cannot serve the whole road: the carriageway here is ~48px wide far away
        # and ~248px wide near the camera. Instead the road's real width is held
        # fixed and the local scale is recovered per centerline sample from the
        # measured pixel width, which absorbs both perspective and curvature.
        scale_cfg = config.get("scale", {}) or {}
        self.road_width_m = float(scale_cfg.get("road_width_m", 0) or 0)
        self.lanes_per_direction = max(1, int(scale_cfg.get("lanes_per_direction", 1)))
        # Width fixes the LATERAL scale only. On an oblique view travel-direction
        # distance is foreshortened further, so a separate multiplier carries the
        # longitudinal correction instead of corrupting road_width_m with it -
        # the width stays a real surveyable number, and speed/length/density all
        # read from the corrected longitudinal scale.
        self.longitudinal_correction = float(scale_cfg.get("longitudinal_correction", 1.0) or 1.0)
        self.occ_scale = occ_scale
        for raw in config.get("segments", []):
            p = self._scale_points(raw["polygon"], self.sx, self.sy)
            small = np.round(p.reshape(-1,2).astype(float) * occ_scale).astype(np.int32).reshape(-1,1,2)
            line = self._scale_points(raw.get("centerline", []), self.sx, self.sy)
            base_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(base_mask, [small], 255)
            if not cv2.countNonZero(base_mask):
                continue
            section_count = max(1, int(raw.get("subdivide_into", 1)))
            cut_lines = self._split_polyline(line, section_count)
            labels = self._section_labels(line, section_count, occ_scale)
            base = {**raw, "polygon_px": p, "centerline_px": line,
                    "direction_vector": self._unit(raw.get("travel_vector", [0, 0])),
                    "children": []}
            # Numbering runs along the centerline by default. `reverse_numbering`
            # flips only the LABEL, so an opposing carriageway can read 4..1 while
            # its geometry stays in centerline order - `section_index` must keep
            # that order because _child_for_point() locates a section by comparing
            # it against a centerline fraction.
            reverse_numbering = bool(raw.get("reverse_numbering", False))
            for index in range(section_count):
                mask = cv2.bitwise_and(base_mask, np.where(labels == index, 255, 0).astype(np.uint8))
                area = int(cv2.countNonZero(mask))
                if not area:
                    continue
                label_number = section_count - index if reverse_numbering else index + 1
                child = {**raw,
                    "parent_segment_id": raw["segment_id"],
                    "segment_id": f"{raw['segment_id']}_{label_number:02d}" if section_count > 1 else raw["segment_id"],
                    "section_index": index+1, "section_label": label_number,
                    "section_count": section_count,
                    "polygon_px": p, "centerline_px": cut_lines[index],
                    "mask": mask, "road_pixels": area,
                    "direction_vector": base["direction_vector"]}
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                child["boundary_contours_px"] = [np.round(c.reshape(-1,2).astype(float) / occ_scale).astype(np.int32).reshape(-1,1,2) for c in contours]
                base["children"].append(child)
                self.segments.append(child)
            if base["children"]:
                self._calibrate_road(base, base_mask, raw)
                self.base_roads.append(base)

    def _calibrate_road(self, base, base_mask, raw):
        """Attach local metres-per-pixel samples and a real length to each section.

        Width is measured on the FULL road mask, never a section slice: a slice's
        longitudinal cut edges would read as road boundary and halve the width
        near every section join.
        """
        width_m = float(raw.get("road_width_m", self.road_width_m) or 0)
        lanes = max(1, int(raw.get("lanes", self.lanes_per_direction)))
        base["road_width_m"] = width_m
        base["lanes"] = lanes
        for child in base["children"]:
            child["lanes"] = lanes
            child["road_width_m"] = width_m
            samples = self._scale_samples(base_mask, child["centerline_px"], width_m)
            child["scale_samples"] = samples
            child["length_m"] = self._polyline_length_m(child["centerline_px"], samples)
            child["mean_meters_per_pixel"] = (
                float(np.mean([s[1] for s in samples])) if samples else None)
            # Measured pixel width, kept separately from the metre scale. Anything
            # needing the road's on-screen width (counting gates, for one) must use
            # this: dividing road_width_m by mean_meters_per_pixel would silently
            # fold in longitudinal_correction and come out several times too small.
            widths = self._width_samples(base_mask, child["centerline_px"])
            child["mean_width_px"] = float(np.mean(widths)) if widths else None

    def _scale_samples(self, mask, line, width_m):
        """[(point_in_source_px, metres_per_pixel)] along a section centerline.

        The scale is assumed ISOTROPIC at each sample: one metres-per-pixel used
        both across and along the road. That is exact for a nadir view and
        optimistic for an oblique one, where travel-direction distance is
        foreshortened more than lateral width, so lengths - and therefore speeds -
        come out UNDER-estimated. Road width alone cannot fix this: a metric
        rectification of a plane needs 8 constraints and parallel edges of known
        separation supply too few, so one longitudinal reference distance would be
        required to close it. `scale_diagnostics()` quantifies the residual error
        from observed vehicle lengths instead of leaving it unstated.
        """
        pts = line.reshape(-1, 2).astype(float) if len(line) else np.empty((0, 2))
        if len(pts) < 2 or width_m <= 0:
            return []
        samples = []
        for i, point in enumerate(pts):
            nxt = pts[min(i + 1, len(pts) - 1)]
            prv = pts[max(i - 1, 0)]
            tangent = nxt - prv
            width_px = self._cross_section_width_px(mask, point, tangent)
            if width_px > 0:
                samples.append((point, width_m / width_px * self.longitudinal_correction))
        return samples

    def _width_samples(self, mask, line):
        """Measured road width in source pixels at each centerline sample."""
        pts = line.reshape(-1, 2).astype(float) if len(line) else np.empty((0, 2))
        if len(pts) < 2:
            return []
        widths = []
        for i, point in enumerate(pts):
            tangent = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
            width = self._cross_section_width_px(mask, point, tangent)
            if width > 0:
                widths.append(width)
        return widths

    def _cross_section_width_px(self, mask, point_src, tangent):
        """Road width in SOURCE pixels, by walking the mask perpendicular to travel."""
        norm = float(np.linalg.norm(tangent))
        if norm == 0:
            return 0.0
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float) / norm
        origin = np.asarray(point_src, dtype=float) * self.occ_scale
        spans = 0.0
        for sign in (1.0, -1.0):
            step = 0.0
            while step < max(self.width, self.height):
                probe = origin + normal * sign * (step + 1.0)
                xi, yi = int(round(probe[0])), int(round(probe[1]))
                if not (0 <= xi < self.width and 0 <= yi < self.height) or mask[yi, xi] == 0:
                    break
                step += 1.0
            spans += step
        return spans / self.occ_scale if spans else 0.0

    @staticmethod
    def _polyline_length_m(line, samples):
        """Arc length in metres, each pixel step scaled by its own local m/px."""
        pts = line.reshape(-1, 2).astype(float) if len(line) else np.empty((0, 2))
        if len(pts) < 2 or not samples:
            return None
        scales = {tuple(np.round(p, 3)): mpp for p, mpp in samples}
        fallback = float(np.mean([mpp for _, mpp in samples]))
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            mpp_a = scales.get(tuple(np.round(a, 3)), fallback)
            mpp_b = scales.get(tuple(np.round(b, 3)), fallback)
            total += float(np.linalg.norm(b - a)) * (mpp_a + mpp_b) / 2.0
        return total

    @staticmethod
    def meters_per_pixel(segment, point):
        """Local scale at `point`, from the nearest calibrated centerline sample."""
        samples = (segment or {}).get("scale_samples") or []
        if not samples:
            return None
        target = np.asarray(point, dtype=float)
        return float(min(samples, key=lambda s: np.linalg.norm(s[0] - target))[1])

    @staticmethod
    def _split_polyline(line, count):
        pts = line.reshape(-1,2).astype(float) if len(line) else np.empty((0,2))
        if len(pts) < 2:
            return [line for _ in range(count)]
        steps=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        cumulative=np.concatenate(([0.0],np.cumsum(steps)))
        total=float(cumulative[-1])
        def at(distance):
            j=min(int(np.searchsorted(cumulative,distance,side="right")-1),len(pts)-2)
            frac=(distance-cumulative[j])/max(steps[j],1e-9)
            return pts[j]+frac*(pts[j+1]-pts[j])
        output=[]
        for i in range(count):
            lo,hi=total*i/count,total*(i+1)/count
            values=[at(lo)]+[pts[j] for j,d in enumerate(cumulative) if lo<d<hi]+[at(hi)]
            output.append(np.round(values).astype(np.int32).reshape(-1,1,2))
        return output

    def _section_labels(self,line,count,occ_scale):
        """Partition each raster road-mask pixel by its nearest centerline arc position."""
        if len(line)<2:
            return np.zeros((self.height,self.width),dtype=np.int16)
        pts=line.reshape(-1,2).astype(float)
        pts*=occ_scale
        steps=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        cumulative=np.concatenate(([0.0],np.cumsum(steps)))
        total=max(float(cumulative[-1]),1e-9)
        yy,xx=np.mgrid[0:self.height,0:self.width]
        pixels=np.stack(((xx+.5).reshape(-1),(yy+.5).reshape(-1)),axis=1)
        best_dist=np.full(len(pixels),np.inf)
        best_progress=np.zeros(len(pixels),dtype=float)
        for j,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
            v=b-a; length2=float(np.dot(v,v))
            if length2<1e-9: continue
            t=np.clip(((pixels-a)@v)/length2,0,1)
            projection=a+t[:,None]*v
            dist=np.sum((pixels-projection)**2,axis=1)
            use=dist<best_dist
            best_dist[use]=dist[use]
            best_progress[use]=(cumulative[j]+t[use]*steps[j])/total
        return np.minimum((best_progress*count).astype(np.int16),count-1).reshape(self.height,self.width)

    @staticmethod
    def _scale_points(points, sx, sy):
        return np.asarray([[int(round(float(x) * sx)), int(round(float(y) * sy))]
                           for x, y in points], dtype=np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _unit(v):
        a = np.asarray(v, dtype=float); n = float(np.linalg.norm(a))
        return a / n if n > 1e-9 else np.zeros(2)

    def assign(self, point, motion_vector=None, direction_confidence=0.0):
        matches = []
        for road in self.base_roads:
            if cv2.pointPolygonTest(road["polygon_px"], (float(point[0]), float(point[1])), False) < 0:
                continue
            minimum = float(road.get("minimum_direction_confidence", 0.0))
            road_vec = road["direction_vector"]
            if (road.get("use_direction_gate", False) and motion_vector is not None and
                    np.linalg.norm(road_vec) > 0 and direction_confidence >= minimum):
                similarity = float(np.dot(self._unit(motion_vector), road_vec))
                if similarity < float(road.get("minimum_direction_similarity", -0.1)):
                    continue
            fraction=self._line_fraction(point,road["centerline_px"])
            section_count=int(road.get("subdivide_into", 1))
            target_index=min(int(fraction*section_count),section_count-1)+1
            child=min(road["children"],key=lambda item:abs(item["section_index"]-target_index))
            matches.append((road,child))
        return min(matches,key=lambda pair:sum(s["road_pixels"] for s in pair[0]["children"]))[1] if matches else None

    @staticmethod
    def _line_fraction(point,line):
        pts=line.reshape(-1,2).astype(float) if len(line) else np.empty((0,2))
        if len(pts)<2: return 0.0
        steps=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        cumulative=np.concatenate(([0.0],np.cumsum(steps))); total=max(float(cumulative[-1]),1e-9)
        p=np.asarray(point,dtype=float); best=(float("inf"),0.0)
        for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
            v=b-a; l2=float(np.dot(v,v))
            if l2<1e-9: continue
            t=float(np.clip(np.dot(p-a,v)/l2,0,1)); q=a+t*v
            d=float(np.dot(p-q,p-q))
            if d<best[0]: best=(d,(cumulative[i]+t*steps[i])/total)
        return min(.999999,max(0.0,best[1]))

    def measure_occupancy(self, detections_by_segment):
        """Union clipped vehicle bbox masks to avoid overlap double counting."""
        output = {}
        for seg in self.segments:
            vehicle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            # Use mask dimensions and the segment's stored source-frame points.
            # The runner supplies boxes in stabilized reference pixels; mapping
            # to raster coordinates uses source dimensions saved below.
            sx, sy = self.width / self.frame_width, self.height / self.frame_height
            for bbox in detections_by_segment.get(seg["segment_id"], []):
                x1, y1, x2, y2 = bbox
                poly = np.asarray([[x1*sx,y1*sy],[x2*sx,y1*sy],[x2*sx,y2*sy],[x1*sx,y2*sy]], dtype=np.float32)
                poly = np.round(poly).astype(np.int32).reshape(-1,1,2)
                cv2.fillPoly(vehicle_mask, [poly], 255)
            clipped = cv2.bitwise_and(vehicle_mask, seg["mask"])
            occupied = int(cv2.countNonZero(clipped))
            output[seg["segment_id"]] = 100.0 * occupied / max(1, seg["road_pixels"])
        return output

    def draw(self, frame, inverse_camera_matrix, states, show_boundaries=True,
             show_labels=True, line_thickness=14):
        overlay = frame.copy()
        for road in self.base_roads:
            for child in road["children"]:
                state = states.get(child["segment_id"], {})
                color = TRAFFIC_BGR.get(state.get("level", "unknown"), TRAFFIC_BGR["unknown"])
                stable = child["centerline_px"].astype(np.float32).reshape(-1, 1, 2)
                raw = cv2.perspectiveTransform(stable, inverse_camera_matrix)
                raw_line = np.round(raw).astype(np.int32)
                cv2.polylines(overlay, [raw_line], False, color, int(line_thickness), cv2.LINE_AA)
                if show_labels and len(raw_line):
                    middle = tuple(map(int, raw_line[len(raw_line)//2, 0]))
                    cv2.circle(overlay, middle, 11, (15, 15, 15), -1, cv2.LINE_AA)
                    cv2.circle(overlay, middle, 11, (240, 240, 240), 2, cv2.LINE_AA)
                    cv2.putText(overlay, str(child.get("section_label", child["section_index"])),
                                (middle[0]-4, middle[1]+4), cv2.FONT_HERSHEY_SIMPLEX,
                                .38, (255, 255, 255), 1, cv2.LINE_AA)
            if show_labels and len(road["centerline_px"]):
                start = cv2.perspectiveTransform(
                    road["centerline_px"][:1].astype(np.float32), inverse_camera_matrix)[0, 0]
                cv2.putText(overlay, str(road.get("direction", "")),
                            (int(start[0])+8, int(start[1])-8), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, .72, frame, .28, 0, frame)
        return frame


class CalibratedSpeedEstimator:
    """Returns km/h only when a pixel-to-ground homography is configured."""
    def __init__(self, config):
        matrix = config.get("ground_homography")
        self.matrix = np.asarray(matrix, dtype=float) if matrix else None
        if self.matrix is not None and self.matrix.shape != (3,3):
            raise ValueError("speed.ground_homography must be a 3x3 image-to-ground-meter matrix")
        self.window = float(config.get("speed_window_seconds", 2.0))
        self.history = defaultdict(deque)

    def update(self, track_id, point, timestamp, reliable=True):
        if self.matrix is None or not reliable:
            return None
        ground = cv2.perspectiveTransform(np.asarray([[point]], dtype=np.float32),
                                          self.matrix).reshape(2).astype(float)
        hist = self.history[int(track_id)]
        hist.append((float(timestamp), ground))
        while hist and timestamp - hist[0][0] > self.window:
            hist.popleft()
        if len(hist) < 2:
            return None
        dt = hist[-1][0] - hist[0][0]
        if dt <= 0:
            return None
        return float(np.linalg.norm(hist[-1][1] - hist[0][1]) / dt * 3.6)


def tracking_quality(track_span, track_segments, fps, sections, flow_from_lines=False):
    """Whether track identities survive long enough for flow to mean anything.

    A vehicle should hold one ID from entering the scene to leaving it, crossing
    every subdivision on the way. When most IDs instead live a couple of seconds
    and touch a single section, each real vehicle is counted many times over and
    veh/h becomes meaningless - while instantaneous measures (occupancy, density)
    and short-window ones (speed) stay sound. Reporting that distinction matters
    more than reporting the flow number itself.
    """
    if not track_span:
        return None
    spans = sorted((last - first + 1) for first, last in track_span.values())
    visited = [len(v) for v in track_segments.values()] or [0]
    single = sum(1 for v in visited if v <= 1)
    full_traverse = sum(1 for v in visited if v >= max(1, int(sections)))
    fragmented = single > 0.4 * len(visited)
    return {
        "track_ids": len(track_span),
        "median_track_seconds": round(spans[len(spans)//2] / max(fps, 1e-9), 2),
        "p90_track_seconds": round(spans[int(.9*len(spans))] / max(fps, 1e-9), 2),
        "single_section_track_pct": round(100.0 * single / max(1, len(visited)), 1),
        "full_traverse_track_pct": round(100.0 * full_traverse / max(1, len(visited)), 1),
        "fragmentation_suspected": bool(fragmented),
        # Fragmentation breaks AREA-based counting, not line crossings: a vehicle
        # passes a gate in one frame transition, so only the id holding it at that
        # instant counts. Flow measured at gates therefore stays usable even when
        # identities are badly fragmented.
        "flow_measurement": "counting_lines" if flow_from_lines else "area_entries",
        "flow_reliable": bool(flow_from_lines or not fragmented),
        "note": (
            "Flow is measured at counting lines, which survive ID fragmentation, "
            "so flow_veh_per_h is usable. Any UNIQUE-VEHICLE total is still an "
            "upper bound, because that does depend on identities holding."
            if flow_from_lines and fragmented else
            "Most track IDs cover one section only, so each vehicle is split "
            "across several IDs: treat flow_veh_per_h and any unique-vehicle "
            "total as an upper bound, not a count. Occupancy, density and "
            "speed are per-frame or short-window and remain usable."
            if fragmented else
            "Track lifetimes span multiple sections, so flow counts are usable."),
    }


class RoadScaleSpeedEstimator:
    """km/h from pixel motion scaled by the road-width calibration.

    The alternative to a surveyed `ground_homography`: callers pass the local
    metres-per-pixel from RoadSegmentManager, so perspective is handled per
    sample rather than by one global constant. Speed uses straight-line
    displacement across the window, not summed per-frame steps, because
    per-frame box jitter inflates path length and would overstate speed for
    stationary vehicles.
    """
    def __init__(self, config):
        self.window = float(config.get("speed_window_seconds", 2.0))
        self.history = defaultdict(deque)

    def update(self, track_id, point, timestamp, meters_per_pixel, reliable=True):
        if not reliable or not meters_per_pixel:
            return None
        hist = self.history[int(track_id)]
        hist.append((float(timestamp), np.asarray(point, dtype=float), float(meters_per_pixel)))
        while hist and timestamp - hist[0][0] > self.window:
            hist.popleft()
        if len(hist) < 2:
            return None
        dt = hist[-1][0] - hist[0][0]
        if dt <= 0:
            return None
        mean_scale = float(np.mean([entry[2] for entry in hist]))
        displacement_px = float(np.linalg.norm(hist[-1][1] - hist[0][1]))
        return displacement_px * mean_scale / dt * 3.6

    def prune(self, active_ids):
        for track_id in [t for t in self.history if t not in active_ids]:
            self.history.pop(track_id, None)


class CongestionClassifier:
    """EMA, threshold hysteresis and minimum-state-duration traffic classifier."""
    def __init__(self, config, fps):
        self.cfg = config
        self.fps = float(fps)
        self.window_frames = max(1, int(float(config.get("smoothing_window_seconds", 10)) * fps))
        self.alpha = float(config.get("ema_alpha", 0.15))
        self.states = {}
        self.histories = defaultdict(lambda: deque(maxlen=self.window_frames))
        self.first_valid_timestamp = {}

    def update(self, key, occupancy_pct, speed_kmh, timestamp, confidence=1.0, valid=True):
        self.first_valid_timestamp.setdefault(key, float(timestamp))
        valid = valid and timestamp - self.first_valid_timestamp[key] >= float(self.cfg.get("minimum_observation_duration_seconds", 3.0))
        prev = self.states.get(key, {"level":"unknown", "since":timestamp,
                                      "pending":"unknown", "pending_since":timestamp, "ema":None})
        if not valid:
            result = {**prev, "level":"unknown", "score":None, "ema":prev.get("ema"),
                      "occupancy_score":None, "speed_score":None, "speed_kmh":None,
                      "occupancy_pct":round(float(occupancy_pct),3),
                      "classification_basis":"occupancy_only", "confidence":confidence}
            self.states[key] = result
            return result
        occ_cfg = self.cfg.get("occupancy", {})
        max_occ = max(.01, float(occ_cfg.get("red_occupancy_pct", 30.0)))
        occ_score = min(1.0, max(0.0, float(occupancy_pct) / max_occ))
        speed_score = None
        if speed_kmh is not None:
            free = max(.1, float(self.cfg.get("speed", {}).get("free_flow_kmh", 60)))
            speed_score = min(1.0, max(0.0, 1.0 - speed_kmh / free))
            weights = self.cfg.get("congestion", {}).get("weights", {"occupancy":.6,"speed":.4})
            score = float(weights.get("occupancy", .6))*occ_score + float(weights.get("speed", .4))*speed_score
        else:
            score = occ_score
        self.histories[key].append((float(timestamp), score))
        # The score history already averages a full smoothing window. Applying
        # an EMA to that rolling mean again adds a second window of lag and can
        # leave a busy section green long after its measured occupancy rises.
        rolling = float(np.mean([v for _, v in self.histories[key]]))
        ema = rolling
        levels = self.cfg.get("congestion", {}).get("thresholds", {"green":.35,"yellow":.65})
        green_max, yellow_max = float(levels.get("green", .35)), float(levels.get("yellow", .65))
        candidate = "green" if ema < green_max else ("yellow" if ema < yellow_max else "red")
        current = prev.get("level", "unknown")
        hysteresis = float(self.cfg.get("hysteresis", .04))
        if current == "green" and candidate != "green" and ema < green_max + hysteresis:
            candidate = "green"
        elif current == "red" and candidate != "red" and ema >= yellow_max - hysteresis:
            candidate = "red"
        elif current == "yellow":
            if candidate == "green" and ema >= green_max - hysteresis: candidate = "yellow"
            if candidate == "red" and ema < yellow_max + hysteresis: candidate = "yellow"
        min_duration = float(self.cfg.get("minimum_state_duration_seconds", 3.0))
        if candidate != current:
            if prev.get("pending") != candidate:
                pending, pending_since = candidate, float(timestamp)
            else:
                pending, pending_since = candidate, prev.get("pending_since", timestamp)
            if current == "unknown" or timestamp - pending_since >= min_duration:
                level, since = candidate, float(timestamp)
            else:
                level, since = current, prev.get("since", timestamp)
        else:
            level, since, pending, pending_since = current, prev.get("since", timestamp), candidate, float(timestamp)
        result = {"level":level, "score":round(score,4), "ema":float(ema),
                  "occupancy_score":round(occ_score,4),
                  "speed_score":None if speed_score is None else round(speed_score,4),
                  "speed_kmh":None if speed_kmh is None else round(float(speed_kmh),2),
                  "confidence":round(float(confidence),4), "since":since,
                  "pending":pending, "pending_since":pending_since,
                  "occupancy_pct":round(float(occupancy_pct),3),
                  "classification_basis":"combined" if speed_kmh is not None else "occupancy_only"}
        self.states[key] = result
        return result


class IntervalAccumulator:
    def __init__(self):
        self.values = defaultdict(lambda: {"n":0, "occ":0.0,"score":0.0,"score_n":0,"conf":0.0,
                                           "speed_sum":0.0,"speed_n":0,"latest":{},"entries":0,"exits":0,
                                           "density_sum":0.0,"density_n":0,"motion":Counter(),
                                           "counterflow":0,"flow_sum":0.0,"flow_n":0})

    def add(self, interval, key, metrics, entries=0, exits=0):
        d = self.values[(interval,key)]
        d["n"] += 1; d["occ"] += float(metrics.get("occupancy_pct",0) or 0)
        if metrics.get("score") is not None:
            d["score"] += float(metrics["score"])
            d["score_n"] += 1
        d["conf"] += float(metrics.get("confidence",0) or 0)
        if metrics.get("speed_kmh") is not None:
            d["speed_sum"] += float(metrics["speed_kmh"]); d["speed_n"] += 1
        if metrics.get("density_veh_per_km_lane") is not None:
            d["density_sum"] += float(metrics["density_veh_per_km_lane"]); d["density_n"] += 1
        # Flow is accumulated PER FRAME, never rebuilt from interval means.
        # Density and speed are strongly anti-correlated - a jam is high k, low v -
        # so mean(k)*mean(v) ignores that covariance and overstates flow. Measured
        # against SUMO ground truth, the per-interval product came out +11% while
        # the per-frame mean tracks the truth.
        if metrics.get("flow_veh_per_h") is not None:
            d["flow_sum"] += float(metrics["flow_veh_per_h"]); d["flow_n"] += 1
        d["motion"].update(metrics.get("motion_counts") or {})
        d["counterflow"] += int(metrics.get("counterflow_count", 0) or 0)
        d["latest"] = metrics; d["entries"] += int(entries); d["exits"] += int(exits)

    def rows(self, interval_seconds, end_time=None, congestion_thresholds=None):
        output=[]
        thresholds = congestion_thresholds or {"green":.35,"yellow":.65}
        green_max = float(thresholds.get("green", .35))
        yellow_max = float(thresholds.get("yellow", .65))
        for (idx,key), d in sorted(self.values.items()):
            n=max(1,d["n"]); last=d["latest"]
            mean_score = d["score"] / d["score_n"] if d["score_n"] else None
            interval_level = ("unknown" if mean_score is None else
                              "green" if mean_score < green_max else
                              "yellow" if mean_score < yellow_max else "red")
            # Flow is a rate, so it comes from entries over the interval's real
            # duration - the last interval is usually short and dividing by the
            # nominal length would understate it.
            start_sec = idx * interval_seconds
            end_sec = (min((idx+1)*interval_seconds, end_time)
                       if end_time is not None else (idx+1)*interval_seconds)
            span = max(1e-6, end_sec - start_sec)
            motion = d["motion"]
            motion_total = sum(motion.values())
            stationary = int(motion.get("stationary", 0))
            moving_counts = {k: v for k, v in motion.items()
                             if k not in ("stationary", "unknown")}
            output.append({"interval_start_sec":round(idx*interval_seconds,2),
                "interval_end_sec":round(min((idx+1)*interval_seconds, end_time) if end_time is not None else (idx+1)*interval_seconds,2), "segment_id":key[0],"direction":key[1],
                "current_vehicle_count":last.get("vehicle_count",0),"class_counts":last.get("class_counts",{}),
                "entry_count":d["entries"],"exit_count":d["exits"],"occupancy_pct":round(d["occ"]/n,3),
                "average_speed_kmh":round(d["speed_sum"]/d["speed_n"],2) if d["speed_n"] else None,
                "congestion_score":None if mean_score is None else round(mean_score,4),
                "congestion_level":interval_level,
                # Flow from the fundamental relation q = k*v*lanes, not from gate
                # crossings. A gate fires only when the SAME track id is seen on
                # both sides of the line, so every id change at the gate deletes a
                # crossing outright; measured against SUMO ground truth that read
                # 0.345 of true flow (0.257 on a jammed approach, r=0.33), while
                # q = k*v read 1.11 with r=0.80. The gate count is kept alongside
                # as `gate_flow_veh_per_h` because it is still the right signal for
                # turning movements and for diagnosing fragmentation - it is just
                # not a flow measurement while tracks fragment.
                "flow_veh_per_h":(round(d["flow_sum"]/d["flow_n"],1)
                                  if d["flow_n"] else None),
                "gate_flow_veh_per_h":round(d["entries"]/span*3600.0,1),
                "density_veh_per_km_lane":(round(d["density_sum"]/d["density_n"],2)
                                          if d["density_n"] else None),
                "segment_length_m":(round(last["segment_length_m"],1)
                                    if last.get("segment_length_m") else None),
                "dominant_motion":(max(moving_counts, key=moving_counts.get)
                                   if moving_counts else "none"),
                "stationary_pct":(round(100.0*stationary/motion_total,1) if motion_total else None),
                "moving_pct":(round(100.0*sum(moving_counts.values())/motion_total,1)
                              if motion_total else None),
                "counterflow_count":d["counterflow"],
                "measurement_confidence":round(d["conf"]/n,4),"data_quality":last.get("data_quality","unknown"),
                "classification_basis":last.get("classification_basis","occupancy_only")})
        return output
