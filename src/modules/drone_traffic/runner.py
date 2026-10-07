#!/usr/bin/env python3
"""Run YOLO + ByteTrack + direction through road-wise occupancy/congestion analysis.

Example:
  python scripts/traffic.py --source videos/test1.mp4 --weights runs/visdrone_vehicles_gcp_l/weights/best.pt
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from webhook import WebhookDispatcher  # noqa: E402
from camera_motion import CameraMotionCompensator  # noqa: E402
from counting import CountingLine, LineCounter, lines_for_segments  # noqa: E402
from direction_estimator import DirectionEstimator, Observation  # noqa: E402
from model_utils import detector_conf_floor, resolve_device, resolve_model  # noqa: E402
from traffic_analysis import (CalibratedSpeedEstimator, CongestionClassifier,
                              IntervalAccumulator, RoadScaleSpeedEstimator,
                              RoadSegmentManager, TRAFFIC_BGR,
                              tracking_quality)  # noqa: E402
from tracking_io import CsvWriter  # noqa: E402
from visdrone_classes import color_for  # noqa: E402
from viz import draw_box  # noqa: E402

FRAME_COLUMNS = ["frame", "timestamp", "segment_id", "road_id", "direction",
                 "vehicle_count", "class_counts", "entry_count", "exit_count",
                 "occupancy_pct", "density_veh_per_km_lane", "average_speed_kmh",
                 "dominant_motion", "stationary_count", "counterflow_count",
                 "congestion_score",
                 "congestion_level", "measurement_confidence", "data_quality",
                 "classification_basis"]


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _mapped_bbox(matrix, box):
    x1,y1,x2,y2 = box
    corners = np.asarray([[[x1,y1]],[[x2,y1]],[[x2,y2]],[[x1,y2]]], dtype=np.float32)
    p = cv2.perspectiveTransform(corners, matrix).reshape(-1,2)
    return float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())


def _summarize_final_frame(current_states):
    """Aggregate per-segment final-frame measurements by travel direction."""
    directions = {}
    all_classes = Counter()
    total_vehicles = total_moving = total_stationary = 0
    for (segment_id, direction), state in sorted(current_states.items()):
        direction = str(direction)
        item = directions.setdefault(direction, {
            "vehicle_count_final_frame": 0,
            "moving_count": 0,
            "stationary_count": 0,
            "class_counts": Counter(),
            "segments": {},
        })
        vehicle_count = int(state.get("vehicle_count", 0))
        classes = {str(name): int(count) for name, count in (state.get("class_counts") or {}).items()}
        motion = {str(name): int(count) for name, count in (state.get("motion_counts") or {}).items()}
        moving_count = sum(count for name, count in motion.items() if name not in ("stationary", "unknown"))
        stationary_count = int(state.get("stationary_count", motion.get("stationary", 0)))
        item["vehicle_count_final_frame"] += vehicle_count
        item["moving_count"] += moving_count
        item["stationary_count"] += stationary_count
        item["class_counts"].update(classes)
        item["segments"][str(segment_id)] = {
            "vehicle_count": vehicle_count,
            "class_counts": classes,
            "moving_count": moving_count,
            "stationary_count": stationary_count,
            "motion_counts": motion,
            "average_speed_kmh": state.get("average_speed_kmh"),
            "congestion_level": state.get("level"),
            "congestion_score": state.get("score"),
        }
        all_classes.update(classes)
        total_vehicles += vehicle_count
        total_moving += moving_count
        total_stationary += stationary_count

    for item in directions.values():
        item["class_counts"] = dict(item["class_counts"])
    return {
        "vehicle_count_final_frame": total_vehicles,
        "moving_count": total_moving,
        "stationary_count": total_stationary,
        "vehicle_class_counts": dict(all_classes),
        "directions": directions,
        "count_basis": "vehicles assigned to road segments in the final processed frame; not unique trip totals",
        "stationary_note": "stationary is a motion classification and does not confirm a parked vehicle",
    }
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--traffic-config", default=str(ROOT / "cfg" / "drone_traffic" / "traffic.yaml"))
    ap.add_argument("--direction-config", default=str(ROOT / "cfg" / "drone_traffic" / "direction.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "drone_traffic"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=960)
    # Default comes from the tracker's track_low_thresh, not a constant: see
    # model_utils.detector_conf_floor for why a higher floor starves ByteTrack's
    # recovery stage and fragments tracks.
    ap.add_argument("--conf", type=float, default=None,
                   help="Detector score floor. Default: the tracker config's "
                        "track_low_thresh, so low-score recovery actually works.")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-save-video", action="store_true")
    ap.add_argument("--tracker-config", default=None,
                   help="Tracker YAML (default: configs/bytetrack.yaml).")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = _load_yaml(args.traffic_config)
    dir_cfg = _load_yaml(args.direction_config)
    camera = CameraMotionCompensator(cfg.get("camera_motion", dir_cfg.get("camera_motion", {})))
    source = Path(args.source)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise SystemExit(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cfg.get("segments"):
        raise SystemExit("Traffic configuration has no road segments. Define calibrated polygons in --traffic-config.")
    roads = RoadSegmentManager(cfg, width, height)
    if not roads.segments:
        raise SystemExit("No usable road masks were created; verify road polygons and frame dimensions.")
    direction = DirectionEstimator(dir_cfg)
    # A surveyed homography wins when present; otherwise fall back to the
    # road-width calibration, which still beats emitting no speed at all.
    speed_cfg = cfg.get("speed", {})
    use_road_scale = not speed_cfg.get("ground_homography") and roads.road_width_m > 0
    speed = (RoadScaleSpeedEstimator(speed_cfg) if use_road_scale
             else CalibratedSpeedEstimator(speed_cfg))
    print(f"speed basis : {'road-width scale (%.1fm)' % roads.road_width_m if use_road_scale else 'ground homography' if speed_cfg.get('ground_homography') else 'unavailable'}")
    classifier = CongestionClassifier(cfg, fps)
    interval_sec = max(1.0, float(cfg.get("aggregation_interval_seconds", 10)))
    accumulator = IntervalAccumulator()
    model = YOLO(resolve_model(args.weights))
    # Counting gates. Flow is measured by line crossing, not by area entries:
    # fragmented IDs inflate area entries 7-10x, while a crossing happens in one
    # frame transition and so is counted once per vehicle. Gates sit on segment
    # boundaries, so one segment's exit gate IS the next one's entry gate and
    # inflow-minus-outflow stays conservative.
    gates = []
    entry_of, exit_of = {}, {}
    for base in roads.base_roads:
        children = sorted(base["children"], key=lambda c: int(c.get("section_index", 1)))
        road_gates = lines_for_segments(children)
        for position, line in enumerate(road_gates):
            entry_of[line.line_id] = line.segment_id
            # A vehicle entering section n has left section n-1.
            exit_of[line.line_id] = (children[position - 1]["segment_id"]
                                     if position > 0 else None)
        last = children[-1]
        tail = last["centerline_px"].reshape(-1, 2).astype(float)
        if len(tail) >= 2:
            # The final section has no downstream neighbour, so without a gate at
            # the road's end its outflow would read zero and it would look like it
            # were filling forever.
            tail_vec = tail[-1] - tail[-2]
            norm = float(np.linalg.norm(tail_vec))
            if norm > 0:
                travel = (float(tail_vec[0] / norm), float(tail_vec[1] / norm))
                normal = (-travel[1], travel[0])
                width_px = last.get("mean_width_px")
                half = (float(width_px) / 2.0 * 1.15) if width_px else 60.0
                line_id = f"{last['segment_id']}:out"
                road_gates.append(CountingLine(
                    line_id=line_id,
                    p1=(tail[-1][0] + normal[0] * half, tail[-1][1] + normal[1] * half),
                    p2=(tail[-1][0] - normal[0] * half, tail[-1][1] - normal[1] * half),
                    travel=travel, segment_id=last["segment_id"], role="exit"))
                entry_of[line_id] = None
                exit_of[line_id] = last["segment_id"]
        gates.extend(road_gates)
    counter = LineCounter(gates)
    print(f"counting gates: {len(gates)}")

    tracker = Path(args.tracker_config) if args.tracker_config else ROOT / "cfg" / "drone_traffic" / "bytetrack.yaml"
    conf_floor = detector_conf_floor(tracker, args.conf)
    print(f"detector conf floor: {conf_floor} (tracker: {tracker.name})")
    output_video = out / f"{stem}_traffic.mp4"
    writer = None if args.no_save_video else cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if writer is not None and not writer.isOpened(): raise SystemExit(f"Cannot open VideoWriter: {output_video}")
    frame_csv = out / f"{stem}_traffic_frames.csv"
    track_csv = out / f"{stem}_traffic_tracks.csv"
    track_columns = ["frame","timestamp","track_id","class_name","confidence","center_x","center_y",
                     "direction","direction_confidence","segment_id","road_id","road_direction","speed_kmh"]
    segment_events = Counter()
    track_segment = {}
    all_track_ids = set()
    # Flow must count each vehicle once per segment. Raw entry transitions
    # double-count a track that jitters across a boundary, so first entries are
    # deduplicated; fragmentation is reported separately because no amount of
    # deduplication can rejoin one vehicle that the tracker split into five IDs.
    counted_entry = set()
    track_span = {}
    track_segments = defaultdict(set)
    last_seen = {}
    current_states = {}
    segment_detection_history = defaultdict(lambda: deque(maxlen=classifier.window_frames))
    trajectories = defaultdict(lambda: deque(maxlen=int(dir_cfg.get("trajectory_length",30))))
    frame_idx = 0
    run_start = time.perf_counter()
    frame_latencies=[]
    car_lengths_m=[]
    interval_rows=[]
    interval_entries=Counter(); interval_exits=Counter()
    gate_crossings=Counter()
    save = cfg.get("visualization", {})

    with CsvWriter(frame_csv, FRAME_COLUMNS) as frame_writer, CsvWriter(track_csv, track_columns) as track_writer:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames: break
            timestamp = frame_idx / fps
            matrix, camera_reliable, camera_status = camera.transform(frame)
            inverse_matrix = np.linalg.inv(matrix)
            result = model.track(frame, persist=True, tracker=str(tracker), imgsz=args.imgsz,
                                 conf=conf_floor, device=resolve_device(args.device),
                                 max_det=int(cfg.get("max_detections", 600)),
                                 verbose=False)[0]
            detections = defaultdict(list)
            class_counts = defaultdict(Counter)
            motion_counts = defaultdict(Counter)
            counterflow = Counter()
            speeds = defaultdict(list)
            confidences = defaultdict(list)
            counts = Counter()
            observed = set()
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for tid,cid,det_conf,xyxy in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                                 boxes.conf.tolist(), boxes.xyxy.tolist()):
                    tid,cid=int(tid),int(cid)
                    x1,y1,x2,y2=map(float,xyxy); cx,cy=(x1+x2)/2,(y1+y2)/2
                    stable_center = camera.map_point(matrix,(cx,cy)) if camera.enabled else (cx,cy)
                    stable_box = _mapped_bbox(matrix,(x1,y1,x2,y2)) if camera.enabled else (x1,y1,x2,y2)
                    cname = str(model.names.get(cid,cid))
                    obs = Observation(frame_idx,timestamp,cx,cy,float(det_conf),(x1,y1,x2,y2),cid,cname)
                    ds = direction.observe(tid,obs,stable_center,camera_reliable)
                    seg = roads.assign(stable_center,ds.get("vector"),ds.get("direction_confidence",0))
                    segid = seg.get("segment_id","") if seg else ""
                    old = track_segment.get(tid,"")
                    # Area transitions are kept as a DIAGNOSTIC only; they no
                    # longer feed interval flow. With IDs fragmenting, one vehicle
                    # re-enters a section under several ids and inflates the rate
                    # 7-10x. The gate crossings below are the flow measurement.
                    if segid != old:
                        if old:
                            segment_events[(old,"exit")] += 1
                        if segid:
                            segment_events[(segid,"entry")] += 1
                            counted_entry.add((tid,segid))
                            track_segment[tid]=segid
                        else:
                            track_segment.pop(tid,None)
                    last_seen[tid]=frame_idx; observed.add(tid); all_track_ids.add(tid)
                    span=track_span.setdefault(tid,[frame_idx,frame_idx]); span[1]=frame_idx
                    if segid: track_segments[tid].add(segid)
                    for line_id in counter.update(tid, stable_center):
                        gate_crossings[line_id] += 1
                        entered, left = entry_of.get(line_id), exit_of.get(line_id)
                        if entered:
                            seg_in=next((s for s in roads.segments if s["segment_id"]==entered),None)
                            if seg_in is not None:
                                interval_entries[(entered,str(seg_in.get("direction","unknown")))]+=1
                        if left:
                            seg_out=next((s for s in roads.segments if s["segment_id"]==left),None)
                            if seg_out is not None:
                                interval_exits[(left,str(seg_out.get("direction","unknown")))]+=1
                    if use_road_scale:
                        mpp=roads.meters_per_pixel(seg,stable_center) if seg else None
                        kmh=speed.update(tid,stable_center,timestamp,mpp,camera_reliable)
                    else:
                        kmh=speed.update(tid,stable_center,timestamp,camera_reliable)
                    track_writer.write(frame=frame_idx,timestamp=round(timestamp,4),track_id=tid,class_name=cname,
                        confidence=round(float(det_conf),4),center_x=round(cx,2),center_y=round(cy,2),
                        direction=ds["direction"],direction_confidence=ds["direction_confidence"],
                        segment_id=segid,road_id=seg.get("road_id","") if seg else "",
                        road_direction=seg.get("direction","") if seg else "",speed_kmh="" if kmh is None else round(kmh,2))
                    trajectories[tid].append((int(cx),int(cy)))
                    if seg:
                        key=(segid,str(seg.get("direction","unknown")))
                        detections[segid].append(stable_box)
                        cname_count = "motorcycle" if cname in ("motor","motorcycle") else cname
                        class_counts[key][cname_count]+=1; counts[key]+=1; confidences[key].append(float(det_conf))
                        if kmh is not None: speeds[key].append(kmh)
                        motion_counts[key][ds["direction"]] += 1
                        # Scale self-check: a car is ~4.4m long, so measuring the
                        # detected box along the travel axis says whether the
                        # width-derived scale is believable. Approximate - an
                        # axis-aligned box over-covers a diagonally moving car.
                        if use_road_scale and mpp and cname == "car":
                            rdx,rdy=seg["direction_vector"]
                            extent=abs((x2-x1)*float(rdx))+abs((y2-y1)*float(rdy))
                            if extent>0: car_lengths_m.append(extent*mpp)
                        # Counter-flow: moving against the carriageway's declared
                        # travel vector. Only judged on confident motion, so a
                        # jittering stationary box is never reported wrong-way.
                        road_vec=seg.get("direction_vector")
                        if (ds["magnitude"]>0 and road_vec is not None
                                and np.linalg.norm(road_vec)>0
                                and ds["direction_confidence"]>=float(cfg.get("counterflow_min_confidence",0.5))):
                            if float(np.dot(ds["vector"],road_vec))<-0.25:
                                counterflow[key]+=1
                    if save.get("show_trajectories",True):
                        pts=list(trajectories[tid])
                        for a,b in zip(pts,pts[1:]): cv2.line(frame,a,b,color_for(cid),1,cv2.LINE_AA)
                    label_parts=[]
                    if save.get("show_classes",True): label_parts.append(cname[:1].upper())
                    if save.get("show_ids",True): label_parts.append(f"#{tid}")
                    if save.get("show_directions",True): label_parts.append(ds["direction"])
                    state_label=" ".join(label_parts)
                    if save.get("show_boxes",True): draw_box(frame,x1,y1,x2,y2,cid,state_label,thickness=1,font_scale=.36)
                    if save.get("show_directions",True) and ds["magnitude"]>0:
                        tip_stable=np.asarray(stable_center,dtype=float)+np.asarray(ds["vector"],dtype=float)*35
                        tip_raw=camera.map_point(inverse_matrix,tip_stable)
                        cv2.arrowedLine(frame,(int(cx),int(cy)),tuple(map(int,tip_raw)),color_for(cid),2,tipLength=.35)

            expired=[tid for tid,last in last_seen.items() if frame_idx-last>direction.max_gap]
            for tid in expired:
                sid=track_segment.pop(tid,"")
                if sid:
                    # A track expiring is an identity ending, not a vehicle
                    # leaving the road, so it must not count as outflow.
                    segment_events[(sid,"exit")]+=1
                last_seen.pop(tid,None)
            counter.forget(expired)
            direction.prune(frame_idx)
            occupancy=roads.measure_occupancy(detections)
            visual_states={}
            current_interval=int((timestamp-1e-9)//interval_sec)
            for seg in roads.segments:
                sid=seg["segment_id"]; direct=str(seg.get("direction","unknown")); key=(sid,direct)
                n=int(counts[key]); conf=float(np.mean(confidences[key])) if confidences[key] else .5
                speed_avg=float(np.mean(speeds[key])) if speeds[key] else None
                segment_detection_history[key].append(n > 0)
                enough_detections = (sum(segment_detection_history[key]) >=
                                     int(cfg.get("minimum_detection_frames_in_window", 0)))
                state=classifier.update(key,occupancy.get(sid,0.0),speed_avg,timestamp,
                                        confidence=conf,valid=camera_reliable and enough_detections)
                quality=("unreliable_camera_motion" if not camera_reliable else
                         ("image_based_approximation" if speed_avg is None else
                          "road_width_scaled_speed" if use_road_scale else "calibrated_speed"))
                # Density is the traffic-engineering measure: vehicles per km per
                # lane, which unlike occupancy_pct is comparable between a far
                # section and a near one.
                length_m=seg.get("length_m")
                lanes=max(1,int(seg.get("lanes",1)))
                density=(n/(length_m/1000.0*lanes)
                         if length_m and length_m>0 else None)
                # Flow by the fundamental relation, formed HERE so density and
                # speed are the same frame's. Both are already measured well
                # (speed is within 0.3% of truth); gate crossings are not, because
                # they need one track id to survive the gate. See the note in
                # IntervalAccumulator.rows.
                # An empty section carries no flow, and speed is undefined there
                # because there was nothing to measure. Reporting None would drop
                # every quiet interval out of the series; reporting 0 states what
                # was actually observed. None stays reserved for "vehicles present
                # but speed unavailable", which is a different claim.
                flow=(0.0 if density == 0 else
                      density*speed_avg*lanes
                      if density is not None and speed_avg is not None else None)
                motion=dict(motion_counts[key])
                moving={k:v for k,v in motion.items() if k not in ("stationary","unknown")}
                metrics={**state,"vehicle_count":n,"class_counts":dict(class_counts[key]),
                         "data_quality":quality,"road_id":seg.get("road_id",sid),"direction":direct,
                         "entry_count":interval_entries[key],"exit_count":interval_exits[key],
                         "density_veh_per_km_lane":density,"segment_length_m":length_m,"average_speed_kmh":speed_avg,
                         "lanes":lanes,"flow_veh_per_h":flow,
                         "motion_counts":motion,"counterflow_count":int(counterflow[key]),
                         "dominant_motion":max(moving,key=moving.get) if moving else "none",
                         "stationary_count":int(motion.get("stationary",0))}
                current_states[key]=metrics; visual_states[sid]=metrics
                frame_writer.write(frame=frame_idx,timestamp=round(timestamp,4),segment_id=sid,
                    road_id=seg.get("road_id",sid),direction=direct,vehicle_count=n,
                    class_counts=json.dumps(dict(class_counts[key]),sort_keys=True),entry_count=interval_entries[key],
                    exit_count=interval_exits[key],occupancy_pct=metrics["occupancy_pct"],
                    density_veh_per_km_lane="" if density is None else round(density,2),
                    average_speed_kmh="" if speed_avg is None else round(speed_avg,2),
                    dominant_motion=metrics["dominant_motion"],
                    stationary_count=metrics["stationary_count"],
                    counterflow_count=metrics["counterflow_count"],
                    congestion_score="" if state.get("score") is None else state["score"],
                    congestion_level=state["level"],measurement_confidence=state["confidence"],
                    data_quality=quality,classification_basis=state.get("classification_basis","occupancy_only"))
                accumulator.add(current_interval,key,metrics,interval_entries[key],interval_exits[key])
                interval_entries[key]=0; interval_exits[key]=0

            if save.get("show_traffic_lines",True):
                roads.draw(frame,inverse_matrix,visual_states,show_boundaries=save.get("show_road_boundaries",True),
                           show_labels=save.get("show_labels",True),line_thickness=cfg.get("visualization",{}).get("traffic_line_thickness",14))
            levels=Counter(v.get("level","unknown") for v in visual_states.values())
            cv2.putText(frame,"Traffic: "+"  ".join(f"{k.upper()} {v}" for k,v in sorted(levels.items())),
                        (10,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA)
            cv2.putText(frame,camera_status,(10,height-14),cv2.FONT_HERSHEY_SIMPLEX,.45,
                        (0,255,0) if camera_reliable else (0,100,255),1,cv2.LINE_AA)
            if writer: writer.write(frame)
            frame_latencies.append(time.perf_counter()-loop_start)
            if frame_idx%100==0: print(f"frame {frame_idx}/{total_frames}: vehicles={sum(counts.values())} segments={len(roads.segments)}")

    cap.release()
    if writer: writer.release()
    interval_csv=out/f"{stem}_traffic_summary.csv"
    rows=accumulator.rows(interval_sec,frame_idx/fps,
                          cfg.get("congestion", {}).get("thresholds", {}))
    fields=["interval_start_sec","interval_end_sec","segment_id","direction","current_vehicle_count","class_counts",
            "entry_count","exit_count","flow_veh_per_h","gate_flow_veh_per_h",
            "occupancy_pct","density_veh_per_km_lane",
            "segment_length_m","average_speed_kmh","dominant_motion","moving_pct","stationary_pct",
            "counterflow_count","congestion_score","congestion_level",
            "measurement_confidence","data_quality","classification_basis"]
    with open(interval_csv,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for row in rows:
            row["class_counts"]=json.dumps(row["class_counts"],sort_keys=True); w.writerow(row)
    elapsed=time.perf_counter()-run_start
    process=psutil.Process(os.getpid())
    final={"source":str(source),"frames_processed":frame_idx,"source_fps":fps,
        "processing_fps":round(frame_idx/max(elapsed,1e-9),3),"elapsed_sec":round(elapsed,2),
        "average_frame_latency_ms":round(1000*float(np.mean(frame_latencies)),2) if frame_latencies else None,
        "peak_memory_mb":round(process.memory_info().rss/1024/1024,1),
        "tracked_ids":len(all_track_ids),
        "camera_motion_enabled":camera.enabled,"camera_motion_status":camera.status,
        "speed_available":use_road_scale or getattr(speed,"matrix",None) is not None,
        "speed_basis":("road_width_scale" if use_road_scale
                       else "ground_homography" if getattr(speed,"matrix",None) is not None
                       else "none"),
        "speed_note":("Speed uses the configured ground-plane homography." if getattr(speed,"matrix",None) is not None
                      else f"Speed is scaled from an assumed {roads.road_width_m}m road width, not a surveyed homography; see scale_calibration." if use_road_scale
                      else "km/h is emitted only when speed.ground_homography maps image pixels into calibrated ground-plane meters; no such calibration is configured."),
        "scale_calibration":{
            "road_width_m":roads.road_width_m or None,
            "lanes_per_direction":roads.lanes_per_direction,
            "segment_length_m":{s["segment_id"]:(round(s["length_m"],1) if s.get("length_m") else None)
                                for s in roads.segments},
            "meters_per_pixel":{s["segment_id"]:(round(s["mean_meters_per_pixel"],4)
                                                 if s.get("mean_meters_per_pixel") else None)
                                for s in roads.segments}},
        "scale_diagnostics":({
            "observed_car_length_m_median":round(float(np.median(car_lengths_m)),2),
            "expected_car_length_m":4.4,
            "implied_length_correction":round(4.4/float(np.median(car_lengths_m)),2),
            "note":("Ratio >1 means the isotropic-scale assumption is foreshortening "
                    "travel-direction distance, so speeds and segment lengths are "
                    "under-estimated by roughly this factor. Multiply road_width_m by "
                    "it only if the view is near-nadir; otherwise supply a longitudinal "
                    "reference distance.")} if car_lengths_m else None),
        "tracking_quality":tracking_quality(track_span,track_segments,fps,
                                           max((int(s.get("section_count",1)) for s in roads.segments),default=1),
                                           flow_from_lines=bool(gates)),
        "counting_lines":{"gates":len(gates),
            "crossings":{k:int(v) for k,v in sorted(gate_crossings.items())},
            "veh_per_h":{k:round(v/max(frame_idx/fps,1e-9)*3600.0,1)
                         for k,v in sorted(gate_crossings.items())},
            "note":"Flow is counted at gates on segment boundaries, which is robust "
                   "to ID fragmentation; road_segment_events below are raw area "
                   "transitions kept only as a diagnostic."},
        "road_segment_events":{f"{sid}:{event}":n for (sid,event),n in segment_events.items()},
        "final_segment_states":{f"{k[0]}:{k[1]}":{k2:v2 for k2,v2 in v.items() if k2 not in ("pending","pending_since","since","ema")}
                                for k,v in current_states.items()},
        "limitations":["Road polygons are manually approximated for the configured reference frame; occupancy is a box-union image-area estimate, not ground-plane area.",
            "Speed and density rest on an ASSUMED road width, not a survey: both scale linearly with scale.road_width_m, so a 10% width error is a 10% speed error.",
            "The width-derived scale is isotropic per sample, which is exact only for a nadir view; on this oblique view travel-direction distance is foreshortened more than width, biasing speeds and lengths DOWNWARD. See scale_diagnostics.implied_length_correction.",
            "Density assumes scale.lanes_per_direction lanes uniformly; it is veh/km/lane only if that count is right for every section.",
            "No manually labelled traffic ground truth was supplied, so congestion accuracy is not claimed.",
            "Counts and occupancy inherit detector misses, box overlap, and ByteTrack fragmentation."]}
    final["source_video"] = os.environ.get("DRONE_SOURCE_FILENAME") or source.name
    final["final_frame_analytics"] = _summarize_final_frame(current_states)
    (out/f"{stem}_traffic_summary.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    # Send the completed analysis through the platform's existing webhook publisher.
    webhook_url = os.environ.get("DRONE_TRAFFIC_WEBHOOK_URL") or os.environ.get("WEBHOOK_URL")
    if webhook_url:
        webhook = WebhookDispatcher(camera_id=os.environ.get("DRONE_CAMERA_ID") or source.stem)
        delivered, status_code, status_desc = webhook.dispatch(webhook_url, {
            "event": "drone_traffic_summary",
            "timestamp_sec": round(frame_idx / fps, 2),
            "source_video": final["source_video"],
            "frames_processed": frame_idx,
            "tracked_ids": len(all_track_ids),
            "processing_fps": final["processing_fps"],
            "segment_count": len(current_states),
            "final_frame_analytics": final["final_frame_analytics"],
            "final_segment_states": final["final_segment_states"],
            "road_segment_events": final["road_segment_events"],
            "counting_lines": final["counting_lines"],
            "tracking_quality": final["tracking_quality"],
            "speed_available": final["speed_available"],
            "speed_basis": final["speed_basis"],
        }, sync=True)
        if delivered:
            print(f"[DRONE WEBHOOK] sent successfully (status={status_code})")
        else:
            print(f"[DRONE WEBHOOK] delivery failed (status={status_code or 'N/A'}): {status_desc}")
    print(f"Annotated video: {output_video if writer else 'not saved'}")
    print(f"Per-frame measurements: {frame_csv}\nInterval summary: {interval_csv}\nJSON summary: {out/f'{stem}_traffic_summary.json'}")
    print(f"Processed {frame_idx} frames at {final['processing_fps']} FPS; speed calibration available={final['speed_available']}")


if __name__=="__main__": main()
