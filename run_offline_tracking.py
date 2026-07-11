import time
import cv2
import os
import sys

# Mock system time globally to map frame index to video-time
_original_time = time.time
virtual_time = 0.0
time.time = lambda: virtual_time

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import PoultryTracker
from src.analytics import PoultryAnalytics
from src.zones import ZoneManager
from src.visualization import Visualizer
from src.reporting import ReportGenerator
from ultralytics import YOLOWorld


def process_video(video_name, input_path, output_video_path, output_dir):
    # Determine mode based on video name
    is_poultry = True
    video_name_lower = str(video_name).lower()
    if any(term in video_name_lower for term in ["demo3", "demo4", "shed3", "shed4", "restaurant", "hotel", "people"]):
        is_poultry = False

    print(f"\n==========================================")
    print(f"Processing {video_name}...")
    print(f"Input: {input_path}")
    print(f"Output Video: {output_video_path}")
    print(f"Mode: {'Poultry (Chickens)' if is_poultry else 'Smart Space (People)'}")
    print(f"==========================================")
    
    # Initialize tracker (using yolov8s.pt and bytetrack)
    tracker = PoultryTracker(model_path='yolov8s.pt', tracker_algo='bytetrack')
    
    # Initialize analytics
    analytics = PoultryAnalytics()
    analytics.is_poultry = is_poultry
    visualizer = Visualizer()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Properties: FPS = {fps:.2f}, Total Frames = {total_frames}")
    
    # Read first frame to initialize ZoneManager
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        cap.release()
        return
        
    frame_resized = cv2.resize(frame, (640, 360))
    h, w = frame_resized.shape[:2]
    zone_manager = ZoneManager(w, h, is_poultry=is_poultry)
    
    # Run pot detection only if in poultry mode
    global virtual_time
    virtual_time = 0.0
    
    if is_poultry:
        print("Running initial zone detection (feeding and water pots)...")
        zone_detector = YOLOWorld("yolov8s-world.pt")
        zone_detector.set_classes(["feeding pot", "water pot"])
        world_results = zone_detector(frame_resized, verbose=False)
        world_dets = []
        for box in world_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = zone_detector.names.get(cls_id, f"object_{cls_id}")
            if conf_val > 0.05:
                world_dets.append({
                    "class_name": cls_name,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "box": (x1, y1, x2, y2),
                })
        zone_manager.update_pots(world_dets)
        print(f"Detected pot coordinates: {zone_manager.pots}")
    else:
        print("Skipping YOLOWorld pot detection; using predefined static coordinates for Smart Space mode.")
        print(f"Static Counter/Table coordinates: {zone_manager.pots}")
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 360))
    
    # Reset to frame 0 so we process all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    ignored_classes = [
        "bowl", "cup", "vase", "potted plant", "fire hydrant",
        "bottle", "wine glass", "traffic light", "chair",
    ]
    
    start_proc_time = _original_time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_resized = cv2.resize(frame, (640, 360))
        frame_count += 1
        
        # Advance virtual time based on video frame rate
        virtual_time = frame_count / fps
        
        # Run tracking on every frame for high quality and smooth overlay
        raw_detections = tracker.process_frame(
            frame_resized, conf_threshold=0.15, classes=None, is_poultry=is_poultry
        )
        
        # Filter ignored classes and overlaps with pots (if in poultry mode)
        detections = []
        for d in raw_detections:
            if d.get("class_name", "") in ignored_classes:
                continue
            
            if is_poultry:
                cx, cy = d["center"]
                is_pot = any(
                    ((cx - pot["center"][0]) ** 2 + (cy - pot["center"][1]) ** 2) ** 0.5 < 20
                    for pot in zone_manager.pots["feed"] + zone_manager.pots["water"]
                )
                if is_pot:
                    continue
            
            detections.append(d)
                
        # Update analytics
        for det in detections:
            track_id = det["track_id"]
            cx, cy = det["center"]
            x1, y1, x2, y2 = det["box"]
            box_area = (x2 - x1) * (y2 - y1)
            class_id = det.get("class_id", 14 if is_poultry else 0)
            class_name = det.get("class_name", "bird" if is_poultry else "person")
            zone = zone_manager.get_zone(cx, cy)
            analytics.update(
                track_id, cx, cy, zone,
                class_id=class_id, class_name=class_name, box_area=box_area
            )
            
        # Analyze flock grouping/crowd
        analytics.analyze_flock()
        
        # Draw visualizations
        frame_disp = visualizer.draw_zones(frame_resized.copy(), zone_manager)
        frame_disp = visualizer.draw_tracking(frame_disp, detections, analytics)
        
        # Write to video
        out.write(frame_disp)
        
        if frame_count % 100 == 0:
            elapsed = _original_time() - start_proc_time
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) | Processing Speed: {fps_proc:.1f} FPS")
            
    cap.release()
    out.release()
    print(f"Finished processing {video_name}. Video saved to {output_video_path}")
    
    # Generate CSV & PDF reports
    print("Generating reports...")
    rg = ReportGenerator(analytics, output_dir=output_dir)
    csv_file = rg.export_csv()
    pdf_file = rg.export_pdf()
    
    # Rename reports to clean names
    std_csv = os.path.join(output_dir, f"{video_name}_report.csv")
    std_pdf = os.path.join(output_dir, f"{video_name}_report.pdf")
    
    if os.path.exists(std_csv):
        os.remove(std_csv)
    if os.path.exists(std_pdf):
        os.remove(std_pdf)
        
    os.rename(csv_file, std_csv)
    os.rename(pdf_file, std_pdf)
    print(f"Reports successfully generated:\n  - CSV: {std_csv}\n  - PDF: {std_pdf}")


if __name__ == "__main__":
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Process all four demo videos
    for idx in range(1, 5):
        video_name = f"demo{idx}"
        input_path = f"videos/{video_name}.mp4"
        output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        
        if os.path.exists(input_path):
            process_video(video_name, input_path, output_path, output_dir)
        else:
            print(f"Warning: {input_path} not found, skipping.")
        
    print("\nOffline processing completed successfully for all files!")
