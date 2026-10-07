# Drone traffic module

`runner.py` runs YOLO vehicle detection, ByteTrack tracking, camera motion compensation, direction estimation, calibrated road-segment analytics, and output generation. The Streamlit entry point is `src/drone_dashboard.py`; it is selected from the existing dashboard in `app.py`.

The bundled `cfg/drone_traffic/traffic.yaml` describes two directions for one camera view. Treat segment-level counts, density, speed, and congestion as view-specific estimates. Calibrate polygons and scale against ground truth before using another camera view or treating the values as operational measurements.

The trained checkpoint is supplied separately (not committed because `.pt` files are ignored). Set `DRONE_TRAFFIC_WEIGHTS` or place it at `models/yolo/drone_vehicles.pt`. A completed run sends one `drone_traffic_summary` event using `DRONE_TRAFFIC_WEBHOOK_URL`, falling back to the platform-wide `WEBHOOK_URL`; no webhook is sent when neither is configured.