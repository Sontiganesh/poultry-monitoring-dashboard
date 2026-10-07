# Poultry Monitoring Dashboard

Streamlit dashboard for poultry and smart-space analytics, with a drone traffic analysis mode.

## Drone traffic mode

Run the existing app and choose **Drone Traffic** under **Analytics mode**. The included road geometry describes two travel directions for one camera view. Use it only with matching footage. Calibrate the road polygons and scale before analyzing a different view or treating segment and speed values as operational measurements.

Install the project dependencies with `pip install -r requirements.txt`. Place the trained vehicle checkpoint at `models/yolo/drone_vehicles.pt`, or set `DRONE_TRAFFIC_WEIGHTS` to its path. Model weights are intentionally not committed to this repository. Analysis outputs are written under `results/drone_traffic/` and include an annotated video, frame/track CSV files, and a JSON summary. Set `DRONE_TRAFFIC_WEBHOOK_URL` (or the existing `WEBHOOK_URL`) to receive a completed analysis summary through the platform webhook dispatcher.