# Poultry Monitoring Dashboard

Streamlit dashboard for poultry and smart-space analytics, with a drone traffic analysis mode.

## Drone traffic mode

Run the existing app and choose **Drone Traffic** under **Analytics mode**. Choose a saved traffic video to play it in a loop with its analytics directly underneath, or upload a video and select **Analyze video**. The result is saved on the server, then the page provides a stable video-and-analytics link plus an iframe snippet you can embed like the existing demos. The embed remains available while its output files remain on the server.

The included road geometry describes two travel directions for one camera view. Use it only with matching footage. Calibrate road polygons and scale before analyzing a different view or treating segment and speed values as operational measurements.

The analysis page accepts an **Analytics webhook URL**. A completed run sends a `drone_traffic_summary` JSON event containing final-frame class/direction/motion counts, per-segment values, crossing rates, tracking quality, and speed-calibration status. A blank field disables delivery for that run. You can also configure `DRONE_TRAFFIC_WEBHOOK_URL` or `WEBHOOK_URL` on the server. The embed link does not contain the webhook URL.

Install the project dependencies with `pip install -r requirements.txt`. Place the trained vehicle checkpoint at `models/yolo/drone_vehicles.pt`, or set `DRONE_TRAFFIC_WEIGHTS` to its path. Model weights are intentionally not committed. Set `PUBLIC_BASE_URL` if the app is behind a proxy and cannot infer its public URL. Results are written under `results/drone_traffic/`; they are runtime artifacts and are not committed.
