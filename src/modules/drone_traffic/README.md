# Drone traffic module

`runner.py` runs YOLO vehicle detection, ByteTrack tracking, camera motion compensation, direction estimation, calibrated road-segment analytics, and output generation. `src/drone_dashboard.py` lets a user select a saved clip or upload a new one, watch it in a loop with analytics below, configure the analytics webhook, and copy a reusable embed URL. `app.py` serves that result at `?mode=drone_embed&traffic_video=<result-id>`.

The bundled `cfg/drone_traffic/traffic.yaml` describes two directions for one camera view. Treat segment counts, speed, density, and congestion as view-specific estimates. Calibrate polygons and scale before using another camera view or treating values as operational measurements. Final-frame moving/stationary totals are snapshots; stationary does not prove that a vehicle is parked.

A completed run sends one `drone_traffic_summary` event through the existing `WebhookDispatcher` using `DRONE_TRAFFIC_WEBHOOK_URL`, falling back to `WEBHOOK_URL`. The event includes final-frame direction/class/motion totals, segment states, gate flow, tracking quality, and speed availability. A saved-video embed can also accept `webhook` and `camera` query parameters and sends a `drone_traffic_playback_snapshot` immediately, then every 30 seconds while the video plays. Use `mode=drone_video_only` for a looping video-only iframe that still sends those snapshots. Camera ID is a separate query parameter; URL-encode the webhook endpoint as the value of `webhook`.

The trained checkpoint is supplied separately because `.pt` files are ignored. Set `DRONE_TRAFFIC_WEIGHTS` or place it at `models/yolo/drone_vehicles.pt`. Set `PUBLIC_BASE_URL` if reverse-proxy headers do not provide the externally reachable app URL.
