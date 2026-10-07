"""Model + device resolution shared by train.py, validate.py, detect.py, track.py.

The model name is never hard-coded to one YOLO version. `resolve_model()` walks a
preference list of detection checkpoints for the requested scale and returns the first one
the *installed* Ultralytics release actually knows about, so the project picks up
a newer architecture automatically when Ultralytics ships one.
"""
from __future__ import annotations

import os
from pathlib import Path

# Newest first, one preference list per model scale. The first entry available in
# the installed Ultralytics asset list wins, so the project picks up a newer
# architecture automatically without a code change.
PREFERRED_NANO = ["yolo26n.pt", "yolo12n.pt", "yolo11n.pt", "yolov8n.pt"]
PREFERRED_SMALL = ["yolo26s.pt", "yolo12s.pt", "yolo11s.pt", "yolov8s.pt"]
PREFERRED_MEDIUM = ["yolo26m.pt", "yolo12m.pt", "yolo11m.pt", "yolov8m.pt"]
PREFERRED_LARGE = ["yolo26l.pt", "yolo12l.pt", "yolo11l.pt", "yolov8l.pt"]
PREFERRED_XLARGE = ["yolo26x.pt", "yolo12x.pt", "yolo11x.pt", "yolov8x.pt"]

# Scale key accepted by resolve_model(size=...) / train.py --size.
PREFERENCES = {
    "n": PREFERRED_NANO,
    "s": PREFERRED_SMALL,
    "m": PREFERRED_MEDIUM,
    "l": PREFERRED_LARGE,
    "x": PREFERRED_XLARGE,
}
SIZES = list(PREFERENCES)

# Overridable without touching code:  set MODEL=yolo11s.pt in the environment.
ENV_MODEL = "MODEL"


def available_assets() -> set:
    """Detection checkpoints the installed Ultralytics version can auto-download."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
        return set(GITHUB_ASSETS_NAMES)
    except Exception:
        return set()


def resolve_model(requested: str | None = None, size: str = "n") -> str:
    """Return a concrete checkpoint name/path to load.

    Resolution order:
      1. `requested` (CLI --model), if given. A local .pt path is used verbatim.
      2. $MODEL environment variable.
      3. The newest entry of the preference list that the installed Ultralytics
         release actually offers.
    """
    cand = requested or os.environ.get(ENV_MODEL)
    if cand:
        # A path that exists (e.g. a fine-tuned best.pt) is always honoured.
        if Path(cand).exists() or cand.endswith((".pt", ".yaml")):
            return cand
        return cand

    assets = available_assets()
    prefs = PREFERENCES.get(size, PREFERRED_NANO)
    for name in prefs:
        if name in assets:
            return name
    return prefs[-1]  # last resort; Ultralytics will error clearly if unavailable


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device string. 'auto' -> cuda:0 / mps / cpu."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def describe_environment() -> dict:
    """Facts worth printing at the top of every run, for reproducibility."""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None)
    except Exception as e:
        info["torch"] = f"unavailable ({e})"
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics"] = f"unavailable ({e})"
    info["cpu_count"] = os.cpu_count()
    return info


def print_environment(prefix: str = ""):
    info = describe_environment()
    print(f"{prefix}ultralytics={info.get('ultralytics')}  torch={info.get('torch')}  "
          f"cuda={info.get('cuda_available')}  device_name={info.get('cuda_device')}  "
          f"cpus={info.get('cpu_count')}")
    return info


def detector_conf_floor(tracker_config, override=None, default=0.1):
    """Lowest detection score the predictor should emit when tracking.

    ByteTrack's whole premise ("BYTE") is a second association stage that rescues
    LOW-score detections to keep a track alive through a flicker. Those boxes only
    reach the tracker if the predictor's own `conf` is at or below the tracker's
    `track_low_thresh` - set conf higher and the recovery stage silently receives
    nothing, which fragments every track that dips in confidence for a few frames.

    Deriving the floor from the tracker file keeps the two from drifting apart.
    `track_high_thresh` still governs which detections can hold a track on the
    first pass, so lowering this floor adds recovery candidates rather than
    loosening the main association.
    """
    if override is not None:
        return float(override)
    try:
        import yaml
        data = yaml.safe_load(Path(tracker_config).read_text(encoding="utf-8")) or {}
        return float(data.get("track_low_thresh", default))
    except Exception:
        return float(default)
