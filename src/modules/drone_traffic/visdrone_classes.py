"""VisDrone2019-DET class definitions and the VisDrone -> YOLO class mapping.

Single source of truth: prepare_visdrone.py, train.py, validate.py, detect.py and
track.py all import from here so the dataset YAML, the trained model and the
inference-time label text can never drift apart.

VisDrone annotation line format (one object per line, one .txt per image):

    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

  bbox_*      : pixel coordinates, top-left origin, width/height (NOT x2/y2)
  score       : DET splits use 1 = evaluated object, 0 = ignored region
  category    : see VISDRONE_CATEGORIES below (0..11)
  truncation  : 0 = none, 1 = partially outside the frame
  occlusion   : 0 = none, 1 = partial (1-50%), 2 = heavy (50-100%)

Note the annotations for the test-dev split are public; test-challenge is not
(its labels are withheld by the benchmark), which is why we ignore it.
"""

# ---------------------------------------------------------------------------
# The 12 original VisDrone category ids, exactly as they appear in column 6.
# ---------------------------------------------------------------------------
VISDRONE_CATEGORIES = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# ---------------------------------------------------------------------------
# Phase 1 target set: the five vehicle classes.
#
# Nothing is merged. Each VisDrone id maps to its own YOLO id and keeps its own
# name; the classes we are not training on are dropped, not folded into another.
# Deliberately excluded, and why:
#   0  ignored-regions  - not an object, marks a region the benchmark ignores
#   1  pedestrian       - not a vehicle
#   2  people           - not a vehicle
#   3  bicycle          - non-motorised, out of scope for traffic counting
#   7  tricycle         - regionally specific, very few instances
#   8  awning-tricycle  - as above
#   11 others           - unlabelled catch-all, no consistent semantics
# ---------------------------------------------------------------------------
VEHICLE_MAP = {
    4: 0,   # car   -> 0
    5: 1,   # van   -> 1
    6: 2,   # truck -> 2
    9: 3,   # bus   -> 3
    10: 4,  # motor -> 4
}
VEHICLE_NAMES = {0: "car", 1: "van", 2: "truck", 3: "bus", 4: "motor"}

# ---------------------------------------------------------------------------
# Full 10-class variant (VisDrone id - 1), the mapping the standard VisDrone
# benchmark uses. Kept so the same tooling can rebuild the all-class dataset
# for comparison without a second converter.
# ---------------------------------------------------------------------------
ALL_MAP = {vid: vid - 1 for vid in range(1, 11)}
ALL_NAMES = {vid - 1: name for vid, name in VISDRONE_CATEGORIES.items() if 1 <= vid <= 10}


def get_mapping(vehicles_only: bool = True):
    """Return (visdrone_id -> yolo_id, yolo_id -> name) for the chosen class set."""
    if vehicles_only:
        return dict(VEHICLE_MAP), dict(VEHICLE_NAMES)
    return dict(ALL_MAP), dict(ALL_NAMES)


# Per-class BGR colours for OpenCV drawing, so each vehicle type is visually
# distinct in annotated video (indexed by YOLO id of the vehicles-only set).
VEHICLE_COLORS = {
    0: (0, 200, 255),    # car    - amber
    1: (255, 160, 0),    # van    - blue
    2: (0, 90, 255),     # truck  - orange-red
    3: (60, 220, 60),    # bus    - green
    4: (220, 80, 255),   # motor  - magenta
}


def color_for(class_id: int):
    """Stable BGR colour for a class id, with a fallback for the 10-class set."""
    if class_id in VEHICLE_COLORS:
        return VEHICLE_COLORS[class_id]
    h = (class_id * 67) % 180
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 180.0, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))
