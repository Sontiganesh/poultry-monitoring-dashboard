"""Streaming CSV writers for detection and tracking output.

Rows are flushed as they are produced rather than accumulated, so a long video
that is interrupted still leaves a usable, complete-up-to-that-frame CSV, and
memory stays flat regardless of video length.

Phase 2 (density, speed) will consume TRACK_COLUMNS, so the schema is fixed here
and both writers emit centre points ready for trajectory maths.
"""
from __future__ import annotations

import csv
from pathlib import Path

TRACK_COLUMNS = [
    "frame",        # 1-based frame index
    "timestamp",    # seconds from start of video (frame / fps)
    "track_id",     # persistent ByteTrack id
    "class_id",     # YOLO class id
    "class_name",   # human-readable class
    "confidence",   # detection confidence for this observation
    "x1", "y1", "x2", "y2",     # pixel bbox, top-left / bottom-right
    "center_x", "center_y",     # bbox centre, for trajectory/speed work
]

# detect.py has no tracker, so no track_id column.
DETECT_COLUMNS = [
    "frame", "timestamp", "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
]


class CsvWriter:
    """Tiny append-as-you-go CSV writer with a fixed header."""

    def __init__(self, path, columns):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.columns)
        self._w.writeheader()
        self.rows = 0

    def write(self, **row):
        self._w.writerow({c: row.get(c, "") for c in self.columns})
        self.rows += 1
        if self.rows % 500 == 0:
            self._fh.flush()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_row(frame, timestamp, class_id, class_name, conf, x1, y1, x2, y2,
             track_id=None):
    """Build one row dict, computing the centre consistently in one place."""
    row = {
        "frame": int(frame),
        "timestamp": round(float(timestamp), 4),
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": round(float(conf), 4),
        "x1": round(float(x1), 2),
        "y1": round(float(y1), 2),
        "x2": round(float(x2), 2),
        "y2": round(float(y2), 2),
        "center_x": round((float(x1) + float(x2)) / 2.0, 2),
        "center_y": round((float(y1) + float(y2)) / 2.0, 2),
    }
    if track_id is not None:
        row["track_id"] = int(track_id)
    return row
