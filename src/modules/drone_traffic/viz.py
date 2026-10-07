"""OpenCV drawing helpers: per-class boxes, labels and the two-part vehicle counter."""
from __future__ import annotations

import cv2

from visdrone_classes import color_for

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Overlays are authored for ~1280px-wide footage. Everything below scales
# relative to that, so a 426x240 clip does not end up with a legend covering
# half the frame (and 4K footage does not get unreadably tiny text).
REF_WIDTH = 1280.0


def ui_scale(img, lo=0.45, hi=1.4):
    """Overlay scale factor derived from frame width, clamped to sane bounds."""
    return max(lo, min(hi, img.shape[1] / REF_WIDTH))


def draw_box(img, x1, y1, x2, y2, class_id, label, thickness=None, font_scale=None):
    """Draw one detection/track box with a filled label chip above it."""
    s = ui_scale(img)
    if font_scale is None:
        font_scale = 0.45 * s
    if thickness is None:
        thickness = max(1, int(round(2 * s)))
    c = color_for(int(class_id))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)

    (tw, th), base = cv2.getTextSize(label, FONT, font_scale, 1)
    # Keep the chip inside the frame: shift up/down at the top edge, and pull it
    # left when the box sits near the right edge, so the text is never clipped.
    ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 6
    tx = min(x1, img.shape[1] - tw - 6)
    tx = max(0, tx)
    cv2.rectangle(img, (tx, ty - th - base), (tx + tw + 4, ty + base), c, -1)
    cv2.putText(img, label, (tx + 2, ty), FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_counter(img, current_counts, unique_counts, names, origin=(10, 10),
                 title_current="IN FRAME", title_unique="UNIQUE", max_rows=8):
    """Draw the two-column vehicle counter.

    The two metrics are deliberately kept separate and labelled:
      * current_counts : vehicles visible in THIS frame
      * unique_counts  : distinct track IDs seen since the video started
    They are different numbers and must never be conflated.

    Only relevant classes are listed. A fine-tuned 5-class model shows all of
    them; a general 80-class COCO model would otherwise paint a legend taller
    than the frame, so there we show only classes actually seen (busiest first).
    """
    if len(names) <= max_rows:
        shown = sorted(names)
    else:
        seen = {c for c, v in current_counts.items() if v} | \
               {c for c, v in unique_counts.items() if v}
        shown = sorted(seen, key=lambda c: (-int(unique_counts.get(c, 0)),
                                            -int(current_counts.get(c, 0)), c))[:max_rows]

    s = ui_scale(img)
    fs = 0.42 * s
    pad = max(3, int(round(8 * s)))
    line_h = max(9, int(round(18 * s)))
    # Width is driven by the widest class name so nothing is clipped.
    name_w = max([len(str(names.get(c, c))) for c in shown] + [len("class")])
    col = f"{{:<{name_w + 2}}}{{:>9}}{{:>9}}"
    probe = col.format("class", title_current, title_unique)
    (tw, _), _ = cv2.getTextSize(probe, FONT, fs, 1)
    box_w = tw + pad * 2 + 6
    box_h = pad * 2 + line_h * (len(shown) + 2)
    x0, y0 = origin

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)

    y = y0 + pad + int(round(12 * s))
    cv2.putText(img, probe, (x0 + pad, y), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
    y += line_h
    cv2.line(img, (x0 + pad, y - int(round(10 * s))),
             (x0 + box_w - pad, y - int(round(10 * s))), (150, 150, 150), 1)

    for cid in shown:
        nm = str(names.get(cid, cid))
        cur = int(current_counts.get(cid, 0))
        uni = int(unique_counts.get(cid, 0))
        cv2.putText(img, col.format(nm, cur, uni), (x0 + pad, y),
                    FONT, fs, color_for(cid), 1, cv2.LINE_AA)
        y += line_h

    # Totals are over ALL classes, not just the rows displayed.
    tot_c = sum(int(v) for v in current_counts.values())
    tot_u = sum(int(v) for v in unique_counts.values())
    cv2.line(img, (x0 + pad, y - int(round(12 * s))),
             (x0 + box_w - pad, y - int(round(12 * s))), (150, 150, 150), 1)
    cv2.putText(img, col.format("TOTAL", tot_c, tot_u), (x0 + pad, y + 2),
                FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_hud(img, text, origin=(10, None), font_scale=None):
    """Bottom-left status line (frame number, fps, model)."""
    if font_scale is None:
        font_scale = 0.5 * ui_scale(img)
    h = img.shape[0]
    x, y = origin[0], (origin[1] if origin[1] is not None else h - 10)
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, 1)
    cv2.rectangle(img, (x - 4, y - th - base - 4), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 2), FONT, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
    return img
