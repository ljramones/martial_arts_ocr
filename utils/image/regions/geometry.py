# utils/image/regions/geometry.py
"""
Geometry helpers for regions and bounding boxes (no heavy deps).

Responsibilities
- Pure bbox math on tuples: size, center, union, intersection, IoU/IoA
- Transforms: translate, scale, clamp, expand/shrink, normalize
- Distance/overlap utilities (center distance, axis overlaps)
- Aspect helpers: grow to aspect, fit into another box
- Thin ImageRegion wrappers for the above

Types
- BBox := (x1, y1, x2, y2) with x2 >= x1, y2 >= y1 (inclusive-exclusive safe)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional, Tuple

try:
    # New path (post-refactor)
    from utils.image.regions.core_types import BBox, ImageRegion, _Box
except Exception:  # pragma: no cover - back-compat
    from core_image import BBox, ImageRegion, _Box  # type: ignore

__all__ = [
    # bbox core
    "normalize_bbox", "bbox_width", "bbox_height", "bbox_area", "bbox_center",
    "bbox_intersection", "bbox_union", "bbox_iou", "bbox_ioa",
    # transforms
    "translate_bbox", "scale_bbox", "expand_bbox", "shrink_bbox", "clamp_bbox",
    "inflate_bbox_percent", "snap_bbox_to_grid",
    # relationships
    "center_distance", "horizontal_overlap", "vertical_overlap",
    "x_overlap_ratio", "y_overlap_ratio",
    # aspect/fit
    "grow_to_aspect", "fit_bbox_into",
    # ImageRegion facades
    "expand_region", "shrink_region", "translate_region", "scale_region",
    "clamp_region", "grow_region_to_aspect",
]

# ---------------------------- core bbox utilities -----------------------------

def _as_box(bb: BBox) -> _Box:
    return _Box(bb[0], bb[1], bb[2], bb[3])

def normalize_bbox(bbox: Iterable[int | float]) -> BBox:
    """Order coordinates so x2>=x1, y2>=y1, return ints."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def bbox_width(bb: BBox) -> int:
    return max(0, int(bb[2]) - int(bb[0]))

def bbox_height(bb: BBox) -> int:
    return max(0, int(bb[3]) - int(bb[1]))

def bbox_area(bb: BBox) -> int:
    return bbox_width(bb) * bbox_height(bb)

def bbox_center(bb: BBox) -> Tuple[float, float]:
    return (bb[0] + bbox_width(bb) / 2.0, bb[1] + bbox_height(bb) / 2.0)

def bbox_intersection(a: BBox, b: BBox) -> BBox:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)

def bbox_union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def bbox_iou(a: BBox, b: BBox) -> float:
    inter = bbox_area(bbox_intersection(a, b))
    if inter == 0:
        return 0.0
    denom = float(bbox_area(a) + bbox_area(b) - inter) or 1.0
    return inter / denom

def bbox_ioa(a: BBox, b: BBox) -> Tuple[float, float]:
    """
    Intersection over area of a and b respectively.
    Returns (IoA(a), IoA(b)).
    """
    inter = bbox_area(bbox_intersection(a, b))
    aa = float(bbox_area(a)) or 1.0
    bb = float(bbox_area(b)) or 1.0
    return (inter / aa, inter / bb)

# ------------------------------- transforms -----------------------------------

def translate_bbox(bb: BBox, dx: int, dy: int) -> BBox:
    return (bb[0] + dx, bb[1] + dy, bb[2] + dx, bb[3] + dy)

def scale_bbox(bb: BBox, sx: float, sy: Optional[float] = None) -> BBox:
    sy = sx if sy is None else sy
    return (
        int(round(bb[0] * sx)), int(round(bb[1] * sy)),
        int(round(bb[2] * sx)), int(round(bb[3] * sy)),
    )

def expand_bbox(bb: BBox, pad: int) -> BBox:
    p = max(0, int(pad))
    return (bb[0] - p, bb[1] - p, bb[2] + p, bb[3] + p)

def shrink_bbox(bb: BBox, pad: int) -> BBox:
    p = max(0, int(pad))
    return (bb[0] + p, bb[1] + p, bb[2] - p, bb[3] - p)

def clamp_bbox(bb: BBox, img_w: int, img_h: int) -> BBox:
    x1 = max(0, min(bb[0], img_w)); y1 = max(0, min(bb[1], img_h))
    x2 = max(0, min(bb[2], img_w)); y2 = max(0, min(bb[3], img_h))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return (x1, y1, x2, y2)

def inflate_bbox_percent(bb: BBox, px: float) -> BBox:
    """
    Expand by a percentage of the current size (e.g., px=0.1 grows by 10% each side).
    """
    w = bbox_width(bb); h = bbox_height(bb)
    dx = int(round(w * px)); dy = int(round(h * px))
    return (bb[0] - dx, bb[1] - dy, bb[2] + dx, bb[3] + dy)

def snap_bbox_to_grid(bb: BBox, step: int) -> BBox:
    """Snap bbox edges to nearest multiples of step (>=1)."""
    s = max(1, int(step))
    from math import floor, ceil
    x1 = s * int(floor(bb[0] / s)); y1 = s * int(floor(bb[1] / s))
    x2 = s * int(ceil(bb[2] / s));  y2 = s * int(ceil(bb[3] / s))
    return (x1, y1, x2, y2)

# ---------------------------- relationships/overlap ---------------------------

def _axis_overlap(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, min(a2, b2) - max(a1, b1))

def horizontal_overlap(a: BBox, b: BBox) -> int:
    """Overlap length along X."""
    return _axis_overlap(a[0], a[2], b[0], b[2])

def vertical_overlap(a: BBox, b: BBox) -> int:
    """Overlap length along Y."""
    return _axis_overlap(a[1], a[3], b[1], b[3])

def x_overlap_ratio(a: BBox, b: BBox) -> float:
    """Horizontal overlap relative to min width of (a,b)."""
    ov = float(horizontal_overlap(a, b))
    denom = float(min(bbox_width(a), bbox_width(b))) or 1.0
    return ov / denom

def y_overlap_ratio(a: BBox, b: BBox) -> float:
    """Vertical overlap relative to min height of (a,b)."""
    ov = float(vertical_overlap(a, b))
    denom = float(min(bbox_height(a), bbox_height(b))) or 1.0
    return ov / denom

def center_distance(a: BBox, b: BBox) -> float:
    """Euclidean distance between centers (float)."""
    from math import hypot
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return hypot(bx - ax, by - ay)

# ------------------------------- aspect / fit ---------------------------------

def _safe_div(x: float, y: float) -> float:
    return x / y if y else 0.0

def _aspect(w: int, h: int) -> float:
    return _safe_div(w, float(h))

def grow_to_aspect(bb: BBox, target_ar: float, *, anchor: str = "center") -> BBox:
    """
    Grow (never shrink) bb to match an aspect ratio (w/h) by expanding the
    smaller dimension. Anchor controls which side remains fixed:
      - "center" (default), "left", "right", "top", "bottom".
    """
    x1, y1, x2, y2 = bb
    w, h = bbox_width(bb), bbox_height(bb)
    if w == 0 or h == 0:
        return bb
    ar = _aspect(w, h)
    if abs(ar - target_ar) < 1e-9:
        return bb

    if ar < target_ar:
        # too tall/narrow -> widen
        new_w = int(round(h * target_ar))
        delta = new_w - w
        if anchor == "left":
            return (x1, y1, x2 + delta, y2)
        elif anchor == "right":
            return (x1 - delta, y1, x2, y2)
        else:  # center/top/bottom -> center horizontally
            dx = delta // 2
            return (x1 - dx, y1, x2 + (delta - dx), y2)
    else:
        # too wide/short -> heighten
        new_h = int(round(w / target_ar))
        delta = new_h - h
        if anchor == "top":
            return (x1, y1, x2, y2 + delta)
        elif anchor == "bottom":
            return (x1, y1 - delta, x2, y2)
        else:  # center/left/right -> center vertically
            dy = delta // 2
            return (x1, y1 - dy, x2, y2 + (delta - dy))

def fit_bbox_into(src: BBox, dst: BBox, *, contain: bool = True) -> BBox:
    """
    Scale src to fit into dst while preserving aspect.
    - contain=True  : entire src fits inside dst (like 'fit/letterbox' without pad)
    - contain=False : src covers dst (like 'fill/cover'), then we return the src box in dst space
    The returned bbox is positioned inside dst with center alignment.
    """
    sw, sh = bbox_width(src), bbox_height(src)
    dw, dh = bbox_width(dst), bbox_height(dst)
    if sw == 0 or sh == 0 or dw == 0 or dh == 0:
        return dst

    if contain:
        scale = min(dw / sw, dh / sh)
    else:
        scale = max(dw / sw, dh / sh)

    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    dx = (dw - nw) // 2
    dy = (dh - nh) // 2

    # place inside dst (offset by its top-left)
    return (dst[0] + dx, dst[1] + dy, dst[0] + dx + nw, dst[1] + dy + nh)

# --------------------------- ImageRegion facades ------------------------------

def _with_bbox(r: ImageRegion, bb: BBox) -> ImageRegion:
    return replace(r, bbox=bb)

def expand_region(r: ImageRegion, pad: int) -> ImageRegion:
    return _with_bbox(r, expand_bbox(r.to_tuple(), pad))

def shrink_region(r: ImageRegion, pad: int) -> ImageRegion:
    return _with_bbox(r, shrink_bbox(r.to_tuple(), pad))

def translate_region(r: ImageRegion, dx: int, dy: int) -> ImageRegion:
    return _with_bbox(r, translate_bbox(r.to_tuple(), dx, dy))

def scale_region(r: ImageRegion, sx: float, sy: Optional[float] = None) -> ImageRegion:
    return _with_bbox(r, scale_bbox(r.to_tuple(), sx, sy))

def clamp_region(r: ImageRegion, img_w: int, img_h: int) -> ImageRegion:
    return _with_bbox(r, clamp_bbox(r.to_tuple(), img_w, img_h))

def grow_region_to_aspect(r: ImageRegion, target_ar: float, *, anchor: str = "center") -> ImageRegion:
    return _with_bbox(r, grow_to_aspect(r.to_tuple(), target_ar, anchor=anchor))
