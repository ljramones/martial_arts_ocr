# utils/image/regions/filters.py
"""
Region filtering and selection utilities (no heavy deps).

Responsibilities
- Size/aspect ratio filters
- Area/edge constraints (min/max)
- Overlap/Iou-based deduplication (simple NMS-lite)
- Convenience sorters for reading-order or area

Notes
- Operates on ImageRegion (from core_types).
- Keeps math simple and dependency-free (no NumPy required).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

try:
    # new path
    from utils.image.regions.core_types import ImageRegion, _Box
except Exception:
    # compatibility with pre-refactor paths
    from core_image import ImageRegion, _Box  # type: ignore

__all__ = [
    "filter_regions_by_size",
    "filter_by_aspect_ratio",
    "filter_by_area",
    "min_area",
    "max_area",
    "dedupe_overlaps",
    "nms",
    "sort_top_left",
    "sort_reading_order_like",
]


# ------------------------------ basic helpers ---------------------------------

def _aspect(w: int, h: int) -> float:
    return (w / float(h)) if h else 0.0


def _iou(a: ImageRegion, b: ImageRegion) -> float:
    return a.iou(b)


def _tl_key(r: ImageRegion) -> Tuple[int, int]:
    # top-left priority
    return (r.y1, r.x1)


def _reading_order_key(r: ImageRegion, y_tolerance: int = 8) -> Tuple[int, int]:
    # "row major" with tolerance for small y jitter: bucket y into bands
    band = r.y1 // max(1, y_tolerance)
    return (band, r.x1)


# ------------------------------- size filters ---------------------------------

def filter_regions_by_size(
    regions: Sequence[ImageRegion],
    *,
    min_width: int = 1,
    min_height: int = 1,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> List[ImageRegion]:
    """
    Keep regions whose width/height fall within [min, max] bounds.
    """
    out: List[ImageRegion] = []
    for r in regions:
        if r.width < min_width or r.height < min_height:
            continue
        if max_width is not None and r.width > max_width:
            continue
        if max_height is not None and r.height > max_height:
            continue
        out.append(r)
    return out


def filter_by_aspect_ratio(
    regions: Sequence[ImageRegion],
    *,
    min_ratio: float = 0.0,
    max_ratio: float = 1e9,
) -> List[ImageRegion]:
    """
    Keep regions whose aspect ratio (w/h) is within [min_ratio, max_ratio].
    """
    out: List[ImageRegion] = []
    for r in regions:
        ar = _aspect(r.width, r.height)
        if ar < min_ratio or ar > max_ratio:
            continue
        out.append(r)
    return out


def filter_by_area(
    regions: Sequence[ImageRegion],
    *,
    min_area_px: int = 1,
    max_area_px: Optional[int] = None,
) -> List[ImageRegion]:
    """
    Keep regions whose pixel area is within [min_area_px, max_area_px].
    """
    out: List[ImageRegion] = []
    for r in regions:
        a = r.area
        if a < min_area_px:
            continue
        if max_area_px is not None and a > max_area_px:
            continue
        out.append(r)
    return out


# Short-hands
def min_area(regions: Sequence[ImageRegion], px: int) -> List[ImageRegion]:
    return filter_by_area(regions, min_area_px=px)


def max_area(regions: Sequence[ImageRegion], px: int) -> List[ImageRegion]:
    return filter_by_area(regions, min_area_px=1, max_area_px=px)


# --------------------------- overlap / deduplication --------------------------

def dedupe_overlaps(
    regions: Sequence[ImageRegion],
    *,
    iou_threshold: float = 0.7,
    prefer: Callable[[ImageRegion, ImageRegion], ImageRegion] | None = None,
) -> List[ImageRegion]:
    """
    Merge-away near-duplicates: if two regions overlap with IoU >= threshold,
    keep only one. By default, keeps the larger area; pass `prefer(a,b)` to
    choose (e.g., by score).

    Args:
      iou_threshold: 0..1; higher is stricter for considering duplicates.
      prefer: optional function returning the preferred region between (a,b).

    Returns:
      Deduplicated list.
    """
    if iou_threshold <= 0:
        return list(regions)

    # simple greedy pass: sort by area desc, drop any that collide with kept
    regs = sorted(regions, key=lambda r: r.area, reverse=True)

    keep: List[ImageRegion] = []
    for r in regs:
        duplicate = False
        for k in keep:
            if _iou(r, k) >= iou_threshold:
                # resolve preference
                if prefer is None:
                    # default: keep the already-kept (larger area due to sort)
                    duplicate = True
                    break
                else:
                    chosen = prefer(k, r)
                    if chosen is k:
                        duplicate = True
                        break
                    else:
                        # replace kept with r
                        keep.remove(k)
                        keep.append(r)
                        duplicate = True
                        break
        if not duplicate:
            keep.append(r)

    return keep


def nms(
    regions: Sequence[ImageRegion],
    *,
    iou_threshold: float = 0.5,
    score_getter: Callable[[ImageRegion], float] | None = None,
) -> List[ImageRegion]:
    """
    Non-Maximum Suppression (NMS) style selection based on IoU & score.

    - Sorts regions by score desc (or area if score not provided).
    - Picks the top region, removes all remaining with IoU >= threshold,
      and repeats.

    Args:
      iou_threshold: overlap threshold for suppression.
      score_getter: function mapping region -> score (float). If None, uses area.

    Returns:
      List of selected (kept) regions.
    """
    if not regions:
        return []

    if score_getter is None:
        score_getter = lambda r: r.score if (r.score is not None) else float(r.area)

    regs = sorted(regions, key=score_getter, reverse=True)
    kept: List[ImageRegion] = []

    while regs:
        best = regs.pop(0)
        kept.append(best)
        survivors: List[ImageRegion] = []
        for r in regs:
            if _iou(best, r) < iou_threshold:
                survivors.append(r)
        regs = survivors

    return kept


# --------------------------------- sorters ------------------------------------

def sort_top_left(regions: Sequence[ImageRegion]) -> List[ImageRegion]:
    """
    Sort by y, then x (top-left first).
    """
    return sorted(regions, key=_tl_key)


def sort_reading_order_like(
    regions: Sequence[ImageRegion],
    *,
    y_tolerance: int = 8,
) -> List[ImageRegion]:
    """
    Sort approximately as a Western reading order: top-to-bottom, left-to-right.
    Rows are formed by bucketing y with a small tolerance, then sorting within
    rows by x.

    Args:
      y_tolerance: vertical jitter in pixels to consider items on the same line.
    """
    return sorted(regions, key=lambda r: _reading_order_key(r, y_tolerance=y_tolerance))
