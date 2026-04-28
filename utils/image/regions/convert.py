# utils/image/regions/convert.py
"""
Adapters between core region types.

Responsibilities
- Convert between:
    * tuple BBox (x1, y1, x2, y2)
    * internal _Box (box class)
    * ImageRegion (dataclass with .bbox and optional .points)
- Provide small helpers to sanitize/normalize bbox tuples.
- Keep imports stable across the refactor (compat shims included).

Conventions
- BBox tuples are integer pixel coords in image space.
- We enforce x2 >= x1 and y2 >= y1 to avoid negative sizes.
- ImageRegion may optionally carry polygon points; these pass through unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Tuple, Union

# --- Import the core types with a small compatibility shim --------------------
try:
    # New structure
    from utils.image.regions.core_types import ImageRegion, _Box  # type: ignore
except Exception:
    # Back-compat with pre-refactor paths
    try:
        from utils.image.regions.core_image import ImageRegion, _Box  # type: ignore
    except Exception:
        # Last resort (legacy single-file module name)
        from core_image import ImageRegion, _Box  # type: ignore

BBox = Tuple[int, int, int, int]
FBox = Tuple[float, float, float, float]

__all__ = [
    "BBox",
    "FBox",
    "normalize_bbox",
    "to_int_bbox",
    "to_float_bbox",
    "region_to_box",
    "region_to_bbox",
    "box_to_region",
    "bbox_to_region",
]


# ----------------------------- bbox sanitizers --------------------------------

def normalize_bbox(bbox: Iterable[Union[int, float]]) -> FBox:
    """
    Ensure bbox=(x1,y1,x2,y2) is well-ordered (x2>=x1, y2>=y1) in float space.
    """
    x1, y1, x2, y2 = map(float, bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def to_int_bbox(bbox: Iterable[Union[int, float]]) -> BBox:
    """
    Convert any numeric iterable to an ordered integer bbox (floor for mins, ceil for maxes).
    Useful when upstream code uses float coords.
    """
    x1, y1, x2, y2 = normalize_bbox(bbox)
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def to_float_bbox(bbox: Iterable[Union[int, float]]) -> FBox:
    """
    Convert to an ordered float bbox (no rounding).
    """
    return normalize_bbox(bbox)


# ------------------------------ region -> box ---------------------------------

def region_to_box(region: ImageRegion) -> _Box:
    """
    Convert an ImageRegion into an internal _Box instance.
    """
    x1, y1, x2, y2 = region.bbox
    return _Box(x1=x1, y1=y1, x2=x2, y2=y2)  # type: ignore[call-arg]


def region_to_bbox(region: ImageRegion, *, as_int: bool = True) -> Union[BBox, FBox]:
    """
    Return the region bbox as a tuple. (x1, y1, x2, y2)
    """
    bbox = (region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3])
    return to_int_bbox(bbox) if as_int else to_float_bbox(bbox)


# ------------------------------ box -> region ---------------------------------

def box_to_region(
    box: _Box,
    *,
    region_type: Optional[str] = None,
    points: Optional[List[Tuple[float, float]]] = None,
) -> ImageRegion:
    """
    Convert an internal _Box into an ImageRegion.

    Args:
        box: internal box (_Box) with x1,y1,x2,y2 members.
        region_type: optional semantic label (e.g., "line", "word", "title").
        points: optional polygon (list of (x,y) float) to attach to the region.

    Returns:
        ImageRegion with .bbox set from the box and .points/region_type as provided.
    """
    bbox = to_int_bbox((box.x1, box.y1, box.x2, box.y2))
    try:
        # Prefer constructor signature (dataclass style)
        return ImageRegion(bbox=bbox, region_type=region_type, points=points)  # type: ignore[call-arg]
    except TypeError:
        # Older ImageRegion without 'points' support
        return ImageRegion(bbox=bbox, region_type=region_type)  # type: ignore[call-arg]


def bbox_to_region(
    bbox: Iterable[Union[int, float]],
    *,
    region_type: Optional[str] = None,
    points: Optional[List[Tuple[float, float]]] = None,
) -> ImageRegion:
    """
    Convert a tuple-like bbox into an ImageRegion.

    Args:
        bbox: iterable of 4 numbers (x1, y1, x2, y2). Order will be normalized.
        region_type: optional semantic label.
        points: optional polygon (list of (x,y) float) to attach to the region.
    """
    bb = to_int_bbox(bbox)
    try:
        return ImageRegion(bbox=bb, region_type=region_type, points=points)  # type: ignore[call-arg]
    except TypeError:
        return ImageRegion(bbox=bb, region_type=region_type)  # type: ignore[call-arg]
