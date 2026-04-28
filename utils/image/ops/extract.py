# utils/image/ops/extract.py
"""
Region extraction utilities for the Martial Arts OCR system.

Responsibilities:
- Extract crops from rectangular (bbox) ImageRegion objects
- Optional padding around regions
- Safe clipping to image bounds (or strict bounds with errors)
- Batch extraction
- Polygonal extraction (mask-based) when region has .points
- Helpers to clamp/transform regions and convert to ndarray slices

Types:
- ImageRegion: provided by utils.image.regions.core_image (must expose .bbox and optional .points)
  .bbox is expected as (x1, y1, x2, y2) in pixel coordinates, inclusive-exclusive safe.

Conventions:
- Images are NumPy ndarrays: GRAY (H×W) or color (H×W×C).
- Coordinates are integer pixels; we clamp to valid bounds when safe_crop=True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from utils.image.regions.core_image import ImageRegion  # expects .bbox and optional .region_type/.points

logger = logging.getLogger(__name__)

__all__ = [
    "extract_region",
    "extract_many",
    "clamp_bbox",
    "region_to_slice",
    "expand_bbox",
    "translate_region",
    "scale_region",
    "extract_polygon_region",
    "save_region_crops",
]

BBox = Tuple[int, int, int, int]


# ------------------------------- core helpers ---------------------------------

def _img_hw(img: np.ndarray) -> Tuple[int, int]:
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError(f"extract expects a NumPy image, got shape={getattr(img, 'shape', None)}")
    h, w = img.shape[:2]
    return h, w


def clamp_bbox(bbox: BBox, img_w: int, img_h: int) -> BBox:
    """Clamp (x1,y1,x2,y2) to image bounds [0,w]×[0,h] with x2>=x1, y2>=y1."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), img_w))
    y1 = max(0, min(int(y1), img_h))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, pad: int) -> BBox:
    """Uniformly expand bbox by pad pixels on all sides (may go negative or beyond)."""
    x1, y1, x2, y2 = bbox
    p = int(max(0, pad))
    return x1 - p, y1 - p, x2 + p, y2 + p


def region_to_slice(bbox: BBox) -> Tuple[slice, slice]:
    """Convert bbox (x1,y1,x2,y2) to numpy [y1:y2, x1:x2] slices."""
    x1, y1, x2, y2 = bbox
    return slice(y1, y2), slice(x1, x2)


def translate_region(region: ImageRegion, dx: int, dy: int) -> ImageRegion:
    """Return a shallow-copied region translated by (dx,dy)."""
    x1, y1, x2, y2 = region.bbox
    new_bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    if hasattr(region, "points") and region.points is not None:
        pts = np.asarray(region.points, dtype=np.float32) + np.array([dx, dy], dtype=np.float32)
        region = region.__class__(bbox=new_bbox, region_type=getattr(region, "region_type", None), points=pts.tolist())
    else:
        region = region.__class__(bbox=new_bbox, region_type=getattr(region, "region_type", None))
    return region


def scale_region(region: ImageRegion, sx: float, sy: float | None = None) -> ImageRegion:
    """Scale region about the origin (0,0). Useful after image resizes."""
    sy = sx if sy is None else sy
    x1, y1, x2, y2 = region.bbox
    new_bbox = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
    if hasattr(region, "points") and region.points is not None:
        pts = np.asarray(region.points, dtype=np.float32)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        region = region.__class__(bbox=new_bbox, region_type=getattr(region, "region_type", None), points=pts.tolist())
    else:
        region = region.__class__(bbox=new_bbox, region_type=getattr(region, "region_type", None))
    return region


# -------------------------------- extraction ----------------------------------

def extract_region(
    image: np.ndarray,
    region: ImageRegion,
    *,
    padding: int = 0,
    safe_crop: bool = True,
) -> np.ndarray:
    """
    Extract a rectangular crop from image using region.bbox with optional padding.

    Args:
        image: ndarray (H×W or H×W×C)
        region: ImageRegion with .bbox (x1,y1,x2,y2)
        padding: extra pixels around the region (uniform)
        safe_crop: clamp to image bounds; if False and bbox exceeds bounds -> ValueError

    Returns:
        Cropped ndarray; returns a minimal 1×1 array if the clamp collapses to empty.
    """
    h, w = _img_hw(image)
    x1, y1, x2, y2 = region.bbox
    if padding > 0:
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), padding)

    if safe_crop:
        x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), w, h)
    else:
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(f"Region {region.bbox} exceeds image bounds {w}x{h} (padding={padding})")

    ys, xs = region_to_slice((x1, y1, x2, y2))
    out = image[ys, xs]
    if out.size == 0:
        logger.warning("extract_region: empty crop for %s (after clamp). Returning 1×1 tile.", getattr(region, "region_type", "region"))
        ch = image.shape[2] if image.ndim == 3 else 1
        return np.zeros((1, 1, ch), dtype=image.dtype) if ch > 1 else np.zeros((1, 1), dtype=image.dtype)
    return out


def extract_many(
    image: np.ndarray,
    regions: Sequence[ImageRegion],
    *,
    padding: int = 0,
    safe_crop: bool = True,
) -> List[np.ndarray]:
    """
    Batch extract crops for a list/tuple of regions.
    """
    crops: List[np.ndarray] = []
    for r in regions:
        try:
            crop = extract_region(image, r, padding=padding, safe_crop=safe_crop)
        except Exception as e:
            logger.error("extract_many: failed on %s: %s", getattr(r, "region_type", "region"), e)
            # keep pipeline moving with an empty tile
            ch = image.shape[2] if image.ndim == 3 else 1
            crop = np.zeros((1, 1, ch), dtype=image.dtype) if ch > 1 else np.zeros((1, 1), dtype=image.dtype)
        crops.append(crop)
    return crops


def save_region_crops(
    image: np.ndarray,
    regions: Sequence[ImageRegion],
    output_dir: str | Path,
    *,
    prefix: str = "region",
    padding: int = 0,
    safe_crop: bool = True,
) -> List[dict]:
    """Extract and save region crops, returning stable metadata records."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: List[dict] = []
    for index, region in enumerate(regions, start=1):
        crop = extract_region(image, region, padding=padding, safe_crop=safe_crop)
        filename = f"{prefix}_{index:03d}.png"
        crop_path = output_path / filename
        if not cv2.imwrite(str(crop_path), crop):
            raise IOError(f"Failed to write crop: {crop_path}")

        saved.append(
            {
                "region_id": getattr(region, "id", None) or f"{prefix}_{index:03d}",
                "image_path": str(crop_path),
                "region": region.to_dict() if hasattr(region, "to_dict") else {"bbox": region.bbox},
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "reading_order": index,
            }
        )
    return saved


# ------------------------------ polygon support -------------------------------

def _polygon_mask(points: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Create a single-channel uint8 mask with filled polygon (255 inside).
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return mask
    pts = np.round(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_polygon_region(
    image: np.ndarray,
    region: ImageRegion,
    *,
    padding: int = 0,
    safe_crop: bool = True,
    return_masked: bool = True,
    bg_value: int | Tuple[int, int, int] = 0,
) -> np.ndarray:
    """
    Extract a polygonal region when `region.points` is provided.
    - We compute a tight bbox around the polygon (+ padding),
      crop, then mask outside the polygon.
    - If return_masked=True, pixels outside polygon are set to bg_value.
      Otherwise, we return the tight rectangular crop without masking.

    Notes:
      - Points should be (x,y) in image coords.
      - If points are missing or invalid, falls back to bbox extraction.
    """
    if not hasattr(region, "points") or region.points is None:
        logger.debug("extract_polygon_region: region has no points; falling back to bbox.")
        return extract_region(image, region, padding=padding, safe_crop=safe_crop)

    pts = np.asarray(region.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        logger.debug("extract_polygon_region: invalid points; falling back to bbox.")
        return extract_region(image, region, padding=padding, safe_crop=safe_crop)

    # Build bbox around polygon
    x_min = int(np.floor(pts[:, 0].min()))
    y_min = int(np.floor(pts[:, 1].min()))
    x_max = int(np.ceil(pts[:, 0].max()))
    y_max = int(np.ceil(pts[:, 1].max()))
    bbox = (x_min, y_min, x_max, y_max)
    if padding > 0:
        bbox = expand_bbox(bbox, padding)

    h, w = _img_hw(image)
    if safe_crop:
        bbox = clamp_bbox(bbox, w, h)
    else:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(f"Polygon bbox exceeds image bounds {w}x{h} (padding={padding})")

    y_slice, x_slice = region_to_slice(bbox)
    crop = image[y_slice, x_slice]
    if not return_masked:
        return crop

    # Shift points into crop-local coordinates and build mask
    x1, y1, x2, y2 = bbox
    local_pts = pts - np.array([x1, y1], dtype=np.float32)
    mask = _polygon_mask(local_pts, (crop.shape[0], crop.shape[1]))

    if crop.ndim == 2:
        out = crop.copy()
        out[mask == 0] = bg_value if isinstance(bg_value, (int, np.integer)) else int(bg_value[0])
        return out

    # Color
    out = crop.copy()
    if isinstance(bg_value, tuple):
        bg_tuple = tuple(int(v) for v in bg_value)
    else:
        bg_tuple = (int(bg_value), int(bg_value), int(bg_value))

    inv = (mask == 0)
    out[inv] = bg_tuple
    return out
