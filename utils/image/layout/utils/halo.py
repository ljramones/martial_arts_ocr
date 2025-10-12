# utils/image/layout/utils/halo.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def _expand_clamped(x: int, y: int, w: int, h: int,
                    ring: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """
    Expand (x,y,w,h) by 'ring' pixels on all sides, clamped to image bounds.
    Returns (xe0, ye0, xe1, ye1) as inclusive-exclusive coords (x0,y0,x1,y1).
    """
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    xe0 = max(0, x0 - ring)
    ye0 = max(0, y0 - ring)
    xe1 = min(W, x1 + ring)
    ye1 = min(H, y1 + ring)
    return xe0, ye0, xe1, ye1


def halo_ratio(gray: np.ndarray,
               x: int, y: int, w: int, h: int,
               ring: int = 4,
               white_thresh: int = 200) -> float:
    """
    Compute the ratio of 'white' pixels in the 1-ring halo around (x,y,w,h).
    white_thresh: intensity threshold (uint8) to consider a pixel white.

    Returns:
        float in [0..1]; 0.0 if the halo band is empty/invalid.
    """
    # Ensure grayscale uint8
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    H, W = gray.shape[:2]
    # clamp inner box and build expanded region
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    xe0, ye0, xe1, ye1 = _expand_clamped(x0, y0, x1 - x0, y1 - y0, ring, W, H)
    if xe1 <= xe0 or ye1 <= ye0:
        return 0.0

    outer = gray[ye0:ye1, xe0:xe1]
    if outer.size == 0:
        return 0.0

    # carve out inner box within 'outer' to form the halo band
    halo_mask = np.ones_like(outer, dtype=bool)
    iy0, ix0 = y0 - ye0, x0 - xe0
    iy1, ix1 = iy0 + (y1 - y0), ix0 + (x1 - x0)
    # guard against rounding/bounds
    iy0 = max(0, min(iy0, outer.shape[0])); iy1 = max(0, min(iy1, outer.shape[0]))
    ix0 = max(0, min(ix0, outer.shape[1])); ix1 = max(0, min(ix1, outer.shape[1]))
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0

    halo_mask[iy0:iy1, ix0:ix1] = False
    band = outer[halo_mask]
    if band.size == 0:
        return 0.0

    return float(np.mean(band > white_thresh))


def halo_ok(gray: np.ndarray,
            x: int, y: int, w: int, h: int,
            ring: int = 4,
            min_white: float = 0.85,
            white_thresh: int = 200) -> bool:
    """
    Return True if the halo around (x,y,w,h) is mostly white.

    Args:
        gray: grayscale uint8 (or BGR) page
        x,y,w,h: candidate region
        ring: halo ring width in pixels (default 4)
        min_white: required white ratio in [0..1] (default 0.85)
        white_thresh: uint8 intensity threshold to consider a pixel white (default 200)

    Usage:
        if halo_ok(gray, x,y,w,h, ring=cfg['halo_ring'], min_white=cfg['halo_min_white']):
            ... accept region ...
    """
    ratio = halo_ratio(gray, x, y, w, h, ring=ring, white_thresh=white_thresh)
    return ratio >= float(min_white)
