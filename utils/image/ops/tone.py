# utils/image/ops/tone.py
"""
Tone & contrast utilities for the Martial Arts OCR system.

Goals:
- Stable, dtype-aware (uint8/uint16/float32) image tone operations
- OCR-friendly defaults (CLAHE, gentle gamma, percentile autocontrast)
- Safe behavior on grayscale (HxW) and color (HxWx3) arrays

Key APIs:
- to_gray(img)
- adjust_brightness_contrast(img, alpha=1.0, beta=0.0)
- gamma_correct(img, gamma=1.2)
- auto_contrast_gray(img, cutoff=1.0)
- rescale_intensity_gray(img, in_percent=(2,98))
- clahe_gray(img, clip=2.5, tile=(8,8))
- clahe_bgr_lab(img, clip=2.5, tile=(8,8))
- white_balance_grayworld_bgr(img)
- unsharp_mask_gray(img, radius=1.0, amount=1.0, threshold=0)

Conventions:
- Functions return same dtype as input unless noted.
- Color funcs expect BGR (OpenCV convention).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "to_gray",
    "adjust_brightness_contrast",
    "gamma_correct",
    "auto_contrast_gray",
    "rescale_intensity_gray",
    "clahe_gray",
    "clahe_bgr_lab",
    "white_balance_grayworld_bgr",
    "unsharp_mask_gray",
]

# ------------------------------ dtype helpers ---------------------------------

def _is_uint(img: np.ndarray) -> bool:
    return img.dtype in (np.uint8, np.uint16)

def _is_float(img: np.ndarray) -> bool:
    return img.dtype in (np.float32, np.float64)

def _range_for_dtype(dtype) -> Tuple[float, float]:
    if dtype == np.uint8:
        return 0.0, 255.0
    if dtype == np.uint16:
        return 0.0, 65535.0
    # floats are assumed 0..1
    return 0.0, 1.0

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for gray: {img.shape}")

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or GRAY to GRAY, preserving dtype."""
    return _ensure_gray(img)

# ------------------------- brightness / contrast ------------------------------

def adjust_brightness_contrast(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Linear tone: out = img * alpha + beta.
    - alpha > 1 increases contrast; 0<alpha<1 lowers contrast.
    - beta shifts brightness (same units as pixel range).
    Returns same dtype, clipping to valid range.
    """
    dtype = img.dtype
    lo, hi = _range_for_dtype(dtype)
    if not _is_float(img):
        work = img.astype(np.float32)
    else:
        work = img.astype(np.float32)

    work = work * float(alpha) + float(beta)
    work = np.clip(work, lo, hi)

    if dtype == np.uint8:
        return work.astype(np.uint8)
    if dtype == np.uint16:
        return work.astype(np.uint16)
    return work.astype(dtype)

# ------------------------------- gamma ----------------------------------------

def gamma_correct(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply power-law gamma correction:
      out = (img_norm ** (1/gamma)) * maxVal
    gamma>1 lightens shadows; gamma<1 darkens.
    Returns same dtype.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    dtype = img.dtype
    lo, hi = _range_for_dtype(dtype)

    if _is_float(img):
        norm = np.clip(img, lo, hi)
        maxv = 1.0
    else:
        norm = img.astype(np.float32) / hi
        maxv = 1.0

    out = np.power(norm, 1.0 / gamma)

    if _is_float(img):
        return out.astype(dtype)
    out = np.clip(out * hi, lo, hi)
    return out.astype(dtype)

# ----------------------------- auto-contrast ----------------------------------

def auto_contrast_gray(
    gray: np.ndarray,
    cutoff: float = 1.0,
) -> np.ndarray:
    """
    Percentile-based autocontrast for grayscale.
    - cutoff: percentage to clip from both low/high tails (0..20 recommended).
    Keeps dtype; robust for OCR scans with background tint.
    """
    g = _ensure_gray(gray)
    dtype = g.dtype
    lo_d, hi_d = _range_for_dtype(dtype)

    # Compute percentiles in float for stability
    if not _is_float(g):
        gf = g.astype(np.float32)
    else:
        gf = g

    low = np.percentile(gf, cutoff)
    high = np.percentile(gf, 100.0 - cutoff)
    if high <= low + 1e-6:
        return g  # avoid division by tiny/zero

    out = (gf - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)

    if dtype == np.uint8:
        return (out * 255.0 + 0.5).astype(np.uint8)
    if dtype == np.uint16:
        return (out * 65535.0 + 0.5).astype(np.uint16)
    return out.astype(dtype)

def rescale_intensity_gray(
    gray: np.ndarray,
    in_percent: Tuple[float, float] = (2.0, 98.0),
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Map grayscale intensities from percentile range -> out_range (float domain).
    Useful when you want a specific output range for subsequent ops.
    Returns float32.
    """
    g = _ensure_gray(gray)
    gf = g.astype(np.float32)
    p0 = np.percentile(gf, in_percent[0])
    p1 = np.percentile(gf, in_percent[1])
    if p1 <= p0 + 1e-6:
        return np.clip(gf, 0, None)  # noop, float32

    lo, hi = out_range
    out = (gf - p0) / (p1 - p0)
    out = np.clip(out, 0.0, 1.0)
    out = out * (hi - lo) + lo
    return out.astype(np.float32)

# --------------------------------- CLAHE --------------------------------------

def _mk_clahe(clip: float, tile: Tuple[int, int]) -> cv2.CLAHE:
    clip = max(0.0, float(clip))
    t = (max(1, int(tile[0])), max(1, int(tile[1])))
    return cv2.createCLAHE(clipLimit=clip if clip > 0 else 2.0, tileGridSize=t)

def clahe_gray(
    gray: np.ndarray,
    clip: float = 2.5,
    tile: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE on grayscale (preferred for text). Keeps dtype.
    """
    g = _ensure_gray(gray)
    clahe = _mk_clahe(clip, tile)

    if g.dtype == np.uint8:
        return clahe.apply(g)

    # For non-uint8, normalize to 8-bit, apply, then map back
    lo, hi = _range_for_dtype(g.dtype)
    norm = np.clip(g.astype(np.float32), lo, hi)
    norm = ((norm - lo) / (hi - lo) * 255.0).astype(np.uint8)
    eq = clahe.apply(norm)

    if g.dtype == np.uint16:
        back = (eq.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
        return back
    if _is_float(g):
        back = eq.astype(np.float32) / 255.0
        return back.astype(g.dtype)
    return eq.astype(g.dtype)

def clahe_bgr_lab(
    bgr: np.ndarray,
    clip: float = 2.5,
    tile: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE on L* channel in LAB, then convert back to BGR. Preserves dtype.
    Recommended for photos with uneven lighting before OCR.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("clahe_bgr_lab expects HxWx3 BGR")

    dtype = bgr.dtype
    lo, hi = _range_for_dtype(dtype)

    # Work in 8-bit LAB for CLAHE stability
    if dtype != np.uint8:
        work = np.clip(bgr.astype(np.float32), lo, hi)
        work = ((work - lo) / (hi - lo) * 255.0).astype(np.uint8)
        revert = True
    else:
        work = bgr
        revert = False

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = _mk_clahe(clip, tile).apply(L)
    lab_eq = cv2.merge([L, A, B])
    out8 = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    if not revert:
        return out8

    # Map back to original dtype range
    if dtype == np.uint16:
        return (out8.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
    if _is_float(bgr):
        return (out8.astype(np.float32) / 255.0).astype(dtype)
    return out8.astype(dtype)

# ----------------------------- white balance ----------------------------------

def white_balance_grayworld_bgr(bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Simple Gray-World white balance:
      scales channels so their means become equal.
    Useful to neutralize color cast before converting to gray.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("white_balance_grayworld_bgr expects HxWx3 BGR")

    dtype = bgr.dtype
    lo, hi = _range_for_dtype(dtype)

    work = bgr.astype(np.float32) if not _is_float(bgr) else bgr.astype(np.float32)
    means = work.reshape(-1, 3).mean(axis=0) + eps
    gray_mean = means.mean()
    scales = gray_mean / means  # 3 scalars

    out = work * scales
    out = np.clip(out, lo, hi)

    if dtype == np.uint8:
        return out.astype(np.uint8)
    if dtype == np.uint16:
        return out.astype(np.uint16)
    return out.astype(dtype)

# ------------------------------ local contrast --------------------------------

def unsharp_mask_gray(
    gray: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    """
    Unsharp mask for grayscale:
      out = gray + amount * (gray - blur(gray, radius))
    - radius: Gaussian sigma (px)
    - amount: strength multiplier
    - threshold: ignore small differences (< threshold in native units) to avoid noise.
    Keeps dtype.
    """
    g = _ensure_gray(gray)
    dtype = g.dtype
    lo, hi = _range_for_dtype(dtype)

    base = g.astype(np.float32) if not _is_float(g) else g.astype(np.float32)
    ksize = max(1, int(round(radius * 3)) * 2 + 1)
    blur = cv2.GaussianBlur(base, (ksize, ksize), sigmaX=radius, sigmaY=radius)
    mask = base - blur

    if threshold > 0:
        # zero out small differences (units match dtype range)
        thr = float(threshold) if _is_float(g) else float(threshold)
        mask[np.abs(mask) < thr] = 0.0

    out = base + amount * mask
    out = np.clip(out, lo, hi)

    if dtype == np.uint8:
        return out.astype(np.uint8)
    if dtype == np.uint16:
        return out.astype(np.uint16)
    return out.astype(dtype)
