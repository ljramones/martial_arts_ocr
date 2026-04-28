# utils/image/pipelines/ocr_prep.py
"""
OCR-oriented preprocessing pipelines.

Design goals
------------
- Produce stable, low-noise grayscale suitable for dense strokes (kanji/kana) and Latin text.
- Keep everything deterministic and CPU-friendly.
- Stay dtype-aware; return uint8 grayscale by default (best for most OCR engines).

Provided pipelines
------------------
- jp_text_prep(...)      : conservative pipeline tuned for JP text (kanji/kana)
- en_text_prep(...)      : slightly stronger local contrast for Latin/English pages
- adaptive_binarize(...) : optional final step if your OCR prefers binary images
"""

from __future__ import annotations
from typing import Tuple, Literal, Optional

import cv2
import numpy as np

from utils.image.ops.tone import (
    to_gray,
    auto_contrast_gray,
    clahe_gray,
    unsharp_mask_gray,
    gamma_correct,
)
from utils.image.ops.resize import resize_long_edge

BinarizeMethod = Literal["otsu", "sauvola", "niblack", "adaptive_mean", "adaptive_gaussian"]

# ------------------------------ helpers ---------------------------------------

def _maybe_resize_long_edge(gray: np.ndarray, target_long_edge: Optional[int]) -> np.ndarray:
    if not target_long_edge or target_long_edge <= 0:
        return gray
    h, w = gray.shape[:2]
    if max(h, w) == target_long_edge:
        return gray
    if h >= w:
        new_w = max(1, int(round(w * target_long_edge / float(h))))
        return cv2.resize(gray, (new_w, target_long_edge), interpolation=cv2.INTER_LINEAR)
    else:
        new_h = max(1, int(round(h * target_long_edge / float(w))))
        return cv2.resize(gray, (target_long_edge, new_h), interpolation=cv2.INTER_LINEAR)

# ------------------------------ pipelines -------------------------------------

def jp_text_prep(
    img_bgr_or_gray: np.ndarray,
    *,
    target_long_edge: int | None = None,
    autocontrast_cutoff: float = 1.0,
    clahe_clip: float = 2.0,
    clahe_tile: Tuple[int, int] = (8, 8),
    unsharp_radius: float = 1.0,
    unsharp_amount: float = 0.6,
    unsharp_threshold: int = 2,
    midtone_gamma: float | None = None,
    return_uint8: bool = True,
) -> np.ndarray:
    """
    Conservative, OCR-friendly grayscale prep for Japanese text:
      1) convert to gray
      2) optional resize to standardize scale
      3) percentile auto-contrast (robust to background tint)
      4) CLAHE (local contrast equalization)
      5) gentle unsharp mask (edge contrast) with threshold to avoid noise
      6) optional midtone gamma lift (e.g., 1.1–1.3)

    Returns grayscale (uint8 by default).
    """
    g = to_gray(img_bgr_or_gray)
    g = _maybe_resize_long_edge(g, target_long_edge)
    g = auto_contrast_gray(g, cutoff=autocontrast_cutoff)
    g = clahe_gray(g, clip=clahe_clip, tile=clahe_tile)
    g = unsharp_mask_gray(g, radius=unsharp_radius, amount=unsharp_amount, threshold=unsharp_threshold)
    if midtone_gamma and abs(midtone_gamma - 1.0) > 1e-3:
        g = gamma_correct(g, gamma=float(midtone_gamma))
    if return_uint8 and g.dtype != np.uint8:
        # normalize defensively
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g


def en_text_prep(
    img_bgr_or_gray: np.ndarray,
    *,
    target_long_edge: int | None = None,
    autocontrast_cutoff: float = 1.0,
    clahe_clip: float = 3.0,
    clahe_tile: Tuple[int, int] = (8, 8),
    unsharp_radius: float = 1.2,
    unsharp_amount: float = 0.8,
    unsharp_threshold: int = 3,
    midtone_gamma: float | None = 1.1,
    return_uint8: bool = True,
) -> np.ndarray:
    """
    Slightly stronger local-contrast variant that often helps Latin/English pages
    (lighter strokes and larger x-height). Same structure as jp_text_prep, with
    a tad more CLAHE and sharpening by default.
    """
    g = jp_text_prep(
        img_bgr_or_gray,
        target_long_edge=target_long_edge,
        autocontrast_cutoff=autocontrast_cutoff,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        unsharp_radius=unsharp_radius,
        unsharp_amount=unsharp_amount,
        unsharp_threshold=unsharp_threshold,
        midtone_gamma=midtone_gamma,
        return_uint8=return_uint8,
    )
    return g

# ------------------------------- binarization ---------------------------------

def adaptive_binarize(
    gray: np.ndarray,
    *,
    method: BinarizeMethod = "otsu",
    block_size: int = 31,
    k: float = 0.2,
    C: int = 5,
) -> np.ndarray:
    """
    Convert grayscale to binary using a chosen method.

    Args:
      gray: uint8 or float image (will be normalized to uint8 internally).
      method:
        - "otsu"               : global Otsu threshold
        - "adaptive_mean"      : cv2 adaptive mean
        - "adaptive_gaussian"  : cv2 adaptive gaussian
        - "sauvola"            : Sauvola thresholding (manual impl)
        - "niblack"            : Niblack thresholding (manual impl)
      block_size: odd window size for adaptive/sauvola/niblack (e.g., 31, 51).
      k: method-specific parameter (Sauvola/Niblack).
      C: constant subtracted in OpenCV adaptive methods.

    Returns:
      uint8 binary image with values {0, 255}.
    """
    if gray.ndim != 2:
        raise ValueError("adaptive_binarize expects grayscale input")
    # Normalize to uint8
    if gray.dtype != np.uint8:
        g8 = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g8 = gray

    if method == "otsu":
        _, th = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    if method == "adaptive_mean":
        bs = max(3, block_size | 1)  # ensure odd
        th = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, C)
        return th

    if method == "adaptive_gaussian":
        bs = max(3, block_size | 1)
        th = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C)
        return th

    # Manual Sauvola/Niblack (vectorized)
    bs = max(3, block_size | 1)
    mean = cv2.boxFilter(g8.astype(np.float32), ddepth=-1, ksize=(bs, bs), normalize=True)
    sqmean = cv2.boxFilter((g8.astype(np.float32) ** 2), ddepth=-1, ksize=(bs, bs), normalize=True)
    var = np.clip(sqmean - mean * mean, 0.0, None)
    std = np.sqrt(var + 1e-6)

    if method == "niblack":
        t = mean + k * std
    elif method == "sauvola":
        R = 128.0  # dynamic range
        t = mean * (1.0 + k * ((std / R) - 1.0))
    else:
        raise ValueError(f"Unknown binarize method: {method}")

    out = (g8 >= t).astype(np.uint8) * 255
    return out

# ------------------------------- convenience ----------------------------------

def ocr_prep(
    img_bgr_or_gray: np.ndarray,
    *,
    lang: Literal["jp", "en"] = "jp",
    target_long_edge: int | None = None,
    binarize: bool = False,
    bin_method: BinarizeMethod = "otsu",
    bin_block: int = 31,
    bin_k: float = 0.2,
    bin_C: int = 5,
) -> np.ndarray:
    """
    One-call convenience:
      - choose JP/EN tuned pipeline
      - optional binarization as a final step
    """
    if lang == "jp":
        g = jp_text_prep(img_bgr_or_gray, target_long_edge=target_long_edge)
    else:
        g = en_text_prep(img_bgr_or_gray, target_long_edge=target_long_edge)

    if binarize:
        g = adaptive_binarize(g, method=bin_method, block_size=bin_block, k=bin_k, C=bin_C)
    return g
