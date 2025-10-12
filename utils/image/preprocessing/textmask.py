from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple

def _blackhat_text(gray: np.ndarray, k: int) -> np.ndarray:
    """Black-hat transform to enhance dark text on light background."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thr = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr

def _mser_glyphs(gray: np.ndarray, delta=5, min_area=30, max_area_ratio=0.2) -> np.ndarray:
    """MSER-based small glyph detection (text candidates)."""
    h, w = gray.shape[:2]
    max_area = int(h * w * max_area_ratio)

    # OpenCV MSER args must be passed positionally (not as _delta, _min_area, etc.)
    mser = cv2.MSER_create(delta, min_area, max_area)
    regions, _ = mser.detectRegions(gray)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        if w * h < min_area or h < 6:
            continue
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

def _join_blocks(text_mask: np.ndarray, join_w: int, join_h: int) -> np.ndarray:
    """Join nearby glyphs into text-line blocks."""
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (join_w, join_h))
    dil = cv2.dilate(text_mask, ker, iterations=1)
    ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, ker2, iterations=1)
    return closed

def build_nontext_mask(
    gray: np.ndarray,
    k_blackhat: int = 31,
    mser_delta: int = 5,
    mser_min_area: int = 30,
    mser_max_area_ratio: float = 0.2,
    join_w: int = 15,
    join_h: int = 3
) -> np.ndarray:
    """
    Build a non-text mask from grayscale.
    Returns uint8 mask: 255 = non-text (keep), 0 = text (suppress).
    """
    # Ensure grayscale uint8
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1) Dark-text candidates (black-hat)
    bh = _blackhat_text(gray, k_blackhat)

    # 2) MSER glyphs
    mser = _mser_glyphs(gray, delta=mser_delta,
                        min_area=mser_min_area,
                        max_area_ratio=mser_max_area_ratio)

    # 3) Union → join into blocks
    text_mask = cv2.bitwise_or(bh, mser)
    blocks = _join_blocks(text_mask, join_w, join_h)

    # 4) Non-text = inverse of text blocks
    nontext = cv2.bitwise_not(blocks)
    return nontext
