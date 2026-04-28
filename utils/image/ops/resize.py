# utils/image/ops/resize.py
"""
Image resize operations for the Martial Arts OCR system.

Responsibilities:
- Aspect-ratio preserving resizes by width/height/long edge/scale
- Fit (contain) and Fill (cover) behaviors with optional padding/cropping
- Safe thumbnail helper with heuristic interpolation
- Letterbox padding for detectors (e.g., YOLO)
- Utility helpers for choosing OpenCV interpolation

Conventions:
- All functions accept NumPy ndarrays (GRAY 2D or color HxWxC).
- Interpolation defaults are chosen based on up/downscale heuristics.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Mode = Literal["fit", "fill"]
PadAlign = Literal["center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"]

__all__ = [
    "resize_to_width",
    "resize_to_height",
    "resize_long_edge",
    "resize_scale",
    "fit_into",
    "fill_into",
    "thumbnail",
    "letterbox",
    "choose_interpolation",
]

# -------------------------- Interpolation utilities ---------------------------

def choose_interpolation(
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    prefer_area_for_downscale: bool = True,
) -> int:
    """
    Choose a reasonable OpenCV interpolation based on scale direction.

    - Downscale: INTER_AREA (good anti-aliasing), or INTER_LINEAR if requested otherwise
    - Upscale: INTER_CUBIC for quality; INTER_LINEAR is faster
    """
    sh, sw = src_hw
    dh, dw = dst_hw
    if dh <= 0 or dw <= 0:
        raise ValueError(f"Invalid destination size {dst_hw}")

    down = (dh < sh) or (dw < sw)
    if down and prefer_area_for_downscale:
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC if (dh > sh or dw > sw) else cv2.INTER_LINEAR


def _ensure_min(val: int, floor: int = 1) -> int:
    return max(floor, int(round(val)))


def _dims(img: np.ndarray) -> Tuple[int, int]:
    h, w = img.shape[:2]
    return h, w


def _resize(img: np.ndarray, w: int, h: int, interpolation: Optional[int]) -> np.ndarray:
    h0, w0 = _dims(img)
    if (w, h) == (w0, h0):
        return img
    if interpolation is None:
        interpolation = choose_interpolation((h0, w0), (h, w))
    return cv2.resize(img, (w, h), interpolation=interpolation)


# ------------------------------ Basic resizers --------------------------------

def resize_to_width(img: np.ndarray, target_w: int, interpolation: Optional[int] = None) -> np.ndarray:
    """
    Resize keeping aspect ratio so that width == target_w.
    """
    if target_w <= 0:
        raise ValueError("target_w must be > 0")
    h, w = _dims(img)
    scale = target_w / w
    target_h = _ensure_min(h * scale)
    return _resize(img, target_w, target_h, interpolation)


def resize_to_height(img: np.ndarray, target_h: int, interpolation: Optional[int] = None) -> np.ndarray:
    """
    Resize keeping aspect ratio so that height == target_h.
    """
    if target_h <= 0:
        raise ValueError("target_h must be > 0")
    h, w = _dims(img)
    scale = target_h / h
    target_w = _ensure_min(w * scale)
    return _resize(img, target_w, target_h, interpolation)


def resize_long_edge(img: np.ndarray, max_long_edge: int, interpolation: Optional[int] = None) -> np.ndarray:
    """
    Resize so that the longer side equals max_long_edge; preserves aspect.
    """
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be > 0")
    h, w = _dims(img)
    if max(h, w) == max_long_edge:
        return img
    if h >= w:
        return resize_to_height(img, max_long_edge, interpolation)
    else:
        return resize_to_width(img, max_long_edge, interpolation)


def resize_scale(img: np.ndarray, scale: float, interpolation: Optional[int] = None) -> np.ndarray:
    """
    Resize by scale factor (e.g., 0.5 downscales by 50%, 2.0 doubles size).
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")
    h, w = _dims(img)
    nh = _ensure_min(h * scale)
    nw = _ensure_min(w * scale)
    return _resize(img, nw, nh, interpolation)


# ---------------------------- Fit / Fill behaviors ----------------------------

def _pad_canvas(
    img: np.ndarray,
    dst_hw: Tuple[int, int],
    bg: Tuple[int, int, int] = (0, 0, 0),
    align: PadAlign = "center",
) -> np.ndarray:
    """
    Paste img into a canvas of dst_hw with a given alignment.
    """
    dh, dw = dst_hw
    h, w = _dims(img)
    if img.ndim == 2:
        canvas = np.full((dh, dw), bg[0] if isinstance(bg, tuple) else bg, dtype=img.dtype)  # type: ignore[index]
    else:
        canvas = np.full((dh, dw, img.shape[2]), bg, dtype=img.dtype)

    # compute top-left
    if align in ("center",):
        y = (dh - h) // 2
        x = (dw - w) // 2
    elif align == "top":
        y, x = 0, (dw - w) // 2
    elif align == "bottom":
        y, x = dh - h, (dw - w) // 2
    elif align == "left":
        y, x = (dh - h) // 2, 0
    elif align == "right":
        y, x = (dh - h) // 2, dw - w
    elif align == "top-left":
        y, x = 0, 0
    elif align == "top-right":
        y, x = 0, dw - w
    elif align == "bottom-left":
        y, x = dh - h, 0
    elif align == "bottom-right":
        y, x = dh - h, dw - w
    else:
        y, x = (dh - h) // 2, (dw - w) // 2

    canvas[y : y + h, x : x + w] = img
    return canvas


def fit_into(
    img: np.ndarray,
    dst_size: Tuple[int, int],
    interpolation: Optional[int] = None,
    bg: Tuple[int, int, int] = (0, 0, 0),
    align: PadAlign = "center",
) -> np.ndarray:
    """
    Resize with aspect preserved so the image fits entirely inside dst_size (contain),
    adding letterbox padding if needed to reach the exact destination size.

    Args:
        dst_size: (width, height) of the desired output.
        bg: background color for padding (BGR for color images).
    """
    dw, dh = dst_size
    if dw <= 0 or dh <= 0:
        raise ValueError("dst_size must be positive")

    h, w = _dims(img)
    scale = min(dw / w, dh / h)
    nw = _ensure_min(w * scale)
    nh = _ensure_min(h * scale)

    resized = _resize(img, nw, nh, interpolation)
    if (nw, nh) == (dw, dh):
        return resized
    return _pad_canvas(resized, (dh, dw), bg=bg, align=align)


def fill_into(
    img: np.ndarray,
    dst_size: Tuple[int, int],
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """
    Resize with aspect preserved so the image covers dst_size (cover)
    and then center-crop to exact dimensions.
    """
    dw, dh = dst_size
    if dw <= 0 or dh <= 0:
        raise ValueError("dst_size must be positive")

    h, w = _dims(img)
    scale = max(dw / w, dh / h)
    nw = _ensure_min(w * scale)
    nh = _ensure_min(h * scale)

    resized = _resize(img, nw, nh, interpolation)
    # center-crop to (dh, dw)
    y0 = max(0, (nh - dh) // 2)
    x0 = max(0, (nw - dw) // 2)
    return resized[y0 : y0 + dh, x0 : x0 + dw]


# ------------------------------- Thumbnails -----------------------------------

def thumbnail(
    img: np.ndarray,
    max_size: Tuple[int, int] = (200, 300),
    prefer_area_for_downscale: bool = True,
) -> np.ndarray:
    """
    Create a small preview while preserving aspect.
    - Uses INTER_AREA for downscale by default (crisper text).
    """
    h, w = _dims(img)
    mw, mh = max_size
    scale = min(mw / w, mh / h)
    if scale >= 1.0:
        # small upscale okay; switch to a gentle interpolator
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA if prefer_area_for_downscale else cv2.INTER_LINEAR
    nw = _ensure_min(w * scale)
    nh = _ensure_min(h * scale)
    return _resize(img, nw, nh, interp)


# ------------------------------- Letterbox ------------------------------------

def letterbox(
    img: np.ndarray,
    dst_size: Tuple[int, int],
    *,
    stride: int = 32,
    auto: bool = True,
    scale_fill: bool = False,
    scale_up: bool = True,
    bg: Tuple[int, int, int] = (114, 114, 114),
    align: PadAlign = "center",
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    YOLO-style letterbox:
    - Fits image into dst_size with preserved aspect.
    - Adds padding (bg color) to reach exact dst_size.
    - Optionally makes final size multiple of 'stride' when auto=True.
    - Returns (image, scale, (pad_w, pad_h)).

    Args:
        dst_size: (width, height)
        stride: multiple to align final dims to (32 for YOLO).
        auto: if True, reduce padding so final dims are multiples of stride.
        scale_fill: if True, ignore aspect ratio and just stretch to dst_size.
        scale_up: if False, do not upscale (only downscale).
        bg: padding color (BGR).
    """
    dw, dh = dst_size
    if scale_fill:
        out = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
        return out, dw / _dims(img)[1], (0, 0)

    h, w = _dims(img)
    r = min(dw / w, dh / h)
    if not scale_up:
        r = min(r, 1.0)

    nw = int(round(w * r))
    nh = int(round(h * r))

    pad_w = dw - nw
    pad_h = dh - nh

    if auto:
        # make padding multiples of stride
        pad_w = pad_w % stride
        pad_h = pad_h % stride

    # resize
    resized = cv2.resize(img, (nw, nh), interpolation=choose_interpolation((h, w), (nh, nw)))

    # distribute padding according to alignment
    if align == "center":
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
    else:
        # map align to offsets
        if align in ("top", "top-left", "top-right"):
            top = 0
            bottom = pad_h
        elif align in ("bottom", "bottom-left", "bottom-right"):
            top = pad_h
            bottom = 0
        else:
            top = pad_h // 2
            bottom = pad_h - top

        if align in ("left", "top-left", "bottom-left"):
            left = 0
            right = pad_w
        elif align in ("right", "top-right", "bottom-right"):
            left = pad_w
            right = 0
        else:
            left = pad_w // 2
            right = pad_w - left

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=bg
    )
    return out, r, (pad_w, pad_h)
