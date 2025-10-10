# utils/image/binarize.py
"""
Binarization & sharpening helpers for document OCR.

- sauvola(gray): robust local thresholding (float32 math)
- adaptive_gaussian(gray), adaptive_mean(gray): OpenCV adaptive methods
- otsu(gray, invert=False): global Otsu
- unsharp(image, strength=1.5, sigma=1.0): edge emphasis
- normalize(image): ensure uint8 [0..255]
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.image.shared_utils import _to_gray_u8

# Reusable CLAHE for gentle pre-contrast before local methods
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _odd_at_least(v: int, lo: int = 3) -> int:
    v = max(int(v), lo)
    return v if (v % 2 == 1) else (v + 1)


def sauvola(
    gray: np.ndarray,
    window: int = 25,
    k: float = 0.2,
    R: float = 128.0,
    use_clahe: bool = True,
) -> np.ndarray:
    """
    Sauvola binarization; returns uint8 binary (0 or 255).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale or BGR image.
    window : int
        Local window size (odd >=3). 25–41 is typical for 300–600dpi scans.
    k : float
        Tuning parameter (0.1–0.5). Higher → more foreground.
    R : float
        Dynamic range normalization constant (128 is common for 8-bit).
    use_clahe : bool
        If True, apply gentle CLAHE first to stabilize low-contrast pages.
    """
    try:
        g = _to_gray_u8(gray)
        if use_clahe:
            g = _CLAHE.apply(g)

        w = _odd_at_least(window, 3)
        f = g.astype(np.float32)

        # Local mean and std via box filters (fast, separable)
        mean = cv2.boxFilter(f, ddepth=-1, ksize=(w, w), normalize=True)
        sqmean = cv2.boxFilter(f * f, ddepth=-1, ksize=(w, w), normalize=True)
        # var = E[x^2] - (E[x])^2  (ensure non-negative)
        var = np.clip(sqmean - mean * mean, 0.0, None)
        std = np.sqrt(var)

        # Sauvola threshold per pixel
        # t = m * (1 + k * (s/R - 1))
        thresh = mean * (1.0 + k * ((std / max(R, 1e-6)) - 1.0))

        # Binary: foreground=255 if g > t
        out = (f > thresh).astype(np.uint8) * 255
        return out
    except Exception:
        # Conservative fallback: OpenCV adaptive Gaussian
        return adaptive_gaussian(gray)


def adaptive_gaussian(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    """
    OpenCV adaptive Gaussian threshold (uint8 output).
    """
    g = _to_gray_u8(gray)
    bs = _odd_at_least(block_size, 3)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C
    )


def adaptive_mean(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    """
    OpenCV adaptive mean threshold (uint8 output).
    """
    g = _to_gray_u8(gray)
    bs = _odd_at_least(block_size, 3)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, C
    )


def otsu(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Global Otsu threshold. Set invert=True to invert foreground/background.
    """
    g = _to_gray_u8(gray)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(g, 0, 255, flag + cv2.THRESH_OTSU)
    return out


def unsharp(image: np.ndarray, strength: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """
    Unsharp masking for edge enhancement (safe for binary/gray).
    For already-binary inputs, this may reintroduce grays; typically apply
    before final binarization, or follow with a tight threshold if needed.
    """
    img = image
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Accept both gray and BGR; operate channel-wise via GaussianBlur
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, strength, blurred, -0.5, 0)
    return sharpened


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Ensure uint8 [0..255].
    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image
