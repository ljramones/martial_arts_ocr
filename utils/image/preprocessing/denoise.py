# utils/image/denoise.py
"""
Denoising & blur utilities.

- blur_var(gray): Laplacian variance (focus proxy)
- preboost_blurry(gray): CLAHE + unsharp when blur is high (rescues OSD/chooser)
- nl_means(img, strength): OpenCV FastNLMeans (gray or color)
"""

from __future__ import annotations

from typing import Literal
import cv2
import numpy as np

from utils.image.shared_utils import _to_gray_u8

Strength = Literal["light", "medium", "strong"]

# One-time CLAHE for the blurry preboost path
_CLAHE_STRONG = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def blur_var(gray: np.ndarray) -> float:
    """
    Variance of Laplacian (higher = sharper). Works best on uint8 gray.
    """
    g = _to_gray_u8(gray)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def preboost_blurry(gray: np.ndarray,
                    blur_threshold:
                    float = 180.0,
                    strength: float = 1.8,
                    blur_weight: float = -0.8
                    ) -> np.ndarray:
    """
    If page looks blurry, apply CLAHE + unsharp to recover edges for orientation/OSD.
    Else return input unchanged.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale or BGR image.
    blur_threshold : float
        If blur_var(gray) <= threshold, apply the boost.

    Returns
    -------
    np.ndarray (uint8 gray)
    """
    g = _to_gray_u8(gray)
    if blur_var(g) > blur_threshold:
        return g

    # Contrast boost (CLAHE), then unsharp mask
    g_eq = _CLAHE_STRONG.apply(g)
    blurred = cv2.GaussianBlur(g_eq, (0, 0), 3)

    sharpened = cv2.addWeighted(g_eq, strength, blurred, blur_weight, 0)

    # Normalize back to full 8-bit range
    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def nl_means(image: np.ndarray, strength: Strength = "light") -> np.ndarray:
    """
    Fast Non-Local Means denoising (OpenCV).
    - Works on gray (2D) or BGR (3D).
    - Strength presets chosen to be gentle with text edges.

    Parameters
    ----------
    image : np.ndarray
        Gray or BGR image (uint8 preferred).
    strength : {"light","medium","strong"}
        Preset controlling filter strength and windows.

    Returns
    -------
    np.ndarray (same shape as input)
    """
    params = {
        "light": (6, 7, 21),  # h, templateWindowSize, searchWindowSize
        "medium": (10, 7, 21),
        "strong": (15, 10, 25),
    }
    h, template_window, search_window = params.get(strength, params["light"])

    # Ensure uint8 for OpenCV’s NLMeans
    if image.dtype != np.uint8:
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img = image

    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)
    elif img.ndim == 3 and img.shape[2] == 3:
        # Use same h for chroma as luma to avoid color bleeding on scans
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)
    else:
        # Fallback: denoise first channel only to avoid shape surprises
        g = _to_gray_u8(img)
        return cv2.fastNlMeansDenoising(g, None, h, template_window, search_window)
