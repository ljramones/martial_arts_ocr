from __future__ import annotations

import cv2
import numpy as np


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Ensure grayscale uint8 [0..255]."""
    if img is None:
        raise ValueError("binarize._to_gray_u8: input image is None")
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img[..., 0]
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g
