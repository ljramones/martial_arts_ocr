from __future__ import annotations
import cv2
import numpy as np

def apply_nontext_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if mask.shape != gray.shape:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    # NEW: binarize and uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # ensure 0/255
    if mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(gray, gray, mask=mask)
