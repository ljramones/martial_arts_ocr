from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def cc_textiness(binary: np.ndarray) -> Tuple[int, float, float]:
    """
    Returns (num_labels, median_area, small_fraction).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return num_labels, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    if areas.size == 0:
        return num_labels, 0.0, 0.0
    median_area = float(np.median(areas))
    small_fraction = float(np.mean(areas < 200))
    return num_labels, median_area, small_fraction
