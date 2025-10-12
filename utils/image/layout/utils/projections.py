from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple

def projection_regularity(roi: np.ndarray, thresh: int = 128) -> Tuple[float, float]:
    """
    Returns (std_h, std_v) of peak distances in horizontal/vertical projections.
    Lower std means more regular (text-like).
    """
    hp = np.sum(roi < thresh, axis=1)
    vp = np.sum(roi < thresh, axis=0)
    std_h = _std_peaks(hp)
    std_v = _std_peaks(vp)
    return std_h, std_v

def _std_peaks(proj: np.ndarray) -> float:
    if proj.size <= 10:
        return float("inf")
    peaks, _ = find_peaks(proj, distance=10)
    if peaks.size <= 10:
        return float("inf")
    d = np.diff(peaks)
    return float(np.std(d)) if d.size else float("inf")
