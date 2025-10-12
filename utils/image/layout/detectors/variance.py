# utils/image/layout/detectors/variance.py
from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from . import BaseDetector


from utils.image.regions.core_image import ImageRegion

logger = logging.getLogger(__name__)


class VarianceDetector(BaseDetector):
    """
    Detect photo-like regions using local variance + gradient smoothness.

    Usage:
        det = VarianceDetector(cfg)
        regions = det.detect(gray_u8)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # Sliding window & thresholds
        self.window_min: int = int(cfg.get('variance_window_min', 128))
        self.window_rel: float = float(cfg.get('variance_window_rel', 1.0 / 8.0))  # min(h,w)/8
        self.stride_rel: float = float(cfg.get('variance_stride_rel', 1.0))        # stride = window * stride_rel

        self.var_min: float = float(cfg.get('variance_min', 100.0))
        self.var_max: float = float(cfg.get('variance_max', 5000.0))
        self.grad_smooth_thresh: float = float(cfg.get('variance_grad_smooth_thresh', 50.0))

        # Expanded region rules
        self.expand_intensity_thresh: float = float(cfg.get('variance_expand_intensity_thresh', 30.0))
        self.min_expanded_area: int = int(cfg.get('variance_min_expanded_area', 20000))

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _expand_region(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> ImageRegion:
        """
        Expand (x,y,w,h) outward while surrounding pixels remain near the ROI mean.
        """
        H, W = gray.shape[:2]
        # Clamp
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = min(w, W - x)
        h = min(h, H - y)
        if w <= 0 or h <= 0:
            return ImageRegion(x, y, 1, 1, region_type="photo", confidence=0.50)

        roi = gray[y:y + h, x:x + w]
        mean_intensity = float(np.mean(roi))
        thr = float(self.expand_intensity_thresh)

        # Expand left
        x0 = x
        while x0 > 0:
            col = gray[y:y + h, x0 - 1:x0]
            if col.size > 0 and abs(float(np.mean(col)) - mean_intensity) <= thr:
                x0 -= 1
            else:
                break

        # Expand right
        x1 = x + w
        while x1 < W:
            col = gray[y:y + h, x1:x1 + 1]
            if col.size > 0 and abs(float(np.mean(col)) - mean_intensity) <= thr:
                x1 += 1
            else:
                break

        # Expand up
        y0 = y
        while y0 > 0:
            row = gray[y0 - 1:y0, x0:x1]
            if row.size > 0 and abs(float(np.mean(row)) - mean_intensity) <= thr:
                y0 -= 1
            else:
                break

        # Expand down
        y1 = y + h
        while y1 < H:
            row = gray[y1:y1 + 1, x0:x1]
            if row.size > 0 and abs(float(np.mean(row)) - mean_intensity) <= thr:
                y1 += 1
            else:
                break

        w2 = max(1, x1 - x0)
        h2 = max(1, y1 - y0)
        return ImageRegion(x=x0, y=y0, width=w2, height=h2, region_type="photo", confidence=0.75)

    def _too_close(self, new_box: Tuple[int, int, int, int], existing: List[Tuple[int, int, int, int]], min_dist: int) -> bool:
        x, y, w, h = new_box
        cx, cy = x + w // 2, y + h // 2
        for ex, ey, ew, eh in existing:
            ecx, ecy = ex + ew // 2, ey + eh // 2
            if (cx - ecx) ** 2 + (cy - ecy) ** 2 < (min_dist ** 2):
                return True
        return False

    def detect(self, gray_in: np.ndarray) -> List[ImageRegion]:
        try:
            gray = self._to_gray_u8(gray_in)
            H, W = gray.shape[:2]
            if H < 8 or W < 8:
                return []

            # Determine window/stride
            base = min(H, W)
            win = max(self.window_min, int(base * self.window_rel))
            stride = max(1, int(win * self.stride_rel))

            regions: List[ImageRegion] = []
            taken: List[Tuple[int, int, int, int]] = []  # expanded boxes kept

            for y in range(0, max(1, H - win + 1), stride):
                for x in range(0, max(1, W - win + 1), stride):
                    roi = gray[y:y + win, x:x + win]
                    if roi.size == 0:
                        continue

                    # Local variance filter
                    var = float(np.var(roi))
                    if not (self.var_min < var < self.var_max):
                        continue

                    # Gradient smoothness (photos show broader/smooth gradients)
                    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                    smooth = float(np.std(gx) * np.std(gy))
                    if smooth <= self.grad_smooth_thresh:
                        continue

                    # Avoid proposing many near-duplicates
                    if self._too_close((x, y, win, win), taken, min_dist=win):
                        continue

                    # Expand to full photo bounds
                    expanded = self._expand_region(gray, x, y, win, win)
                    if expanded.area < self.min_expanded_area:
                        continue

                    regions.append(expanded)
                    taken.append((expanded.x, expanded.y, expanded.width, expanded.height))
                    logger.debug("VarianceDetector: kept expanded region at (%d,%d) area=%d var=%.1f smooth=%.1f",
                                 expanded.x, expanded.y, expanded.area, var, smooth)

            return regions

        except Exception as e:
            logger.error("Variance detection failed: %s", e)
            return []
