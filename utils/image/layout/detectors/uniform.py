# utils/image/layout/detectors/uniform.py
from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

from utils.image.regions.core_image import ImageRegion
from . import BaseDetector

logger = logging.getLogger(__name__)


class UniformDetector(BaseDetector):
    """
    Detect large uniform regions (background panels, shaded blocks, etc.)
    using morphological closing + Otsu and a local std gate.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        # Config knobs (with sensible defaults)
        self.close_k: int = int(cfg.get("uniform_close_kernel", 15))
        self.min_area_ratio: float = float(cfg.get("uniform_min_area_ratio", 0.03))  # ≥ 3% of page
        self.max_area_ratio: float = float(cfg.get("uniform_max_area_ratio", 0.50))  # ≤ 50% of page
        self.aspect_min: float = float(cfg.get("uniform_aspect_min", 0.3))
        self.aspect_max: float = float(cfg.get("uniform_aspect_max", 3.0))
        self.std_min: float = float(cfg.get("uniform_std_min", 10.0))
        self.std_max: float = float(cfg.get("uniform_std_max", 100.0))

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray if gray.dtype == np.uint8 else cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    def detect(self, gray_in: np.ndarray) -> List[ImageRegion]:
        try:
            gray = self._to_gray_u8(gray_in)
            H, W = gray.shape[:2]
            if H < 10 or W < 10:
                return []

            # Morph close to smooth uniform areas (ensure odd kernel ≥ 3)
            k = int(max(3, int(self.close_k) | 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Otsu threshold
            _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []

            page_area = H * W
            min_area = max(int(page_area * self.min_area_ratio), 1)
            max_area = max(int(page_area * self.max_area_ratio), min_area + 1)

            regions: List[ImageRegion] = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Clamp and validate ROI bounds defensively
                x = max(0, x)
                y = max(0, y)
                w = min(w, W - x)
                h = min(h, H - y)
                if w <= 0 or h <= 0:
                    continue

                area = w * h
                if area < min_area or area > max_area:
                    continue

                aspect = w / float(h) if h > 0 else 0.0
                if not (self.aspect_min < aspect < self.aspect_max):
                    continue

                roi = gray[y:y + h, x:x + w]

                # Avoid page margins / pure white blocks
                if float(np.mean(roi)) > 245.0:
                    continue

                # Uniform but not completely flat — keep within [std_min, std_max]
                local_std = float(np.std(roi))
                if self.std_min < local_std < self.std_max:
                    regions.append(ImageRegion(
                        x=x, y=y, width=w, height=h,
                        region_type="image", confidence=0.70
                    ))
                    logger.debug(
                        "UniformDetector: (%d,%d,%d,%d) area=%d std=%.1f",
                        x, y, w, h, area, local_std
                    )

            return regions

        except Exception as e:
            logger.error("Uniform detection failed: %s", e)
            return []
