# utils/image/layout/detectors/figure.py
from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Callable, Dict, Any, Optional

from utils.image.regions.core_image import ImageRegion
from . import BaseDetector

logger = logging.getLogger(__name__)


class FigureDetector(BaseDetector):
    """
    Detect regions that look like figures/diagrams based on spatial isolation.

    Usage:
        det = FigureDetector(cfg, halo_check=halo_ok_fn)
        regions = det.detect(gray_u8)
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 halo_check: Callable[[np.ndarray, int, int, int, int], bool]):
        """
        Args:
            cfg: layout config (typically get_config().LAYOUT_DETECTION)
            halo_check: function(gray, x, y, w, h) -> bool
                        checks whitespace 'halo' around candidate box
        """
        self.cfg = cfg
        self.halo_ok = halo_check

        # Tunables (with safe defaults)
        self.min_area: int = int(cfg.get('figure_min_area', 10000))
        self.max_area_ratio: float = float(cfg.get('figure_max_area_ratio', 0.5))
        self.left_bias_xmax: float = float(cfg.get('figure_left_bias_xmax', 0.7))
        self.aspect_min: float = float(cfg.get('figure_aspect_min', 0.4))
        self.aspect_max: float = float(cfg.get('figure_aspect_max', 3.0))
        self.isolation_white: float = float(cfg.get('figure_isolation_white', 0.75))
        self.close_kernel: int = int(cfg.get('figure_close_kernel', 5))
        self.margin = int(cfg.get('figure_margin', cfg.get('margin_threshold', 20)))

    def _to_gray_u8(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 and img.shape[2] == 3 else img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def detect(self, gray_in: np.ndarray) -> List[ImageRegion]:
        """
        Args:
            gray_in: grayscale uint8 page (or color; will be converted)

        Returns:
            list of ImageRegion with region_type="figure"
        """
        try:
            gray = self._to_gray_u8(gray_in)
            h, w = gray.shape[:2]
            regions: List[ImageRegion] = []

            # 1) Binarize (invert) to connect dark components
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 2) Morph close with a smaller kernel to avoid over-merging text into figures
            k = max(3, self.close_kernel | 1)  # odd-ish / >=3
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 3) Contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return regions

            max_area = int(h * w * self.max_area_ratio)

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = width * height

                # Size constraints
                if area < self.min_area or area > max_area:
                    continue

                # Left bias (empirical heuristic; keep configurable)
                if x > int(w * self.left_bias_xmax):
                    continue

                # Aspect ratio
                aspect = width / float(height) if height > 0 else 0.0
                if not (self.aspect_min < aspect < self.aspect_max):
                    continue

                # Isolation: surrounding whitespace
                mx = self.margin
                x0 = max(0, x - mx); y0 = max(0, y - mx)
                x1 = min(w, x + width + mx); y1 = min(h, y + height + mx)
                surrounding = gray[y0:y1, x0:x1]
                if surrounding.size == 0:
                    continue

                _, surrounding_binary = cv2.threshold(surrounding, 200, 255, cv2.THRESH_BINARY)
                surrounding_white_ratio = float(np.mean(surrounding_binary == 255))
                if surrounding_white_ratio <= self.isolation_white:
                    continue

                # Halo around candidate must be white enough
                if not self.halo_ok(gray, x, y, width, height):
                    logger.debug("Figure rejected (halo): (%d,%d,%d,%d)", x, y, width, height)
                    continue

                logger.info("Figure detected at (%d,%d): area=%d, isolation=%.2f",
                            x, y, area, surrounding_white_ratio)
                regions.append(ImageRegion(
                    x=x, y=y, width=width, height=height,
                    region_type="figure", confidence=0.90
                ))

            return regions

        except Exception as e:
            logger.error("Figure detection failed: %s", e)
            return []
