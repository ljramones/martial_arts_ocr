# utils/image/layout/detectors/contours.py
from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Callable, Dict, Any

from utils.image.regions.core_image import ImageRegion
from . import BaseDetector

logger = logging.getLogger(__name__)


class ContourDetector(BaseDetector):
    """
    Detect line-art / diagram-like regions using edge → dilate → contour analysis.

    Supports relaxed 'line_mode' for single-line diagrams and
    optional halo requirement (configurable).
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

        # Tunable thresholds with sensible defaults
        self.min_area: int = int(cfg.get('contour_min_area', 15000))
        self.max_area_ratio: float = float(cfg.get('contour_max_area_ratio', 0.4))
        self.aspect_min: float = float(cfg.get('contour_aspect_min', 0.3))
        self.aspect_max: float = float(cfg.get('contour_aspect_max', 3.5))
        self.left_bias_xmax: float = float(cfg.get('contour_left_bias_xmax', 0.6))

        # Edge + dilate params
        self.canny_lo: int = int(cfg.get('contour_canny_lo', 30))
        self.canny_hi: int = int(cfg.get('contour_canny_hi', 90))
        self.dilate_k: int = int(cfg.get('contour_dilate_kernel', 5))
        self.dilate_iters: int = int(cfg.get('contour_dilate_iters', 2))

        # Diagram structure acceptance
        self.hough_min_len: int = int(cfg.get('contour_hough_min_line_length', 30))
        self.hough_gap: int = int(cfg.get('contour_hough_max_gap', 10))
        self.hough_thresh: int = int(cfg.get('contour_hough_threshold', 50))
        self.edge_density_min: float = float(cfg.get('contour_edge_density_min', 0.02))
        self.edge_density_max: float = float(cfg.get('contour_edge_density_max', 0.15))
        self.white_ratio_min: float = float(cfg.get('contour_white_ratio_min', 0.6))
        self.white_ratio_left_bias: float = float(cfg.get('contour_white_ratio_left_bias', 0.7))

        # CC-based text rejection
        self.reject_text_like_early: bool = bool(cfg.get('contour_reject_text_like_early', False))
        self.reject_page_edge_text_like: bool = bool(cfg.get('contour_reject_page_edge_text_like', True))
        self.page_edge_margin: int = int(cfg.get('image_border_margin', 10))
        self.cc_small_thresh: int = int(cfg.get('contour_cc_small_thresh', 200))
        self.cc_small_count: int = int(cfg.get('contour_cc_small_count', 30))
        self.cc_median_area_max: int = int(cfg.get('contour_cc_median_area_max', 150))

        # Limit number of final regions
        self.topk: int = int(cfg.get('contour_topk', 6))

        # New: optional relaxed line mode and halo requirement
        self.require_halo: bool = bool(cfg.get('contours_require_halo', False))
        self.line_mode: bool = bool(cfg.get('contours_line_mode', True))

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray if gray.dtype == np.uint8 else cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect(self, gray_in: np.ndarray) -> List[ImageRegion]:
        """
        Args:
            gray_in: grayscale uint8 page (or color; will be converted)
        Returns:
            list of ImageRegion with region_type="diagram"
        """
        try:
            gray = self._to_gray_u8(gray_in)
            h, w = gray.shape[:2]
            regions: List[ImageRegion] = []

            # 1) Edge → Dilate
            edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
            k = max(3, self.dilate_k | 1)  # odd/safe
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            dilated = cv2.dilate(edges, kernel, iterations=max(1, self.dilate_iters))

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return regions

            max_area = int(h * w * self.max_area_ratio)

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = width * height
                if not (self.min_area < area < max_area):
                    continue

                # Aspect ratio screen
                aspect_ratio = width / float(height) if height > 0 else 0.0
                if self.line_mode:
                    ok_aspect = 0.05 < aspect_ratio < 20.0
                else:
                    ok_aspect = self.aspect_min < aspect_ratio < self.aspect_max
                if not ok_aspect:
                    continue

                # Optional left-bias (empirical)
                if x >= int(w * self.left_bias_xmax):
                    roi_tmp = gray[y:y + height, x:x + width]
                    white_ratio_local = float(np.sum(roi_tmp > 200)) / float(area)
                    # relax white bias in line_mode
                    left_bias_thresh = 0.40 if self.line_mode else self.white_ratio_left_bias
                    if white_ratio_local < left_bias_thresh:
                        continue

                roi = gray[y:y + height, x:x + width]

                # 2) Optional early text-like rejection. Keep the broad form
                # disabled by default because sparse symbols and labeled diagrams
                # can look glyph-like before the final TextRegionFilter sees full
                # context. Still reject page-edge text-like bands early: those
                # large exterior contours can hide nested real drawings when
                # RETR_EXTERNAL is used.
                touches_page_edge = (
                    x <= self.page_edge_margin
                    or y <= self.page_edge_margin
                )
                should_check_text_like_early = (
                    self.reject_text_like_early
                    or (self.reject_page_edge_text_like and touches_page_edge)
                )
                if should_check_text_like_early:
                    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)
                    if num_labels > 2:
                        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
                        median_comp_area = np.median(areas) if areas.size else 0
                        num_small_components = int(np.sum(areas < self.cc_small_thresh))
                        if (num_small_components > self.cc_small_count and
                                median_comp_area < self.cc_median_area_max):
                            logger.debug("ContourDetector: text-like CC pattern at (%d,%d)", x, y)
                            continue

                # 3) Diagram-ish structure: edges / lines / whitespace
                roi_edges = cv2.Canny(roi, 50, 150)
                lines = cv2.HoughLinesP(
                    roi_edges, 1, np.pi / 180, self.hough_thresh,
                    minLineLength=self.hough_min_len, maxLineGap=self.hough_gap
                )

                edge_density = float(np.sum(roi_edges > 0)) / float(area)
                white_ratio = float(np.sum(roi > 200)) / float(area)
                has_significant_lines = lines is not None and len(lines) > 5

                if self.line_mode:
                    edge_min, edge_max = 0.001, 0.20  # relaxed for ultra-sparse drawings
                    white_thresh = 0.40  # relaxed white-space requirement
                else:
                    edge_min, edge_max = self.edge_density_min, self.edge_density_max
                    white_thresh = self.white_ratio_min

                accept = (
                        has_significant_lines or
                        (edge_min < edge_density < edge_max and white_ratio > white_thresh) or
                        (x < int(w * self.left_bias_xmax) and white_ratio > (
                            0.40 if self.line_mode else self.white_ratio_left_bias))
                )
                if not accept:
                    continue

                # 4) Optional whitespace halo check
                if self.require_halo and not self.halo_ok(gray, x, y, width, height):
                    logger.debug("ContourDetector: rejected by halo at (%d,%d,%d,%d)", x, y, width, height)
                    continue

                metadata = {
                    "detector": "contours",
                    "edge_density": edge_density,
                    "white_ratio": white_ratio,
                    "hough_line_count": int(len(lines)) if lines is not None else 0,
                    "area_ratio": area / float(max(1, h * w)),
                }
                logger.info(
                    "ContourDetector: diagram at (%d,%d) area=%d ed=%.3f wr=%.2f",
                    x, y, area, edge_density, white_ratio
                )
                regions.append(ImageRegion(
                    x=int(x), y=int(y),
                    width=int(width), height=int(height),
                    region_type="diagram", confidence=0.85,
                    metadata=metadata
                ))

            # Keep only top-K largest to avoid clutter
            if self.topk > 0 and len(regions) > self.topk:
                regions = sorted(regions, key=lambda r: r.area, reverse=True)[:self.topk]

            return regions

        except Exception as e:
            logger.error("Contour detection failed: %s", e)
            return []
