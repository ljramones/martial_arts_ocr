from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from scipy.signal import find_peaks
from utils.image.regions.core_image import ImageRegion
from utils.image.layout.options import RegionDetectionOptions
from ..post.merge import remove_overlaps

logger = logging.getLogger(__name__)


class TextRegionFilter:
    """
    Provides:
      - detect_mser(gray): MSER-based text region proposal
      - filter(gray, regions): reject text-like candidates among image regions

    New features:
      • Configurable regularity and CC thresholds
      • Exempt region types (e.g., YOLO 'figure') from text filtering
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # Tunable thresholds
        self.peak_distance = int(cfg.get('text_peak_distance', 8))       # default 8
        self.proj_std_max = float(cfg.get('text_proj_std_max', 8.0))     # default 8.0
        self.cc_overflow = int(cfg.get('text_cc_overflow_labels', 600))  # default 600
        self.cc_median_max = float(cfg.get('text_cc_median_max', 180.0)) # default 180.0
        self.options = RegionDetectionOptions.from_config(cfg)

        # Exempt region types. Classical contour diagrams should not be exempt,
        # because real-page review showed they may actually be text fragments.
        self.exempt_types = set(cfg.get('filter_text_exempt_types', ['figure']))

    # --------- Detection of text blocks (MSER) ----------
    def detect_mser(self, gray: np.ndarray,
                    min_area: int = None,
                    max_area_ratio: float = 0.3) -> List[ImageRegion]:
        h, w = gray.shape[:2]
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        min_area = min_area or int(self.cfg.get('text_block_min_area', 1000))
        max_area = int(h * w * max_area_ratio)

        mser = cv2.MSER_create(
            5, int(min_area), int(max_area),
            0.25, 0.2, 200, 1.01, 0.003, 5
        )

        regions, _ = mser.detectRegions(gray)
        out: List[ImageRegion] = []
        for r in regions:
            x, y, ww, hh = cv2.boundingRect(r.reshape(-1, 1, 2))
            area = ww * hh
            ar = ww / float(hh) if hh > 0 else 0.0
            if area >= min_area and 0.1 <= ar <= 20.0:
                out.append(ImageRegion(
                    x=x, y=y, width=ww, height=hh,
                    region_type="text", confidence=0.8
                ))

        return remove_overlaps(out, iou_threshold=0.3)

    # --------- Rejection of text-like regions (for image candidates) ----------
    def filter(self, gray: np.ndarray, regions: List[ImageRegion]) -> List[ImageRegion]:
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        kept: List[ImageRegion] = []
        for r in regions:
            reason = self.rejection_reason(gray, r)
            if reason == "invalid_bounds":
                continue
            if reason is None:
                kept.append(r)

        return kept

    def rejection_reason(self, gray: np.ndarray, region: ImageRegion) -> str | None:
        """Return a rejection reason for text-like image candidates, or None."""
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if getattr(region, "region_type", "") in self.exempt_types:
            return None

        x, y, w, h = region.x, region.y, region.width, region.height
        x = max(0, x); y = max(0, y)
        w = min(w, gray.shape[1] - x); h = min(h, gray.shape[0] - y)
        if w <= 0 or h <= 0:
            return "invalid_bounds"

        options = self.options
        area = w * h
        aspect = w / float(h) if h else 0.0
        if area < options.min_area or w < options.min_width or h < options.min_height:
            return "too_small"
        if not (options.min_diagram_aspect_ratio <= aspect <= options.max_diagram_aspect_ratio):
            return "bad_aspect_ratio"
        if not options.reject_text_like:
            return None

        roi = gray[y:y + h, x:x + w]
        metrics = self.text_like_metrics(roi)
        if metrics["density"] > options.max_text_like_density:
            return "text_like_density"

        if self._looks_like_text_fragment(metrics, options):
            return "text_like_components"
        if options.reject_rotated_text_like and self._looks_like_rotated_text(metrics, options):
            return "rotated_text_like"
        if self._looks_like_text_line(metrics, options):
            return "text_line_like"
        if self._looks_like_title_text(metrics, options):
            return "title_text_like"
        if (
            metrics["component_count"] >= options.text_like_min_components
            and metrics["small_component_fraction"] <= options.text_like_max_small_component_fraction
            and self._has_regular_text_projection(roi)
        ):
            return "regular_text_projection"
        return None

    def text_like_metrics(self, roi: np.ndarray) -> dict[str, float]:
        """Compute OCR-free text-likeness metrics for a candidate crop."""
        _, inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        area = max(1, int(inv.shape[0] * inv.shape[1]))
        density = float(np.mean(inv > 0))

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        component_count = int(max(0, num_labels - 1))
        median_area = float(np.median(areas)) if areas.size else 0.0
        mean_area = float(np.mean(areas)) if areas.size else 0.0
        small_fraction = float(np.mean(areas < 80)) if areas.size else 0.0

        horizontal_projection = np.sum(inv > 0, axis=1)
        vertical_projection = np.sum(inv > 0, axis=0)
        row_occupancy = float(np.mean(horizontal_projection > 0))
        col_occupancy = float(np.mean(vertical_projection > 0))

        h, w = inv.shape[:2]
        return {
            "width": float(w),
            "height": float(h),
            "area": float(area),
            "aspect_ratio": float(w / h) if h else 0.0,
            "density": density,
            "component_count": float(component_count),
            "median_component_area": median_area,
            "mean_component_area": mean_area,
            "small_component_fraction": small_fraction,
            "row_occupancy": row_occupancy,
            "col_occupancy": col_occupancy,
        }

    def _looks_like_text_fragment(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        return (
            metrics["component_count"] >= options.text_like_min_components
            and options.text_like_min_density <= metrics["density"] <= options.text_like_max_density
            and options.text_like_min_median_component_area <= metrics["median_component_area"] <= options.text_like_max_median_component_area
            and metrics["small_component_fraction"] <= options.text_like_max_small_component_fraction
        )

    def _looks_like_rotated_text(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        narrow_vertical = metrics["aspect_ratio"] <= options.vertical_text_max_aspect_ratio
        dense_repeated = (
            metrics["component_count"] >= options.text_like_min_components
            and metrics["density"] >= options.text_like_min_density
            and metrics["row_occupancy"] >= options.rotated_text_min_row_occupancy
            and metrics["col_occupancy"] >= options.rotated_text_min_col_occupancy
        )
        return bool(narrow_vertical and dense_repeated)

    def _looks_like_title_text(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        return (
            metrics["component_count"] <= options.title_text_max_components
            and metrics["aspect_ratio"] > 1.8
            and metrics["density"] >= 0.06
            and metrics["row_occupancy"] <= options.title_text_max_row_occupancy
            and metrics["col_occupancy"] >= options.title_text_min_col_occupancy
        )

    def _looks_like_text_line(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        return (
            metrics["height"] <= options.text_line_max_height
            and metrics["aspect_ratio"] >= options.text_line_min_aspect_ratio
            and options.text_line_min_density <= metrics["density"] <= options.text_line_max_density
            and metrics["row_occupancy"] >= options.title_text_max_row_occupancy
            and metrics["col_occupancy"] >= options.text_line_min_col_occupancy
        )

    def _has_regular_text_projection(self, roi: np.ndarray) -> bool:
        try:
            hp = np.sum(roi < 128, axis=1)
            if hp.size > 10:
                ph, _ = find_peaks(hp, distance=self.peak_distance)
                if ph.size > 10 and np.std(np.diff(ph)) < self.proj_std_max:
                    logger.debug("Reject: horizontal regularity (std<%.1f)", self.proj_std_max)
                    return True
        except Exception:
            pass

        try:
            vp = np.sum(roi < 128, axis=0)
            if vp.size > 10:
                pv, _ = find_peaks(vp, distance=self.peak_distance)
                if pv.size > 10 and np.std(np.diff(pv)) < self.proj_std_max:
                    logger.debug("Reject: vertical regularity (std<%.1f)", self.proj_std_max)
                    return True
        except Exception:
            pass

        return False
