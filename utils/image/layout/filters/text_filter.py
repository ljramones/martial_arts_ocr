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

        # Exempt region types. Keep empty by default: real-page review showed
        # heuristic "figure" candidates can be plain text fragments.
        self.exempt_types = set(cfg.get('filter_text_exempt_types', []))

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
        return self.candidate_diagnostics(gray, region).get("rejection_reason")

    def candidate_diagnostics(self, gray: np.ndarray, region: ImageRegion) -> Dict[str, Any]:
        """Return scoring diagnostics plus the final accept/reject decision."""
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if getattr(region, "region_type", "") in self.exempt_types:
            return {
                "rejection_reason": None,
                "accepted_reason": "exempt_region_type",
                "features": {"region_type": getattr(region, "region_type", None)},
                "scores": {
                    "text_like_score": 0.0,
                    "figure_like_score": 1.0,
                    "photo_like_score": 0.0,
                    "sparse_symbol_score": 0.0,
                    "crop_quality_score": 1.0,
                },
            }

        x, y, w, h = region.x, region.y, region.width, region.height
        x = max(0, x); y = max(0, y)
        w = min(w, gray.shape[1] - x); h = min(h, gray.shape[0] - y)
        base = {
            "features": {
                "region_type": getattr(region, "region_type", None),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            },
            "scores": {
                "text_like_score": 0.0,
                "figure_like_score": 0.0,
                "photo_like_score": 0.0,
                "sparse_symbol_score": 0.0,
                "crop_quality_score": 0.0,
            },
        }
        if w <= 0 or h <= 0:
            return {**base, "rejection_reason": "invalid_bounds", "accepted_reason": None}

        options = self.options
        area = w * h
        aspect = w / float(h) if h else 0.0
        if area < options.min_area or w < options.min_width or h < options.min_height:
            return {**base, "rejection_reason": "too_small", "accepted_reason": None}
        if not (options.min_diagram_aspect_ratio <= aspect <= options.max_diagram_aspect_ratio):
            return {**base, "rejection_reason": "bad_aspect_ratio", "accepted_reason": None}
        if not options.reject_text_like:
            return {**base, "rejection_reason": None, "accepted_reason": "text_filter_disabled"}

        roi = gray[y:y + h, x:x + w]
        metrics = self.text_like_metrics(roi)
        metrics["page_area_ratio"] = float(area) / float(max(1, gray.shape[0] * gray.shape[1]))
        scores = self._candidate_scores(metrics, options)
        features = {**base["features"], **metrics}
        old_reason = self._legacy_rejection_reason(metrics, options)
        visual_score = max(
            scores["figure_like_score"],
            scores["photo_like_score"],
            scores["sparse_symbol_score"],
        )

        reason = None
        if (
            old_reason in {"regular_text_projection", "text_line_like", "title_text_like"}
            and (
                (
                    scores["text_like_score"] >= options.text_score_reject_threshold
                    and scores["photo_like_score"] < options.broad_crop_visual_override_threshold
                )
                or (
                    metrics["max_dark_component_area_ratio"] < 0.20
                    and visual_score < options.broad_crop_visual_override_threshold
                )
            )
        ):
            reason = old_reason
        elif (
            old_reason == "regular_text_projection"
            and scores["photo_like_score"] < options.broad_crop_visual_override_threshold
            and (
                scores["text_like_score"] >= 0.55
                or (
                    scores["figure_like_score"] < 0.75
                    and metrics["max_dark_component_area_ratio"] < 0.12
                )
            )
        ):
            reason = old_reason
        elif (
            old_reason == "text_like_density"
            and bool(metrics.get("regular_text_projection", 0.0))
            and scores["text_like_score"] >= 0.55
        ):
            reason = old_reason
        elif old_reason and visual_score < options.visual_score_override_threshold:
            reason = old_reason
        elif (
            scores["text_like_score"] >= options.text_score_reject_threshold
            and visual_score < options.visual_score_override_threshold
        ):
            reason = "scored_text_like"
        elif (
            metrics["page_area_ratio"] >= options.broad_crop_area_ratio
            and scores["text_like_score"] >= 0.55
            and visual_score < options.broad_crop_visual_override_threshold
        ):
            reason = "broad_text_like_crop"

        if reason:
            return {
                "rejection_reason": reason,
                "accepted_reason": None,
                "features": features,
                "scores": scores,
            }

        return {
            "rejection_reason": None,
            "accepted_reason": self._accepted_reason(scores),
            "features": features,
            "scores": scores,
        }

    def _legacy_rejection_reason(self, metrics: dict[str, float], options: RegionDetectionOptions) -> str | None:
        if metrics["density"] > options.max_text_like_density:
            return "text_like_density"

        if self._looks_like_sparse_text_band(metrics, options):
            return "sparse_text_band"
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
            and bool(metrics.get("regular_text_projection", 0.0))
        ):
            return "regular_text_projection"
        if (
            bool(metrics.get("regular_text_projection", 0.0))
            and metrics["aspect_ratio"] >= 1.5
            and metrics["component_count"] >= 20
            and metrics["max_dark_component_area_ratio"] < 0.02
        ):
            return "regular_text_projection"
        if (
            bool(metrics.get("regular_text_projection", 0.0))
            and metrics["density"] >= 0.08
            and metrics["component_count"] <= options.title_text_max_components + 20
            and metrics["max_dark_component_area_ratio"] < 0.12
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
        edge_density, hough_line_count = self._edge_metrics(roi)
        max_dark_component_ratio, large_dark_component_count = self._dark_component_metrics(roi)
        horizontal_peak_count, horizontal_peak_std = self._projection_peak_metrics(horizontal_projection, max(1, w))
        vertical_peak_count, vertical_peak_std = self._projection_peak_metrics(vertical_projection, max(1, h))
        component_area_ratio = mean_area / max(1.0, median_area)
        return {
            "width": float(w),
            "height": float(h),
            "area": float(area),
            "aspect_ratio": float(w / h) if h else 0.0,
            "density": density,
            "component_count": float(component_count),
            "median_component_area": median_area,
            "mean_component_area": mean_area,
            "component_area_ratio": float(component_area_ratio),
            "small_component_fraction": small_fraction,
            "row_occupancy": row_occupancy,
            "col_occupancy": col_occupancy,
            "edge_density": edge_density,
            "hough_line_count": float(hough_line_count),
            "gray_std": float(np.std(roi)),
            "dark80_fraction": float(np.mean(roi < 80)),
            "dark120_fraction": float(np.mean(roi < 120)),
            "max_dark_component_area_ratio": max_dark_component_ratio,
            "large_dark_component_count": float(large_dark_component_count),
            "horizontal_peak_count": float(horizontal_peak_count),
            "horizontal_peak_spacing_std": float(horizontal_peak_std),
            "vertical_peak_count": float(vertical_peak_count),
            "vertical_peak_spacing_std": float(vertical_peak_std),
            "regular_text_projection": float(
                self._regular_peak_pattern(horizontal_peak_count, horizontal_peak_std)
                or self._regular_peak_pattern(vertical_peak_count, vertical_peak_std)
            ),
        }

    def _looks_like_text_fragment(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        return (
            metrics["component_count"] >= options.text_like_min_components
            and options.text_like_min_density <= metrics["density"] <= options.text_like_max_density
            and options.text_like_min_median_component_area <= metrics["median_component_area"] <= options.text_like_max_median_component_area
            and metrics["small_component_fraction"] <= options.text_like_max_small_component_fraction
        )

    def _candidate_scores(
        self,
        metrics: dict[str, float],
        options: RegionDetectionOptions,
    ) -> dict[str, float]:
        text_score = 0.0
        if options.text_like_min_density <= metrics["density"] <= options.text_like_max_density:
            text_score += 0.18
        if (
            metrics["component_count"] >= options.text_like_min_components
            and metrics["small_component_fraction"] >= 0.85
            and metrics["median_component_area"] <= 90.0
        ):
            text_score += 0.35
        if metrics["row_occupancy"] >= 0.70 and metrics["col_occupancy"] >= 0.70:
            text_score += 0.15
        if metrics["regular_text_projection"]:
            text_score += 0.25
        if (
            metrics["height"] <= options.text_line_max_height
            and metrics["aspect_ratio"] >= options.text_line_min_aspect_ratio
        ):
            text_score += 0.20
        if metrics["page_area_ratio"] >= options.broad_crop_area_ratio:
            text_score += 0.18
        if metrics["aspect_ratio"] <= options.vertical_text_max_aspect_ratio + 0.15:
            text_score += 0.12
        text_score = min(1.0, text_score)

        figure_score = 0.0
        if 0.015 <= metrics["edge_density"] <= 0.28:
            figure_score += 0.18
        if metrics["hough_line_count"] >= 3:
            figure_score += 0.25
        if metrics["max_dark_component_area_ratio"] >= 0.025:
            figure_score += 0.25
        if (
            metrics["component_area_ratio"] >= options.labeled_diagram_min_component_area_ratio
            and 0.25 <= metrics["small_component_fraction"] <= 0.85
            and metrics["density"] <= options.labeled_diagram_max_density
        ):
            figure_score += 0.25
        figure_score = min(1.0, figure_score)

        photo_score = 0.0
        if (
            metrics["gray_std"] >= options.photo_like_min_std
            and metrics["dark120_fraction"] >= options.photo_like_min_dark_fraction
            and metrics["edge_density"] >= options.photo_like_min_edge_density
            and min(metrics["width"], metrics["height"]) >= options.visual_min_dimension_for_photo
        ):
            photo_score += 0.58
        min_dimension = min(metrics["width"], metrics["height"])
        if metrics["max_dark_component_area_ratio"] >= 0.08 and min_dimension >= 80:
            photo_score += 0.25
        if metrics["max_dark_component_area_ratio"] >= 0.15 and min_dimension >= 80:
            photo_score += 0.50
        if metrics["large_dark_component_count"] >= 1 and min_dimension >= 80:
            photo_score += 0.10
        photo_score = min(1.0, photo_score)

        sparse_symbol_score = 0.0
        if (
            metrics["density"] <= 0.16
            and metrics["edge_density"] >= 0.015
            and metrics["hough_line_count"] >= 2
            and (
                metrics["max_dark_component_area_ratio"] >= 0.02
                or metrics["component_count"] <= 24
            )
        ):
            sparse_symbol_score += 0.65
        if metrics["max_dark_component_area_ratio"] >= 0.02 and metrics["density"] <= 0.25:
            sparse_symbol_score += 0.20
        sparse_symbol_score = min(1.0, sparse_symbol_score)

        crop_quality = 1.0
        if metrics["page_area_ratio"] >= options.broad_crop_area_ratio:
            crop_quality -= min(0.45, (metrics["page_area_ratio"] - options.broad_crop_area_ratio) * 1.4)
        if metrics["aspect_ratio"] < options.min_diagram_aspect_ratio or metrics["aspect_ratio"] > options.max_diagram_aspect_ratio:
            crop_quality -= 0.4
        if text_score > 0.65 and max(figure_score, photo_score, sparse_symbol_score) < 0.6:
            crop_quality -= 0.25
        crop_quality = max(0.0, min(1.0, crop_quality))

        return {
            "text_like_score": round(text_score, 4),
            "figure_like_score": round(figure_score, 4),
            "photo_like_score": round(photo_score, 4),
            "sparse_symbol_score": round(sparse_symbol_score, 4),
            "crop_quality_score": round(crop_quality, 4),
        }

    @staticmethod
    def _accepted_reason(scores: dict[str, float]) -> str:
        visual = {
            "figure_like": scores["figure_like_score"],
            "photo_like": scores["photo_like_score"],
            "sparse_symbol": scores["sparse_symbol_score"],
        }
        best = max(visual, key=visual.get)
        if visual[best] >= 0.58:
            return best
        return "low_text_evidence"

    def _looks_like_labeled_diagram(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        """Preserve drawings that contain labels plus larger strokes/shapes.

        Labeled diagrams often have enough glyph-like components to trip the
        text-fragment rule, but their component sizes are mixed: small label
        glyphs coexist with larger arrows, outlines, or hand-drawn strokes.
        Typewritten text blocks are more uniform, so their mean/median component
        area ratio stays close to 1.
        """
        if not options.preserve_labeled_diagrams:
            return False
        median_area = max(1.0, metrics["median_component_area"])
        component_area_ratio = metrics["mean_component_area"] / median_area
        return (
            metrics["component_count"] >= options.text_like_min_components
            and options.text_like_min_density <= metrics["density"] <= options.labeled_diagram_max_density
            and metrics["small_component_fraction"] >= options.labeled_diagram_min_small_component_fraction
            and component_area_ratio >= options.labeled_diagram_min_component_area_ratio
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

    def _looks_like_sparse_text_band(self, metrics: dict[str, float], options: RegionDetectionOptions) -> bool:
        return (
            metrics["height"] <= options.sparse_text_band_max_height
            and metrics["aspect_ratio"] >= options.sparse_text_band_min_aspect_ratio
            and options.sparse_text_band_min_density <= metrics["density"] <= options.sparse_text_band_max_density
            and metrics["median_component_area"] <= options.sparse_text_band_max_median_component_area
            and metrics["small_component_fraction"] >= options.sparse_text_band_min_small_component_fraction
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

    @staticmethod
    def _edge_metrics(roi: np.ndarray) -> tuple[float, int]:
        edges = cv2.Canny(roi, 50, 150)
        edge_density = float(np.mean(edges > 0))
        min_len = max(12, int(min(roi.shape[:2]) * 0.20))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            35,
            minLineLength=min_len,
            maxLineGap=8,
        )
        return edge_density, int(len(lines)) if lines is not None else 0

    @staticmethod
    def _dark_component_metrics(roi: np.ndarray) -> tuple[float, int]:
        _, binary = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if not areas.size:
            return 0.0, 0
        area = max(1, int(roi.shape[0] * roi.shape[1]))
        return float(np.max(areas) / area), int(np.sum(areas >= 500))

    @staticmethod
    def _projection_peak_metrics(projection: np.ndarray, span: int) -> tuple[int, float]:
        if projection.size <= 10:
            return 0, 999.0
        prominence = max(1.0, span * 0.02)
        peaks, _ = find_peaks(projection, distance=8, prominence=prominence)
        if peaks.size <= 2:
            return int(peaks.size), 999.0
        return int(peaks.size), float(np.std(np.diff(peaks)))

    @staticmethod
    def _regular_peak_pattern(count: int | float, spacing_std: float) -> bool:
        return bool(count >= 5 and spacing_std < 8.0)
