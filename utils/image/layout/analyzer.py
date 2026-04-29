from __future__ import annotations
import cv2
import numpy as np
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from config import get_config
from utils.image.regions.core_image import ImageRegion
from .detectors.figure import FigureDetector
from .detectors.contours import ContourDetector
from .detectors.variance import VarianceDetector
from .detectors.uniform import UniformDetector
from .filters.text_filter import TextRegionFilter
from .utils.halo import halo_ok
from .utils.masks import apply_nontext_mask

from .options import RegionDetectionOptions
from .post.merge import consolidate_regions, merge_overlapping

from .detectors.yolo_figure import YOLOFigureDetector



logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    LayoutAnalyzer – unified page layout orchestrator.

    Delegates detection to:
      - FigureDetector
      - ContourDetector
      - VarianceDetector
      - UniformDetector
    Then merges, filters text-like noise, and returns structured region metadata.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        cfg = get_config().LAYOUT_DETECTION.copy()
        if config_override:
            cfg.update(config_override)
        self.cfg = cfg

        # Halo closure
        self._halo = lambda gray, x, y, w, h: halo_ok(
            gray, x, y, w, h,
            ring=int(self.cfg.get('halo_ring', 4)),
            min_white=float(self.cfg.get('halo_min_white', 0.85)),
            white_thresh=200
        )

        # Detectors
        use_yolo = bool(self.cfg.get("use_yolo_figure", False)) and YOLOFigureDetector.available
        if self.cfg.get("use_yolo_figure", False) and not YOLOFigureDetector.available:
            logger.warning("YOLO figure detector requested but ultralytics is unavailable; using heuristic detector")

        if use_yolo:
            # YOLO-based figure detector (primary)
            self.figure = YOLOFigureDetector(self.cfg)
        else:
            # Heuristic figure detector (legacy)
            self.figure = FigureDetector(self.cfg, halo_check=self._halo)

        self.contours = ContourDetector(self.cfg, halo_check=self._halo)
        self.variance = VarianceDetector(self.cfg)
        self.uniform = UniformDetector(self.cfg)

        # Filter
        self.text_filter = TextRegionFilter(self.cfg)
        self.region_options = RegionDetectionOptions.from_config(self.cfg)

        # Which detectors to run (configurable)
        self.enabled_detectors = self.cfg.get(
            "enabled_detectors",
            ["figure", "contours", "variance", "uniform"]
        )

    def debug_draw_regions(self, image: np.ndarray, regions: List[ImageRegion], color=(0, 255, 0)) -> np.ndarray:
        out = image.copy()
        for r in regions:
            cv2.rectangle(out, (r.x, r.y), (r.x + r.width, r.y + r.height), color, 2)
        return out

    @staticmethod
    def _to_gray_u8(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def detect_image_regions(
            self,
            image: np.ndarray,
            nontext_mask: Optional[np.ndarray] = None,
            ocr_text_boxes: Optional[list[Any]] = None,
    ) -> List[ImageRegion]:
        gray = self._to_gray_u8(image)
        if isinstance(nontext_mask, np.ndarray):
            gray = apply_nontext_mask(gray, nontext_mask)

        candidates = self._detect_image_region_candidates(gray)

        # Optional: allow bypassing text-like rejection (A/B / debugging)
        if bool(self.cfg.get("filter_text_like", True)):
            filtered = self._filter_candidates_with_diagnostics(
                gray,
                candidates,
                ocr_text_boxes=ocr_text_boxes,
            )[0]
        else:
            filtered = candidates

        filtered, _events = consolidate_regions(filtered, self.region_options)

        logger.info("Image regions detected: %d", len(filtered))
        return filtered

    def detect_image_regions_with_diagnostics(
        self,
        image: np.ndarray,
        nontext_mask: Optional[np.ndarray] = None,
        ocr_text_boxes: Optional[list[Any]] = None,
    ) -> Dict[str, Any]:
        """Return accepted and rejected image candidates with rejection reasons."""
        gray = self._to_gray_u8(image)
        if isinstance(nontext_mask, np.ndarray):
            gray = apply_nontext_mask(gray, nontext_mask)

        candidates = self._detect_image_region_candidates(gray)
        accepted: List[ImageRegion] = []
        rejected: List[Dict[str, Any]] = []
        ocr_text_mask = (
            self.text_filter._build_ocr_text_mask(gray, ocr_text_boxes)
            if bool(self.cfg.get("filter_text_like", True))
            else None
        )
        for region in candidates:
            diagnostic = (
                self.text_filter.candidate_diagnostics(
                    gray,
                    region,
                    ocr_text_boxes=ocr_text_boxes,
                    ocr_text_mask=ocr_text_mask,
                )
                if bool(self.cfg.get("filter_text_like", True))
                else {"rejection_reason": None, "accepted_reason": "text_filter_disabled", "scores": {}, "features": {}}
            )
            reason = diagnostic.get("rejection_reason")
            if reason:
                rejected.append({
                    "region": region.to_dict(),
                    "rejection_reason": reason,
                    "region_role": diagnostic.get("region_role"),
                    "scores": diagnostic.get("scores", {}),
                    "features": diagnostic.get("features", {}),
                })
            else:
                accepted.append(self._region_with_diagnostics(region, diagnostic))

        accepted, consolidation_events = consolidate_regions(accepted, self.region_options)

        return {
            "accepted_regions": accepted,
            "accepted": [region.to_dict() for region in accepted],
            "rejected": rejected,
            "consolidation": consolidation_events,
        }

    def _filter_candidates_with_diagnostics(
        self,
        gray: np.ndarray,
        candidates: List[ImageRegion],
        *,
        ocr_text_boxes: Optional[list[Any]] = None,
    ) -> tuple[List[ImageRegion], List[Dict[str, Any]]]:
        accepted: List[ImageRegion] = []
        rejected: List[Dict[str, Any]] = []
        ocr_text_mask = self.text_filter._build_ocr_text_mask(gray, ocr_text_boxes)
        for region in candidates:
            diagnostic = self.text_filter.candidate_diagnostics(
                gray,
                region,
                ocr_text_boxes=ocr_text_boxes,
                ocr_text_mask=ocr_text_mask,
            )
            reason = diagnostic.get("rejection_reason")
            if reason:
                rejected.append({
                    "region": region.to_dict(),
                    "rejection_reason": reason,
                    "region_role": diagnostic.get("region_role"),
                    "scores": diagnostic.get("scores", {}),
                    "features": diagnostic.get("features", {}),
                })
            else:
                accepted.append(self._region_with_diagnostics(region, diagnostic))
        return accepted, rejected

    @staticmethod
    def _region_with_diagnostics(region: ImageRegion, diagnostic: Dict[str, Any]) -> ImageRegion:
        metadata = dict(getattr(region, "metadata", {}) or {})
        metadata.update({
            "accepted_reason": diagnostic.get("accepted_reason"),
            "text_like_score": diagnostic.get("scores", {}).get("text_like_score"),
            "figure_like_score": diagnostic.get("scores", {}).get("figure_like_score"),
            "photo_like_score": diagnostic.get("scores", {}).get("photo_like_score"),
            "sparse_symbol_score": diagnostic.get("scores", {}).get("sparse_symbol_score"),
            "crop_quality_score": diagnostic.get("scores", {}).get("crop_quality_score"),
            "region_role": diagnostic.get("region_role"),
            "diagnostic_features": diagnostic.get("features", {}),
        })
        return replace(region, metadata=metadata)

    def _detect_image_region_candidates(self, gray: np.ndarray) -> List[ImageRegion]:
        regions: List[ImageRegion] = []

        # Order matters: fast/well-isolated first
        if "figure" in self.enabled_detectors:
            figs = self.figure.detect(gray)
            regions += figs
            logger.debug("FigureDetector found %d regions", len(figs))

        if "contours" in self.enabled_detectors and (not regions or self.cfg.get("contours_always", False)):
            cons = self.contours.detect(gray)
            regions += cons
            logger.debug("ContourDetector found %d regions", len(cons))

        if "variance" in self.enabled_detectors:
            vars_ = self.variance.detect(gray)
            regions += vars_
            logger.debug("VarianceDetector found %d regions", len(vars_))

        if "uniform" in self.enabled_detectors:
            unif = self.uniform.detect(gray)
            regions += unif
            logger.debug("UniformDetector found %d regions", len(unif))

        # Merge adjacent/overlapping diagram boxes first (cleaner outputs)
        diagrams = [r for r in regions if r.region_type == "diagram"]
        others = [r for r in regions if r.region_type != "diagram"]
        if diagrams:
            # configurable merge aggressiveness
            iou_th = float(self.cfg.get("diagram_merge_iou", 0.10))
            gap_px = int(self.cfg.get("diagram_merge_gap", 12))
            diagrams = merge_overlapping(diagrams, iou_threshold=iou_th, gap=gap_px)
        regions = others + diagrams

        return regions

    def detect_text_regions(
        self,
        image: np.ndarray,
        nontext_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[ImageRegion]:
        # Keep your MSER-based approach in TextRegionFilter
        gray = self._to_gray_u8(image)
        if isinstance(nontext_mask, np.ndarray):
            gray = apply_nontext_mask(gray, nontext_mask)
        # Reuse the filter’s MSER detect (centralizes thresholds)
        return self.text_filter.detect_mser(gray, **kwargs)

    def analyze_page_layout(
        self,
        image: np.ndarray,
        proc: Optional[Any] = None,
        nontext_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        mask = nontext_mask
        if mask is None and proc is not None and hasattr(proc, "get_last_nontext_mask"):
            try:
                mask = proc.get_last_nontext_mask()
            except Exception:
                mask = None

        text_regions = self.detect_text_regions(image, nontext_mask=mask)
        image_regions = self.detect_image_regions(image, nontext_mask=mask)

        h, w = image.shape[:2]
        total = max(1, h * w)
        text_area = sum(r.area for r in text_regions)
        image_area = sum(r.area for r in image_regions)
        mask_keep = None
        if isinstance(mask, np.ndarray) and mask.size == total:
            mask_keep = float(np.mean(mask == 255))

        return {
            "text_regions": [r.to_dict() for r in text_regions],
            "image_regions": [r.to_dict() for r in image_regions],
            "statistics": {
                "num_text_regions": len(text_regions),
                "num_image_regions": len(image_regions),
                "text_coverage": text_area / total,
                "image_coverage": image_area / total,
                "page_size": (w, h),
                "nontext_keep_ratio": mask_keep,
            },
        }
