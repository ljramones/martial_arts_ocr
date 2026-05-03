from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import cv2
import numpy as np

from utils.image.regions.core_image import ImageRegion
from utils.image.layout.filters.text_filter import TextRegionFilter
from . import BaseDetector


logger = logging.getLogger(__name__)


class MultiFigureRowDetector(BaseDetector):
    """Propose sibling figure panels in horizontal rows for review workflows.

    This detector is intentionally conservative and advisory. It is meant to
    improve review-mode recall on pages with repeated figure/photo panels, while
    leaving final filtering and consolidation to the existing layout pipeline.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        halo_check: Callable[[np.ndarray, int, int, int, int], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.halo_ok = halo_check
        self.min_area = int(cfg.get("multi_figure_row_min_area", 8_000))
        self.min_width = int(cfg.get("multi_figure_row_min_width", 70))
        self.min_height = int(cfg.get("multi_figure_row_min_height", 90))
        self.max_area_ratio = float(cfg.get("multi_figure_row_max_area_ratio", 0.25))
        self.min_aspect = float(cfg.get("multi_figure_row_min_aspect", 0.20))
        self.max_aspect = float(cfg.get("multi_figure_row_max_aspect", 4.0))
        self.min_edge_density = float(cfg.get("multi_figure_row_min_edge_density", 0.006))
        self.max_edge_density = float(cfg.get("multi_figure_row_max_edge_density", 0.22))
        self.min_dark_fraction = float(cfg.get("multi_figure_row_min_dark_fraction", 0.012))
        self.max_dark_fraction = float(cfg.get("multi_figure_row_max_dark_fraction", 0.55))
        self.min_band_members = int(cfg.get("multi_figure_row_min_band_members", 2))
        self.band_center_tolerance_ratio = float(cfg.get("multi_figure_row_band_center_tolerance_ratio", 0.45))
        self.max_candidates = int(cfg.get("multi_figure_row_topk", 12))
        self.child_min_area = int(cfg.get("broad_rejected_child_min_area", max(3_000, self.min_area // 2)))
        self.child_min_width = int(cfg.get("broad_rejected_child_min_width", max(45, self.min_width // 2)))
        self.child_min_height = int(cfg.get("broad_rejected_child_min_height", max(55, self.min_height // 2)))
        self.child_max_candidates = int(cfg.get("broad_rejected_child_topk", 8))
        self.child_text_reject_threshold = float(cfg.get("broad_rejected_child_text_reject_threshold", 0.55))
        self.child_min_large_component_ratio = float(cfg.get("broad_rejected_child_min_large_component_ratio", 0.025))
        self.child_min_figure_score = float(cfg.get("broad_rejected_child_min_figure_score", 0.55))
        self.child_min_photo_score = float(cfg.get("broad_rejected_child_min_photo_score", 0.50))
        self.text_filter = TextRegionFilter(cfg)
        self.last_diagnostics: Dict[str, Any] = {
            "detector": "multi_figure_rows",
            "raw_count": 0,
            "returned_count": 0,
            "candidates": [],
        }

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
            h, w = gray.shape[:2]
            mask = self._visual_mask(gray)
            raw, rejected = self._component_candidates(gray, mask)
            banded, bands = self._keep_row_siblings(raw)
            regions = [
                self._region(candidate, page_shape=(h, w), sibling_index=index)
                for index, candidate in enumerate(banded[: self.max_candidates], start=1)
            ]
            self.last_diagnostics = {
                "detector": "multi_figure_rows",
                "page_size": [int(w), int(h)],
                "thresholds": self._thresholds(),
                "raw_count": len(raw),
                "returned_count": len(regions),
                "rejected_count": len(rejected),
                "rejected": rejected[: self.max_candidates],
                "bands_considered": bands,
                "candidates": [region.to_dict() for region in regions],
            }
            logger.debug("MultiFigureRowDetector found %d regions", len(regions))
            return regions
        except Exception as exc:
            logger.error("Multi-figure row detection failed: %s", exc)
            self.last_diagnostics = {
                "detector": "multi_figure_rows",
                "status": "failed",
                "error": str(exc),
                "raw_count": 0,
                "returned_count": 0,
                "rejected_count": 0,
                "rejected": [],
                "bands_considered": [],
                "candidates": [],
            }
            return []

    def propose_child_visuals(
        self,
        gray_in: np.ndarray,
        rejected_records: list[dict[str, Any]],
    ) -> list[ImageRegion]:
        """Propose child visual regions inside broad rejected text-like parents.

        This is review-mode-only support. It does not accept the broad parent;
        it offers smaller visual children for human inspection.
        """
        gray = self._to_gray_u8(gray_in)
        h_img, w_img = gray.shape[:2]
        output: list[ImageRegion] = []
        diagnostics: list[dict[str, Any]] = []
        allowed_reasons = {
            "text_like_components",
            "regular_text_projection",
            "broad_text_like_crop",
            "scored_text_like",
            "text_like_density",
        }
        for parent_index, record in enumerate(rejected_records, start=1):
            reason = str(record.get("rejection_reason") or "")
            if reason not in allowed_reasons:
                continue
            parent_region = _region_from_record(record.get("region"))
            if parent_region is None:
                continue
            if parent_region.area < max(self.min_area * 2, 20_000):
                continue
            x, y, width, height = parent_region.x, parent_region.y, parent_region.width, parent_region.height
            roi = gray[y : y + height, x : x + width]
            raw, rejected = self._child_component_candidates(
                roi,
                offset=(x, y),
                parent_area=parent_region.area,
            )
            diagnostics.append(
                {
                    "parent_bbox": [x, y, width, height],
                    "parent_rejection_reason": reason,
                    "raw_count": len(raw),
                    "rejected_count": len(rejected),
                    "rejected": rejected[: self.child_max_candidates],
                }
            )
            for child_index, candidate in enumerate(raw[: self.child_max_candidates], start=1):
                output.append(
                    self._child_region(
                        candidate,
                        page_shape=(h_img, w_img),
                        parent_bbox=[x, y, width, height],
                        parent_reason=reason,
                        parent_index=parent_index,
                        child_index=child_index,
                    )
                )

        self.last_child_diagnostics = {
            "detector": "broad_rejected_child_visuals",
            "parent_count": len(diagnostics),
            "returned_count": len(output),
            "parents": diagnostics,
            "candidates": [region.to_dict() for region in output],
        }
        return output

    def _visual_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 35, 120)
        dark = cv2.inRange(blurred, 0, 185)
        mask = cv2.bitwise_or(edges, dark)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        return mask

    def _component_candidates(self, gray: np.ndarray, mask: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        h_img, w_img = gray.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        max_area = h_img * w_img * self.max_area_ratio
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            base = {"bbox": [int(x), int(y), int(width), int(height)], "area": int(area)}
            if area < self.min_area:
                rejected.append({**base, "reason": "area_too_small", "threshold": self.min_area})
                continue
            if area > max_area:
                rejected.append({**base, "reason": "area_too_large", "threshold": int(max_area)})
                continue
            if width < self.min_width:
                rejected.append({**base, "reason": "width_too_small", "threshold": self.min_width})
                continue
            if height < self.min_height:
                rejected.append({**base, "reason": "height_too_small", "threshold": self.min_height})
                continue
            aspect = width / float(height or 1)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                rejected.append({**base, "reason": "aspect_invalid", "value": round(float(aspect), 4)})
                continue
            roi = gray[y : y + height, x : x + width]
            roi_edges = cv2.Canny(roi, 50, 150)
            edge_density = float(np.sum(roi_edges > 0)) / float(max(1, area))
            dark_fraction = float(np.sum(roi < 190)) / float(max(1, area))
            if not (self.min_edge_density <= edge_density <= self.max_edge_density):
                rejected.append({**base, "reason": "edge_density_out_of_range", "value": round(edge_density, 4)})
                continue
            if not (self.min_dark_fraction <= dark_fraction <= self.max_dark_fraction):
                rejected.append({**base, "reason": "dark_fraction_out_of_range", "value": round(dark_fraction, 4)})
                continue
            candidates.append(
                {
                    "bbox": (int(x), int(y), int(width), int(height)),
                    "area": int(area),
                    "edge_density": edge_density,
                    "dark_fraction": dark_fraction,
                    "center_y": y + height / 2.0,
                }
            )
        return (
            sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])),
            sorted(rejected, key=lambda item: (item["bbox"][1], item["bbox"][0])),
        )

    def _child_component_candidates(
        self,
        roi: np.ndarray,
        *,
        offset: tuple[int, int],
        parent_area: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        mask = self._visual_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        offset_x, offset_y = offset
        max_area = parent_area * 0.55
        for contour in contours:
            local_x, local_y, width, height = cv2.boundingRect(contour)
            x = offset_x + local_x
            y = offset_y + local_y
            area = width * height
            base = {"bbox": [int(x), int(y), int(width), int(height)], "area": int(area)}
            if area < self.child_min_area:
                rejected.append({**base, "reason": "area_too_small", "threshold": self.child_min_area})
                continue
            if area > max_area:
                rejected.append({**base, "reason": "area_too_large", "threshold": int(max_area)})
                continue
            if width < self.child_min_width or height < self.child_min_height:
                rejected.append({**base, "reason": "dimension_too_small"})
                continue
            aspect = width / float(height or 1)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                rejected.append({**base, "reason": "aspect_invalid", "value": round(float(aspect), 4)})
                continue
            child = roi[local_y : local_y + height, local_x : local_x + width]
            edge_density = float(np.sum(cv2.Canny(child, 50, 150) > 0)) / float(max(1, area))
            dark_fraction = float(np.sum(child < 190)) / float(max(1, area))
            if edge_density < self.min_edge_density:
                rejected.append({**base, "reason": "edge_density_low", "value": round(edge_density, 4)})
                continue
            if dark_fraction < self.min_dark_fraction:
                rejected.append({**base, "reason": "dark_fraction_low", "value": round(dark_fraction, 4)})
                continue
            if _looks_like_caption_strip(width, height, edge_density=edge_density, dark_fraction=dark_fraction):
                rejected.append({**base, "reason": "caption_strip_like"})
                continue
            child_text = self._child_text_diagnostics(roi, local_x, local_y, width, height)
            child_reason = self._child_rejection_reason(child_text)
            if child_reason:
                rejected.append({
                    **base,
                    "reason": child_reason,
                    "scores": child_text.get("scores", {}),
                    "features": _compact_child_features(child_text.get("features", {})),
                })
                continue
            candidates.append(
                {
                    "bbox": (int(x), int(y), int(width), int(height)),
                    "area": int(area),
                    "edge_density": edge_density,
                    "dark_fraction": dark_fraction,
                    "center_y": y + height / 2.0,
                    "child_visual_score": round(min(1.0, edge_density * 3.0 + dark_fraction), 4),
                    "scores": child_text.get("scores", {}),
                    "features": _compact_child_features(child_text.get("features", {})),
                }
            )
        contour_candidates, contour_rejected = self._child_contour_candidates(
            roi,
            offset=offset,
            existing=candidates,
        )
        candidates.extend(contour_candidates)
        rejected.extend(contour_rejected)
        return (
            sorted(candidates, key=lambda item: item["area"], reverse=True),
            sorted(rejected, key=lambda item: (item["bbox"][1], item["bbox"][0])),
        )

    def _child_contour_candidates(
        self,
        roi: np.ndarray,
        *,
        offset: tuple[int, int],
        existing: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from .contours import ContourDetector

        detector = ContourDetector(
            {
                **self.cfg,
                "contour_min_area": self.child_min_area,
                "contour_max_area_ratio": 0.9,
                "contour_topk": self.child_max_candidates,
                "contours_require_halo": False,
                "contour_left_bias_xmax": 1.0,
            },
            halo_check=lambda *_args: True,
        )
        offset_x, offset_y = offset
        output: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for region in detector.detect(roi):
            bbox = (offset_x + region.x, offset_y + region.y, region.width, region.height)
            base = {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "area": int(region.area),
                "proposal_source": "child_contour",
            }
            if region.width < self.child_min_width or region.height < self.child_min_height:
                rejected.append({**base, "reason": "dimension_too_small"})
                continue
            aspect = region.width / float(region.height or 1)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                rejected.append({**base, "reason": "aspect_invalid", "value": round(float(aspect), 4)})
                continue
            if _duplicates_existing(bbox, existing + output):
                rejected.append({**base, "reason": "duplicate"})
                continue
            metadata = dict(getattr(region, "metadata", {}) or {})
            edge_density = float(metadata.get("edge_density") or 0.0)
            dark_fraction = float(np.mean(roi[region.y : region.y + region.height, region.x : region.x + region.width] < 190))
            if _looks_like_caption_strip(region.width, region.height, edge_density=edge_density, dark_fraction=dark_fraction):
                rejected.append({**base, "reason": "caption_strip_like"})
                continue
            child_text = self._child_text_diagnostics(roi, region.x, region.y, region.width, region.height)
            child_reason = self._child_rejection_reason(child_text)
            if child_reason:
                rejected.append({
                    **base,
                    "reason": child_reason,
                    "scores": child_text.get("scores", {}),
                    "features": _compact_child_features(child_text.get("features", {})),
                })
                continue
            output.append(
                {
                    "bbox": tuple(int(value) for value in bbox),
                    "area": int(region.area),
                    "edge_density": edge_density,
                    "dark_fraction": dark_fraction,
                    "center_y": bbox[1] + bbox[3] / 2.0,
                    "child_visual_score": round(min(1.0, edge_density * 3.0 + dark_fraction), 4),
                    "proposal_source": "child_contour",
                    "scores": child_text.get("scores", {}),
                    "features": _compact_child_features(child_text.get("features", {})),
                }
            )
        return output, rejected

    def _keep_row_siblings(self, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[list[dict[str, Any]]] = []
        for candidate in candidates:
            placed = False
            _, _y, _w, height = candidate["bbox"]
            tolerance = max(30.0, height * self.band_center_tolerance_ratio)
            for row in rows:
                row_center = sum(item["center_y"] for item in row) / len(row)
                if abs(candidate["center_y"] - row_center) <= tolerance:
                    row.append(candidate)
                    placed = True
                    break
            if not placed:
                rows.append([candidate])

        kept: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for row in rows:
            row = sorted(row, key=lambda item: item["bbox"][0])
            band_bbox = _union_xywh([item["bbox"] for item in row])
            visual_band_score = round(float(sum(item["area"] for item in row) / max(1, band_bbox[2] * band_bbox[3])), 4)
            diagnostics.append({
                "band_bbox": list(band_bbox),
                "component_count": len(row),
                "accepted_count": len(row) if len(row) >= self.min_band_members else 0,
                "rejected": [] if len(row) >= self.min_band_members else [
                    {
                        "bbox": list(item["bbox"]),
                        "reason": "not_enough_siblings",
                        "threshold": self.min_band_members,
                    }
                    for item in row
                ],
            })
            if len(row) < self.min_band_members:
                continue
            for index, item in enumerate(row, start=1):
                item = dict(item)
                item["band_bbox"] = band_bbox
                item["sibling_index"] = index
                item["visual_band_score"] = visual_band_score
                kept.append(item)
        return kept, diagnostics

    def _region(self, candidate: dict[str, Any], *, page_shape: tuple[int, int], sibling_index: int) -> ImageRegion:
        x, y, width, height = candidate["bbox"]
        metadata = {
            "detector": "multi_figure_rows",
            "band_bbox": candidate.get("band_bbox"),
            "sibling_index": candidate.get("sibling_index", sibling_index),
            "visual_band_score": candidate.get("visual_band_score"),
            "edge_density": round(float(candidate.get("edge_density", 0.0)), 4),
            "dark_fraction": round(float(candidate.get("dark_fraction", 0.0)), 4),
            "reason": "multi_figure_row_sibling",
        }
        return ImageRegion(
            x=x,
            y=y,
            width=width,
            height=height,
            region_type="diagram",
            confidence=0.78,
            metadata=metadata,
        ).clamp(page_shape[1], page_shape[0])

    def _child_region(
        self,
        candidate: dict[str, Any],
        *,
        page_shape: tuple[int, int],
        parent_bbox: list[int],
        parent_reason: str,
        parent_index: int,
        child_index: int,
    ) -> ImageRegion:
        x, y, width, height = candidate["bbox"]
        metadata = {
            "detector": "broad_rejected_child_visuals",
            "parent_bbox": parent_bbox,
            "parent_rejection_reason": parent_reason,
            "parent_index": parent_index,
            "child_index": child_index,
            "child_visual_score": candidate.get("child_visual_score"),
            "edge_density": round(float(candidate.get("edge_density", 0.0)), 4),
            "dark_fraction": round(float(candidate.get("dark_fraction", 0.0)), 4),
            "proposal_source": candidate.get("proposal_source", "child_component"),
            "scores": candidate.get("scores", {}),
            "diagnostic_features": candidate.get("features", {}),
            "reason": "visual_child_from_rejected_parent",
            "region_role": "image_candidate",
            "needs_review": True,
            "source": "broad_rejected_child_visuals",
        }
        return ImageRegion(
            x=x,
            y=y,
            width=width,
            height=height,
            region_type="diagram",
            confidence=0.62,
            metadata=metadata,
        ).clamp(page_shape[1], page_shape[0])

    def _thresholds(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "max_area_ratio": self.max_area_ratio,
            "min_aspect": self.min_aspect,
            "max_aspect": self.max_aspect,
            "min_edge_density": self.min_edge_density,
            "max_edge_density": self.max_edge_density,
            "min_dark_fraction": self.min_dark_fraction,
            "max_dark_fraction": self.max_dark_fraction,
            "min_band_members": self.min_band_members,
            "child_text_reject_threshold": self.child_text_reject_threshold,
            "child_min_large_component_ratio": self.child_min_large_component_ratio,
            "child_min_figure_score": self.child_min_figure_score,
            "child_min_photo_score": self.child_min_photo_score,
        }

    def _child_text_diagnostics(
        self,
        roi: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        region = ImageRegion(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            region_type="diagram",
        )
        return self.text_filter.candidate_diagnostics(roi, region)

    def _child_rejection_reason(self, diagnostic: dict[str, Any]) -> str | None:
        scores = diagnostic.get("scores", {})
        features = diagnostic.get("features", {})
        text_like = float(scores.get("text_like_score") or 0.0)
        figure_like = float(scores.get("figure_like_score") or 0.0)
        photo_like = float(scores.get("photo_like_score") or 0.0)
        regular_projection = bool(features.get("regular_text_projection"))
        large_count = float(features.get("large_dark_component_count") or 0.0)
        large_ratio = float(features.get("max_dark_component_area_ratio") or 0.0)
        if (
            text_like >= self.child_text_reject_threshold
            and regular_projection
            and photo_like < self.child_min_photo_score
            and large_ratio < 0.08
        ):
            return "child_text_like"
        has_visual_mass = (
            (large_count > 0 and not regular_projection)
            or large_ratio >= self.child_min_large_component_ratio
            or (figure_like >= self.child_min_figure_score and not regular_projection)
            or photo_like >= self.child_min_photo_score
        )
        if text_like >= self.child_text_reject_threshold and regular_projection and not has_visual_mass:
            return "child_text_like"
        if diagnostic.get("rejection_reason") and not has_visual_mass:
            return "child_text_like"
        if not has_visual_mass:
            return "child_visual_mass_low"
        return None


def _union_xywh(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def _region_from_record(raw: Any) -> ImageRegion | None:
    if not isinstance(raw, dict):
        return None
    if all(key in raw for key in ("x", "y", "width", "height")):
        return ImageRegion(
            x=int(raw["x"]),
            y=int(raw["y"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            region_type=raw.get("region_type"),
            confidence=raw.get("confidence") or raw.get("score"),
            metadata=raw.get("metadata") or {},
        )
    bbox = raw.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return ImageRegion(
            x=x1,
            y=y1,
            width=max(1, x2 - x1),
            height=max(1, y2 - y1),
            region_type=raw.get("region_type"),
            confidence=raw.get("confidence") or raw.get("score"),
            metadata=raw.get("metadata") or {},
        )
    return None


def _looks_like_caption_strip(
    width: int,
    height: int,
    *,
    edge_density: float,
    dark_fraction: float,
) -> bool:
    aspect = width / float(height or 1)
    return (
        height < 70
        and aspect >= 2.5
        and dark_fraction < 0.22
        and edge_density < 0.18
    )


def _compact_child_features(features: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "regular_text_projection",
        "component_count",
        "median_component_area",
        "small_component_fraction",
        "row_occupancy",
        "col_occupancy",
        "edge_density",
        "hough_line_count",
        "max_dark_component_area_ratio",
        "large_dark_component_count",
        "horizontal_peak_count",
        "horizontal_peak_spacing_std",
        "vertical_peak_count",
        "vertical_peak_spacing_std",
    )
    compact: dict[str, Any] = {}
    for key in keys:
        value = features.get(key)
        if isinstance(value, float):
            compact[key] = round(value, 4)
        else:
            compact[key] = value
    return compact


def _duplicates_existing(
    bbox: tuple[int, int, int, int],
    candidates: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.65,
) -> bool:
    x, y, width, height = bbox
    area = max(1, width * height)
    x2 = x + width
    y2 = y + height
    for candidate in candidates:
        cx, cy, cw, ch = candidate["bbox"]
        cx2 = cx + cw
        cy2 = cy + ch
        inter_w = max(0, min(x2, cx2) - max(x, cx))
        inter_h = max(0, min(y2, cy2) - max(y, cy))
        inter = inter_w * inter_h
        other_area = max(1, cw * ch)
        iou = inter / float(area + other_area - inter)
        if iou >= iou_threshold:
            return True
    return False
