from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import cv2
import numpy as np

from utils.image.regions.core_image import ImageRegion
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
            raw = self._component_candidates(gray, mask)
            banded = self._keep_row_siblings(raw)
            regions = [
                self._region(candidate, page_shape=(h, w), sibling_index=index)
                for index, candidate in enumerate(banded[: self.max_candidates], start=1)
            ]
            self.last_diagnostics = {
                "detector": "multi_figure_rows",
                "raw_count": len(raw),
                "returned_count": len(regions),
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
                "candidates": [],
            }
            return []

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

    def _component_candidates(self, gray: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
        h_img, w_img = gray.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, Any]] = []
        max_area = h_img * w_img * self.max_area_ratio
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < self.min_area or area > max_area:
                continue
            if width < self.min_width or height < self.min_height:
                continue
            aspect = width / float(height or 1)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            roi = gray[y : y + height, x : x + width]
            roi_edges = cv2.Canny(roi, 50, 150)
            edge_density = float(np.sum(roi_edges > 0)) / float(max(1, area))
            dark_fraction = float(np.sum(roi < 190)) / float(max(1, area))
            if not (self.min_edge_density <= edge_density <= self.max_edge_density):
                continue
            if not (self.min_dark_fraction <= dark_fraction <= self.max_dark_fraction):
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
        return sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0]))

    def _keep_row_siblings(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        for row in rows:
            if len(row) < self.min_band_members:
                continue
            row = sorted(row, key=lambda item: item["bbox"][0])
            band_bbox = _union_xywh([item["bbox"] for item in row])
            visual_band_score = round(float(sum(item["area"] for item in row) / max(1, band_bbox[2] * band_bbox[3])), 4)
            for index, item in enumerate(row, start=1):
                item = dict(item)
                item["band_bbox"] = band_bbox
                item["sibling_index"] = index
                item["visual_band_score"] = visual_band_score
                kept.append(item)
        return kept

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


def _union_xywh(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
