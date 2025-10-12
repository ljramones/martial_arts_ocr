from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from scipy.signal import find_peaks
from utils.image.regions.core_image import ImageRegion
from ..post.merge import remove_overlaps

logger = logging.getLogger(__name__)


class TextRegionFilter:
    """
    Provides:
      - detect_mser(gray): MSER-based text region proposal
      - filter(gray, regions): reject text-like candidates among image regions

    New features:
      • Configurable regularity and CC thresholds
      • Exempt region types (e.g., 'diagram') from text filtering
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # Tunable thresholds
        self.peak_distance = int(cfg.get('text_peak_distance', 8))       # default 8
        self.proj_std_max = float(cfg.get('text_proj_std_max', 8.0))     # default 8.0
        self.cc_overflow = int(cfg.get('text_cc_overflow_labels', 600))  # default 600
        self.cc_median_max = float(cfg.get('text_cc_median_max', 180.0)) # default 180.0

        # Exempt region types (e.g. diagrams)
        self.exempt_types = set(cfg.get('filter_text_exempt_types', ['diagram']))

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
            # Exempt certain region types (e.g., diagrams)
            if getattr(r, "region_type", "") in self.exempt_types:
                kept.append(r)
                continue

            x, y, w, h = r.x, r.y, r.width, r.height
            x = max(0, x); y = max(0, y)
            w = min(w, gray.shape[1] - x); h = min(h, gray.shape[0] - y)
            if w <= 0 or h <= 0:
                continue

            roi = gray[y:y + h, x:x + w]
            is_text = False

            # --- Horizontal regularity ---
            try:
                hp = np.sum(roi < 128, axis=1)
                if hp.size > 10:
                    ph, _ = find_peaks(hp, distance=self.peak_distance)
                    if ph.size > 10 and np.std(np.diff(ph)) < self.proj_std_max:
                        is_text = True
                        logger.debug("Reject: horizontal regularity (std<%.1f)", self.proj_std_max)
            except Exception:
                pass

            # --- Vertical regularity ---
            if not is_text:
                try:
                    vp = np.sum(roi < 128, axis=0)
                    if vp.size > 10:
                        pv, _ = find_peaks(vp, distance=self.peak_distance)
                        if pv.size > 10 and np.std(np.diff(pv)) < self.proj_std_max:
                            is_text = True
                            logger.debug("Reject: vertical regularity (std<%.1f)", self.proj_std_max)
                except Exception:
                    pass

            # --- Connected components ---
            if not is_text:
                try:
                    _, binr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binr, connectivity=8)
                    if num_labels > self.cc_overflow:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        median_area = float(np.median(areas)) if areas.size else 0.0
                        if median_area < self.cc_median_max:
                            is_text = True
                            logger.debug("Reject: CC overflow small median")
                    elif num_labels > 50:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        if areas.size:
                            median_area = float(np.median(areas))
                            mean_area = float(np.mean(areas))
                            if median_area < 100 and mean_area > 0:
                                if (np.std(areas) / mean_area) < 1.5:
                                    is_text = True
                                    logger.debug("Reject: CC uniform small components")
                except Exception:
                    pass

            # --- Dense but structureless fill ---
            if not is_text:
                try:
                    fill = float(np.sum(roi < 128)) / float(w * h)
                    if 0.4 < fill < 0.7 and (w / float(h)) > 0.8:
                        edges = cv2.Canny(roi, 50, 150)
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30)
                        if lines is None or len(lines) < 5:
                            is_text = True
                            logger.debug("Reject: dense & structureless")
                except Exception:
                    pass

            if not is_text:
                kept.append(r)

        return kept
