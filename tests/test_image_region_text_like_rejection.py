from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.filters.text_filter import TextRegionFilter
from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.regions.core_types import ImageRegion


def _filter() -> TextRegionFilter:
    return TextRegionFilter(
        {
            "filter_text_exempt_types": ["figure"],
            "region_reject_text_like": True,
            "region_reject_rotated_text_like": True,
            "region_text_like_min_components": 12,
        }
    )


def _horizontal_text_fragment() -> np.ndarray:
    image = np.full((220, 520), 255, dtype=np.uint8)
    for i, text in enumerate(["MARTIAL ARTS & WAYS.", "HANDSCROLLS.", "DRAEGER NOTES"]):
        cv2.putText(image, text, (24, 52 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, 0, 3, cv2.LINE_AA)
    return image


def _vertical_text_fragment() -> np.ndarray:
    horizontal = np.full((160, 420), 255, dtype=np.uint8)
    for i, text in enumerate(["the same typewritten", "fragments should not", "be treated as images"]):
        cv2.putText(horizontal, text, (18, 42 + i * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return cv2.rotate(horizontal, cv2.ROTATE_90_CLOCKWISE)


def _diagram_like_shape() -> np.ndarray:
    image = np.full((260, 360), 255, dtype=np.uint8)
    cv2.rectangle(image, (38, 52), (320, 215), 0, 3)
    cv2.line(image, (55, 200), (305, 70), 0, 3)
    cv2.circle(image, (220, 136), 40, 0, 3)
    cv2.line(image, (75, 70), (150, 185), 0, 2)
    return image


def test_horizontal_typewritten_fragment_is_rejected():
    gray = _horizontal_text_fragment()
    region = ImageRegion(x=0, y=0, width=gray.shape[1], height=gray.shape[0], region_type="diagram")

    reason = _filter().rejection_reason(gray, region)

    assert reason in {"text_like_components", "title_text_like", "text_line_like", "regular_text_projection"}


def test_vertical_rotated_text_fragment_is_rejected():
    gray = _vertical_text_fragment()
    region = ImageRegion(x=0, y=0, width=gray.shape[1], height=gray.shape[0], region_type="diagram")

    reason = _filter().rejection_reason(gray, region)

    assert reason in {"rotated_text_like", "text_like_components", "regular_text_projection"}


def test_diagram_like_shape_is_retained():
    gray = _diagram_like_shape()
    region = ImageRegion(x=0, y=0, width=gray.shape[1], height=gray.shape[0], region_type="diagram")

    assert _filter().rejection_reason(gray, region) is None


def test_layout_analyzer_rejects_text_but_retains_diagram_and_ignores_noise():
    page = np.full((460, 780), 255, dtype=np.uint8)
    text = _horizontal_text_fragment()
    diagram = _diagram_like_shape()
    page[35 : 35 + text.shape[0], 30 : 30 + text.shape[1]] = text
    page[160 : 160 + diagram.shape[0], 390 : 390 + diagram.shape[1]] = diagram
    cv2.circle(page, (680, 420), 2, 0, -1)

    analyzer = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": True,
            "filter_text_exempt_types": ["figure"],
            "contour_min_area": 3000,
            "contour_max_area_ratio": 0.6,
            "contour_left_bias_xmax": 1.0,
            "contour_topk": 10,
            "region_text_like_min_components": 12,
        }
    )

    diagnostics = analyzer.detect_image_regions_with_diagnostics(page)
    regions = diagnostics["accepted_regions"]

    assert len(regions) == 1
    assert regions[0].region_type == "diagram"
    assert regions[0].x >= 360
    assert diagnostics["rejected"]
