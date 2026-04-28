from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.filters.text_filter import TextRegionFilter
from utils.image.regions.core_types import ImageRegion


def _analyzer(**overrides) -> LayoutAnalyzer:
    cfg = {
        "enabled_detectors": ["contours"],
        "contours_always": True,
        "filter_text_like": True,
        "filter_text_exempt_types": ["figure"],
        "contour_min_area": 1200,
        "contour_max_area_ratio": 0.6,
        "contour_left_bias_xmax": 1.0,
        "contour_topk": 12,
        "diagram_merge_gap": 18,
        "region_text_like_min_components": 12,
    }
    cfg.update(overrides)
    return LayoutAnalyzer(cfg)


def _body_text_block(width: int = 520, height: int = 180) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    lines = [
        "DRAEGER LECTURE NOTES",
        "These typewritten lines should",
        "not become diagram crops.",
        "They are dense repeated glyphs.",
    ]
    for i, text in enumerate(lines):
        cv2.putText(image, text, (18, 38 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, 0, 2, cv2.LINE_AA)
    return image


def _large_illustration_page() -> np.ndarray:
    page = np.full((760, 980), 255, dtype=np.uint8)
    text = _body_text_block(760, 170)
    page[45 : 45 + text.shape[0], 80 : 80 + text.shape[1]] = text

    # Large line-art/photo-like figure that would be missed if max area is too low.
    cv2.rectangle(page, (180, 285), (820, 700), 0, 4)
    cv2.ellipse(page, (500, 485), (230, 155), 8, 0, 360, 0, 5)
    for offset in range(-160, 181, 80):
        cv2.line(page, (310 + offset, 660), (470 + offset, 330), 0, 3)
    cv2.circle(page, (610, 440), 70, 0, 4)
    return page


def _sparse_symbol_page() -> np.ndarray:
    page = np.full((620, 900), 255, dtype=np.uint8)
    page[35:215, 40:560] = _body_text_block(520, 180)

    # Sparse arrows/symbols with enough extent to be semantically figure-like.
    for dx in (0, 190, 380):
        start = (150 + dx, 430)
        end = (300 + dx, 315)
        cv2.arrowedLine(page, start, end, 0, 4, tipLength=0.18)
        cv2.circle(page, (210 + dx, 472), 34, 0, 3)
        cv2.line(page, (180 + dx, 505), (258 + dx, 535), 0, 3)

    # Tiny noise should not become a region.
    cv2.circle(page, (830, 570), 2, 0, -1)
    return page


def _labeled_diagram() -> np.ndarray:
    image = np.full((320, 360), 255, dtype=np.uint8)
    cv2.rectangle(image, (34, 42), (315, 270), 0, 3)
    cv2.circle(image, (175, 150), 62, 0, 4)
    cv2.arrowedLine(image, (70, 245), (143, 181), 0, 3, tipLength=0.18)
    cv2.arrowedLine(image, (290, 74), (218, 124), 0, 3, tipLength=0.18)
    cv2.putText(image, "A", (55, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.82, 0, 2, cv2.LINE_AA)
    cv2.putText(image, "label", (230, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 2, cv2.LINE_AA)
    return image


def test_large_illustration_region_is_retained_while_text_block_is_rejected():
    diagnostics = _analyzer().detect_image_regions_with_diagnostics(_large_illustration_page())

    accepted = diagnostics["accepted_regions"]
    assert any(region.y > 240 and region.area > 180_000 for region in accepted)
    assert not any(region.y < 230 and region.area > 70_000 for region in accepted)


def test_sparse_symbol_and_arrow_cluster_is_retained_without_noise_regions():
    diagnostics = _analyzer(contour_min_area=900).detect_image_regions_with_diagnostics(_sparse_symbol_page())

    accepted = diagnostics["accepted_regions"]
    assert any(region.y > 270 and region.area > 20_000 for region in accepted)
    assert all(region.area > 900 for region in accepted)
    assert not any(region.y < 230 and region.area > 70_000 for region in accepted)


def test_labeled_diagram_is_not_rejected_as_plain_text_components():
    diagram = _labeled_diagram()
    text = _body_text_block(360, 180)
    text_filter = TextRegionFilter(
        {
            "filter_text_exempt_types": ["figure"],
            "region_text_like_min_components": 12,
        }
    )

    diagram_region = ImageRegion(x=0, y=0, width=diagram.shape[1], height=diagram.shape[0], region_type="diagram")
    text_region = ImageRegion(x=0, y=0, width=text.shape[1], height=text.shape[0], region_type="diagram")

    assert text_filter.rejection_reason(diagram, diagram_region) is None
    assert text_filter.rejection_reason(text, text_region) in {
        "text_like_components",
        "title_text_like",
        "regular_text_projection",
    }


def test_labeled_diagram_page_keeps_diagram_and_rejects_adjacent_body_text():
    page = np.full((560, 860), 255, dtype=np.uint8)
    text = _body_text_block(360, 180)
    diagram = _labeled_diagram()
    page[80 : 80 + text.shape[0], 45 : 45 + text.shape[1]] = text
    page[150 : 150 + diagram.shape[0], 465 : 465 + diagram.shape[1]] = diagram

    diagnostics = _analyzer(contour_min_area=1500).detect_image_regions_with_diagnostics(page)
    accepted = diagnostics["accepted_regions"]

    assert any(region.x > 420 and region.y > 120 for region in accepted)
    assert not any(region.x < 420 and region.y < 280 and region.area > 30_000 for region in accepted)
    assert diagnostics["rejected"]


def test_sparse_wide_text_band_is_rejected_without_blocking_sparse_symbols():
    band = np.full((130, 500), 255, dtype=np.uint8)
    for i, text in enumerate(["thin annotation text", "more sparse glyphs"]):
        cv2.putText(band, text, (18, 46 + i * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.78, 0, 2, cv2.LINE_AA)

    symbols = np.full((360, 500), 255, dtype=np.uint8)
    cv2.arrowedLine(symbols, (70, 250), (250, 90), 0, 4, tipLength=0.16)
    cv2.circle(symbols, (320, 170), 58, 0, 3)
    cv2.line(symbols, (285, 245), (395, 300), 0, 3)

    text_filter = TextRegionFilter({"filter_text_exempt_types": ["figure"]})
    band_region = ImageRegion(x=0, y=0, width=band.shape[1], height=band.shape[0], region_type="diagram")
    symbol_region = ImageRegion(x=0, y=0, width=symbols.shape[1], height=symbols.shape[0], region_type="diagram")

    assert text_filter.rejection_reason(band, band_region) in {
        "sparse_text_band",
        "text_line_like",
        "regular_text_projection",
    }
    assert text_filter.rejection_reason(symbols, symbol_region) is None
