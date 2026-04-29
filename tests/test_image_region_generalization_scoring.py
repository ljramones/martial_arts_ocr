from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.filters.text_filter import TextRegionFilter
from utils.image.regions.core_types import ImageRegion


def _analyzer(**overrides) -> LayoutAnalyzer:
    cfg = {
        "enabled_detectors": ["figure", "contours"],
        "contours_always": True,
        "filter_text_like": True,
        "filter_text_exempt_types": [],
        "figure_isolation_white": 0.55,
        "figure_left_bias_xmax": 1.0,
        "figure_min_area": 900,
        "contour_min_area": 900,
        "contour_max_area_ratio": 0.7,
        "contour_left_bias_xmax": 1.0,
        "contour_topk": 16,
    }
    cfg.update(overrides)
    return LayoutAnalyzer(cfg)


def _text_filter() -> TextRegionFilter:
    return TextRegionFilter({"filter_text_exempt_types": []})


def _paragraph_text() -> np.ndarray:
    image = np.full((260, 520), 255, dtype=np.uint8)
    lines = [
        "Donn Draeger wrote about classical arts.",
        "This paragraph should remain text only.",
        "Repeated glyph rows are not diagrams.",
        "The detector should reject this block.",
        "More body text follows in regular lines.",
    ]
    for index, text in enumerate(lines):
        cv2.putText(image, text, (18, 42 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 0, 2, cv2.LINE_AA)
    return image


def _bold_title() -> np.ndarray:
    image = np.full((150, 620), 255, dtype=np.uint8)
    cv2.putText(image, "MARTIAL ARTS RESEARCH NOTES", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 0, 4, cv2.LINE_AA)
    return image


def _large_diagram() -> np.ndarray:
    image = np.full((360, 500), 255, dtype=np.uint8)
    cv2.rectangle(image, (45, 45), (455, 315), 0, 4)
    cv2.circle(image, (250, 180), 76, 0, 4)
    cv2.line(image, (70, 290), (430, 70), 0, 4)
    cv2.arrowedLine(image, (90, 82), (215, 150), 0, 3, tipLength=0.16)
    return image


def _labeled_diagram() -> np.ndarray:
    image = _large_diagram()
    cv2.putText(image, "A", (68, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    cv2.putText(image, "label", (338, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 2, cv2.LINE_AA)
    return image


def _sparse_symbols() -> np.ndarray:
    image = np.full((320, 500), 255, dtype=np.uint8)
    cv2.arrowedLine(image, (70, 245), (235, 92), 0, 4, tipLength=0.18)
    cv2.circle(image, (330, 160), 58, 0, 3)
    cv2.line(image, (280, 242), (402, 298), 0, 3)
    return image


def _photo_like_panel(width: int = 230, height: int = 180) -> np.ndarray:
    rng = np.random.default_rng(42)
    panel = rng.normal(115, 55, (height, width)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), 0, 3)
    cv2.circle(panel, (width // 2, height // 2), min(width, height) // 5, 35, -1)
    cv2.line(panel, (15, height - 28), (width - 15, height - 45), 20, 4)
    return panel


def test_scoring_rejects_plain_bold_and_vertical_text():
    text_filter = _text_filter()

    paragraph = _paragraph_text()
    paragraph_region = ImageRegion(x=0, y=0, width=paragraph.shape[1], height=paragraph.shape[0], region_type="figure")
    assert text_filter.rejection_reason(paragraph, paragraph_region) is not None

    title = _bold_title()
    title_region = ImageRegion(x=0, y=0, width=title.shape[1], height=title.shape[0], region_type="figure")
    assert text_filter.rejection_reason(title, title_region) in {
        "title_text_like",
        "regular_text_projection",
        "scored_text_like",
        "broad_text_like_crop",
    }

    vertical = cv2.rotate(paragraph, cv2.ROTATE_90_CLOCKWISE)
    vertical_region = ImageRegion(x=0, y=0, width=vertical.shape[1], height=vertical.shape[0], region_type="figure")
    assert text_filter.rejection_reason(vertical, vertical_region) is not None


def test_scoring_retains_diagram_labeled_diagram_and_sparse_symbols():
    text_filter = _text_filter()
    for image in (_large_diagram(), _labeled_diagram(), _sparse_symbols()):
        region = ImageRegion(x=0, y=0, width=image.shape[1], height=image.shape[0], region_type="diagram")
        assert text_filter.rejection_reason(image, region) is None


def test_photo_like_panel_and_photo_grid_are_detected():
    page = np.full((620, 760), 255, dtype=np.uint8)
    panel_a = _photo_like_panel()
    panel_b = _photo_like_panel()
    panel_c = _photo_like_panel()
    page[80 : 80 + panel_a.shape[0], 70 : 70 + panel_a.shape[1]] = panel_a
    page[80 : 80 + panel_b.shape[0], 430 : 430 + panel_b.shape[1]] = panel_b
    page[330 : 330 + panel_c.shape[0], 250 : 250 + panel_c.shape[1]] = panel_c

    diagnostics = _analyzer().detect_image_regions_with_diagnostics(page)
    accepted = diagnostics["accepted_regions"]

    assert len(accepted) >= 2
    assert all(region.metadata.get("photo_like_score", 0.0) >= 0.58 for region in accepted[:2])


def test_broad_mixed_text_crop_is_penalized_but_strong_visual_crop_survives():
    text_filter = _text_filter()

    mixed = np.full((420, 620), 255, dtype=np.uint8)
    paragraph = _paragraph_text()
    mixed[30 : 30 + paragraph.shape[0], 30 : 30 + paragraph.shape[1]] = paragraph
    panel = _photo_like_panel(120, 110)
    mixed[275 : 275 + panel.shape[0], 440 : 440 + panel.shape[1]] = panel
    broad_region = ImageRegion(x=0, y=0, width=mixed.shape[1], height=mixed.shape[0], region_type="figure")

    assert text_filter.rejection_reason(mixed, broad_region) in {
        "broad_text_like_crop",
        "scored_text_like",
        "regular_text_projection",
    }

    strong_visual = _photo_like_panel(260, 220)
    visual_region = ImageRegion(x=0, y=0, width=strong_visual.shape[1], height=strong_visual.shape[0], region_type="figure")
    assert text_filter.rejection_reason(strong_visual, visual_region) is None


def test_jpeg_like_noise_does_not_create_regions():
    rng = np.random.default_rng(123)
    page = np.full((520, 720), 247, dtype=np.uint8)
    noise = rng.normal(0, 5, page.shape).astype(np.int16)
    page = np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(20):
        x = int(rng.integers(0, page.shape[1]))
        y = int(rng.integers(0, page.shape[0]))
        cv2.circle(page, (x, y), 1, 90, -1)

    diagnostics = _analyzer().detect_image_regions_with_diagnostics(page)

    assert diagnostics["accepted_regions"] == []
