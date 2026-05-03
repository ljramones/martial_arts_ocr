from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.detectors.multi_figure_rows import MultiFigureRowDetector


def _panel(width: int = 150, height: int = 150) -> np.ndarray:
    panel = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(panel, (8, 8), (width - 9, height - 9), 0, 4)
    cv2.circle(panel, (width // 2, height // 2), min(width, height) // 5, 0, 4)
    cv2.line(panel, (25, height - 25), (width - 25, 28), 0, 4)
    cv2.arrowedLine(panel, (width - 35, height - 35), (width // 2, height // 2), 0, 3, tipLength=0.16)
    return panel


def _three_panel_row() -> np.ndarray:
    page = np.full((360, 760), 255, dtype=np.uint8)
    for x in (55, 305, 555):
        panel = _panel(140, 150)
        page[95 : 95 + panel.shape[0], x : x + panel.shape[1]] = panel
    return page


def _caption_strips() -> np.ndarray:
    page = np.full((260, 760), 255, dtype=np.uint8)
    for x, text in ((50, "caption one"), (300, "caption two"), (550, "caption three")):
        cv2.putText(page, text, (x, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return page


def _paragraph_text() -> np.ndarray:
    page = np.full((360, 760), 255, dtype=np.uint8)
    lines = [
        "These lines are dense paragraph text.",
        "They should not be treated as figures.",
        "Repeated glyph rows are not panels.",
        "The row detector should stay quiet.",
    ]
    for index, text in enumerate(lines):
        cv2.putText(page, text, (55, 90 + index * 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2, cv2.LINE_AA)
    return page


def _broad_text_parent_with_child_figures() -> np.ndarray:
    page = np.full((500, 800), 255, dtype=np.uint8)
    lines = [
        "The first kata sequence includes visual reference.",
        "More paragraph text surrounds the diagrams.",
        "The broad parent should be text-like.",
    ]
    for index, text in enumerate(lines):
        cv2.putText(page, text, (60, 70 + index * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2, cv2.LINE_AA)
    for x in (100, 520):
        panel = _panel(120, 130)
        page[230 : 230 + panel.shape[0], x : x + panel.shape[1]] = panel
    return page


def _detector(**overrides) -> MultiFigureRowDetector:
    cfg = {
        "multi_figure_row_min_area": 5_000,
        "multi_figure_row_min_width": 70,
        "multi_figure_row_min_height": 90,
        "multi_figure_row_topk": 8,
    }
    cfg.update(overrides)
    return MultiFigureRowDetector(cfg)


def test_three_separated_figure_panels_produce_three_candidates():
    detector = _detector()

    regions = detector.detect(_three_panel_row())

    assert len(regions) == 3
    assert [region.metadata["detector"] for region in regions] == ["multi_figure_rows"] * 3
    assert [region.metadata["sibling_index"] for region in regions] == [1, 2, 3]
    assert all(region.metadata["band_bbox"] for region in regions)
    assert detector.last_diagnostics["returned_count"] == 3


def test_caption_text_strips_are_not_emitted_as_figure_candidates():
    regions = _detector().detect(_caption_strips())

    assert regions == []


def test_dense_paragraph_text_is_not_emitted_as_figure_candidates():
    regions = _detector().detect(_paragraph_text())

    assert regions == []


def test_zero_result_diagnostics_include_rejection_reasons():
    detector = _detector()

    regions = detector.detect(_paragraph_text())

    diagnostics = detector.last_diagnostics
    assert regions == []
    assert diagnostics["page_size"] == [760, 360]
    assert "thresholds" in diagnostics
    assert "rejected" in diagnostics
    assert "bands_considered" in diagnostics
    assert diagnostics["returned_count"] == 0


def test_broad_rejected_parent_proposes_child_visuals_for_review():
    detector = _detector(
        broad_rejected_child_min_area=3_000,
        broad_rejected_child_min_width=55,
        broad_rejected_child_min_height=55,
    )
    rejected = [
        {
            "region": {
                "x": 45,
                "y": 40,
                "width": 700,
                "height": 380,
                "region_type": "diagram",
                "metadata": {},
            },
            "rejection_reason": "text_like_components",
        }
    ]

    regions = detector.propose_child_visuals(_broad_text_parent_with_child_figures(), rejected)

    assert len(regions) == 2
    assert all(region.metadata["detector"] == "broad_rejected_child_visuals" for region in regions)
    assert all(region.metadata["needs_review"] is True for region in regions)
    assert all(region.metadata["parent_rejection_reason"] == "text_like_components" for region in regions)
    assert detector.last_child_diagnostics["returned_count"] == 2


def test_caption_strips_are_not_proposed_as_child_visuals():
    detector = _detector(broad_rejected_child_min_area=1_000)
    rejected = [
        {
            "region": {
                "x": 35,
                "y": 90,
                "width": 690,
                "height": 90,
                "region_type": "diagram",
                "metadata": {},
            },
            "rejection_reason": "text_like_components",
        }
    ]

    regions = detector.propose_child_visuals(_caption_strips(), rejected)

    assert regions == []


def test_dense_paragraph_text_does_not_produce_child_visuals():
    detector = _detector(broad_rejected_child_min_area=1_000)
    rejected = [
        {
            "region": {
                "x": 35,
                "y": 45,
                "width": 690,
                "height": 240,
                "region_type": "diagram",
                "metadata": {},
            },
            "rejection_reason": "regular_text_projection",
        }
    ]

    regions = detector.propose_child_visuals(_paragraph_text(), rejected)

    assert regions == []


def test_layout_analyzer_imports_multi_figure_rows_without_duplicate_central_panel():
    analyzer = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours", "multi_figure_rows"],
            "contours_always": True,
            "filter_text_like": True,
            "contour_min_area": 5_000,
            "contour_max_area_ratio": 0.5,
            "contour_left_bias_xmax": 1.0,
            "contour_topk": 8,
            "multi_figure_row_min_area": 5_000,
            "multi_figure_row_min_width": 70,
            "multi_figure_row_min_height": 90,
        }
    )

    diagnostics = analyzer.detect_image_regions_with_diagnostics(_three_panel_row())
    accepted = diagnostics["accepted_regions"]
    row_diagnostics = [
        detector for detector in diagnostics["detector_diagnostics"]
        if detector["detector"] == "multi_figure_rows"
    ]

    assert row_diagnostics
    assert row_diagnostics[0]["returned_count"] == 3
    assert len(accepted) == 3
    assert len({(region.x, region.y, region.width, region.height) for region in accepted}) == 3


def test_layout_analyzer_imports_child_visuals_from_rejected_parent():
    analyzer = LayoutAnalyzer(
        {
            "enabled_detectors": [],
            "filter_text_like": True,
            "enable_broad_rejected_child_visuals": True,
            "broad_rejected_child_min_area": 3_000,
            "broad_rejected_child_min_width": 55,
            "broad_rejected_child_min_height": 55,
        }
    )
    parent = {
        "region": {
            "x": 45,
            "y": 40,
            "width": 700,
            "height": 380,
            "region_type": "diagram",
            "metadata": {},
        },
        "rejection_reason": "text_like_components",
    }

    child_regions = analyzer.multi_figure_rows.propose_child_visuals(
        _broad_text_parent_with_child_figures(),
        [parent],
    )

    assert len(child_regions) == 2
    assert all(region.metadata["needs_review"] for region in child_regions)
