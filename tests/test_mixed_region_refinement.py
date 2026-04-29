from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.layout.options import RegionDetectionOptions
from utils.image.layout.refinement import refine_mixed_candidate
from utils.image.regions.core_types import ImageRegion
from utils.text.geometry import OCRTextBox


def _region_for(image: np.ndarray) -> ImageRegion:
    return ImageRegion(x=0, y=0, width=image.shape[1], height=image.shape[0], region_type="diagram")


def _broad_figure_text_page() -> tuple[np.ndarray, list[OCRTextBox]]:
    image = np.full((260, 520), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 45), (190, 205), 0, 3)
    cv2.line(image, (50, 190), (180, 60), 0, 3)
    boxes: list[OCRTextBox] = []
    for index, y in enumerate(range(50, 220, 22), start=1):
        cv2.line(image, (260, y), (490, y), 0, 2)
        boxes.append(OCRTextBox(text=f"body {index}", bbox=(250, y - 8, 250, 16), confidence=0.94, level="line"))
    return image, boxes


def _embedded_mixed_page() -> tuple[np.ndarray, list[OCRTextBox]]:
    image = np.full((260, 520), 255, dtype=np.uint8)
    cv2.rectangle(image, (180, 75), (330, 190), 0, 3)
    cv2.line(image, (195, 175), (315, 90), 0, 2)
    boxes: list[OCRTextBox] = []
    for index, y in enumerate(range(40, 230, 24), start=1):
        cv2.line(image, (30, y), (165, y), 0, 2)
        cv2.line(image, (350, y), (500, y), 0, 2)
        boxes.append(OCRTextBox(text=f"left {index}", bbox=(25, y - 8, 150, 16), confidence=0.9, level="line"))
        boxes.append(OCRTextBox(text=f"right {index}", bbox=(345, y - 8, 160, 16), confidence=0.9, level="line"))
    return image, boxes


def _labeled_diagram_page() -> tuple[np.ndarray, list[OCRTextBox]]:
    image = np.full((320, 420), 255, dtype=np.uint8)
    cv2.rectangle(image, (120, 95), (180, 145), 0, 3)
    cv2.rectangle(image, (240, 95), (300, 145), 0, 3)
    cv2.rectangle(image, (180, 205), (240, 255), 0, 3)
    cv2.line(image, (180, 120), (240, 120), 0, 3)
    cv2.line(image, (210, 145), (210, 205), 0, 3)
    boxes = [
        OCRTextBox("A", (112, 76, 25, 20), 0.9),
        OCRTextBox("B", (292, 78, 25, 20), 0.9),
        OCRTextBox("C", (170, 257, 25, 20), 0.9),
        OCRTextBox("D", (245, 257, 25, 20), 0.9),
        OCRTextBox("E", (190, 103, 25, 20), 0.9),
        OCRTextBox("F", (218, 182, 25, 20), 0.9),
    ]
    for box in boxes:
        x, y, _w, _h = box.bbox
        cv2.putText(image, box.text, (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
    return image, boxes


def _options(**overrides) -> RegionDetectionOptions:
    values = {
        "enable_mixed_region_refinement": True,
        "mixed_region_min_ocr_overlap": 0.03,
    }
    values.update(overrides)
    return RegionDetectionOptions(**values)


def test_broad_parent_crop_refines_to_figure_side():
    image, boxes = _broad_figure_text_page()

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=boxes, options=_options())

    assert result.refinement_applied is True
    assert result.bbox[0] < 80
    assert result.bbox[2] < image.shape[1] * 0.6
    assert result.metadata["refined_bbox"] != result.metadata["original_bbox"]


def test_embedded_mixed_crop_is_downgraded_to_needs_review():
    image, boxes = _embedded_mixed_page()

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=boxes, options=_options())

    assert result.refinement_applied is False
    assert result.needs_review is True
    assert result.bbox == (0, 0, image.shape[1], image.shape[0])


def test_labeled_diagram_preserves_all_labels():
    image, boxes = _labeled_diagram_page()

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=boxes, options=_options())

    assert result.refinement_applied is False
    assert result.region_role == "mixed_labeled"
    x, y, width, height = result.bbox
    for box in boxes:
        bx, by, bw, bh = box.bbox
        assert x <= bx and y <= by
        assert bx + bw <= x + width
        assert by + bh <= y + height


def test_plain_text_still_rejected_before_refinement():
    image = np.full((160, 420), 255, dtype=np.uint8)
    boxes = []
    for index, y in enumerate(range(30, 140, 18), start=1):
        cv2.line(image, (25, y), (390, y), 0, 2)
        boxes.append(OCRTextBox(f"line {index}", (20, y - 8, 380, 16), 0.95, level="line"))

    diagnostic = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": True,
            "region_enable_mixed_region_refinement": True,
        }
    ).text_filter.candidate_diagnostics(image, _region_for(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] is not None
    assert diagnostic["region_role"] == "rejected_text"


def test_clean_photo_low_ocr_overlap_is_unchanged():
    rng = np.random.default_rng(22)
    image = rng.integers(30, 220, size=(220, 260), dtype=np.uint8)
    boxes = [OCRTextBox("caption", (4, 205, 70, 10), 0.91)]

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=boxes, options=_options(mixed_region_min_ocr_overlap=0.20))

    assert result.refinement_applied is False
    assert result.needs_review is False
    assert result.bbox == (0, 0, image.shape[1], image.shape[0])


def test_high_ocr_overlap_weak_visual_does_not_emit_refined_bbox():
    image = np.full((180, 360), 255, dtype=np.uint8)
    boxes = []
    for y in range(25, 160, 20):
        cv2.line(image, (20, y), (330, y), 0, 2)
        boxes.append(OCRTextBox("body", (15, y - 8, 325, 16), 0.95, level="line"))

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=boxes, options=_options())

    assert result.refinement_applied is False
    assert result.needs_review is True


def test_refinement_metadata_round_trips_through_image_region():
    image, boxes = _embedded_mixed_page()
    region = _region_for(image)
    result = refine_mixed_candidate(image, region, ocr_text_boxes=boxes, options=_options())
    metadata = {
        **result.metadata,
        "mixed_region": True,
        "needs_review": result.needs_review,
        "refinement_applied": result.refinement_applied,
        "mixed_reason": result.mixed_reason,
    }

    output = ImageRegion(x=0, y=0, width=10, height=10, region_type="diagram", metadata=metadata).to_dict()

    assert output["metadata"]["mixed_region"] is True
    assert output["metadata"]["needs_review"] is True
    assert output["metadata"]["refinement_applied"] is False
    assert "original_bbox" in output["metadata"]
    assert "refined_bbox" in output["metadata"]


def test_no_ocr_boxes_is_noop():
    image, _boxes = _broad_figure_text_page()

    result = refine_mixed_candidate(image, _region_for(image), ocr_text_boxes=[], options=_options())

    assert result.refinement_applied is False
    assert result.needs_review is False
    assert result.bbox == (0, 0, image.shape[1], image.shape[0])


def test_refinement_disabled_keeps_analyzer_behavior_identical():
    image, boxes = _broad_figure_text_page()
    base = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": True,
            "region_enable_mixed_region_refinement": False,
        }
    ).detect_image_regions_with_diagnostics(image, ocr_text_boxes=boxes)
    disabled = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": True,
            "region_enable_mixed_region_refinement": False,
        }
    ).detect_image_regions_with_diagnostics(image, ocr_text_boxes=boxes)

    assert base["accepted"] == disabled["accepted"]
    assert base["refinement"] == []
