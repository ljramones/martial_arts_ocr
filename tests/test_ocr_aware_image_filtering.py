from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.regions.core_types import ImageRegion
from utils.text.geometry import (
    OCRTextBox,
    build_text_mask,
    compute_bbox_overlap_ratio,
    compute_mean_text_confidence,
    compute_text_box_count,
    compute_text_box_coverage,
    compute_text_mask_overlap,
)


def _analyzer(**overrides) -> LayoutAnalyzer:
    cfg = {
        "enabled_detectors": ["contours"],
        "contours_always": True,
        "filter_text_like": True,
        "contour_min_area": 200,
        "contour_max_area_ratio": 1.0,
        "contour_left_bias_xmax": 1.0,
        "contour_topk": 10,
    }
    cfg.update(overrides)
    return LayoutAnalyzer(cfg)


def _paragraph(width: int = 420, height: int = 220) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    for y in range(25, height - 20, 18):
        cv2.line(image, (20, y), (width - 25, y), 0, 2)
    return image


def _vertical_text(width: int = 140, height: int = 360) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    for y in range(18, height - 18, 18):
        cv2.rectangle(image, (52, y), (88, y + 8), 0, -1)
    return image


def _diagram(width: int = 320, height: int = 240, *, labeled: bool = False) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (width - 35, height - 35), 0, 3)
    cv2.line(image, (50, height - 55), (width - 55, 55), 0, 3)
    cv2.arrowedLine(image, (60, 70), (width - 70, height - 70), 0, 3)
    if labeled:
        cv2.putText(image, "A", (72, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.putText(image, "B", (width - 105, height - 86), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return image


def _photo_like(width: int = 260, height: int = 220) -> np.ndarray:
    rng = np.random.default_rng(10)
    base = rng.integers(35, 220, size=(height, width), dtype=np.uint8)
    cv2.GaussianBlur(base, (5, 5), 0, dst=base)
    cv2.rectangle(base, (15, 15), (width - 15, height - 15), 20, 2)
    return base


def _full_region(image: np.ndarray, region_type: str = "diagram") -> ImageRegion:
    return ImageRegion(x=0, y=0, width=image.shape[1], height=image.shape[0], region_type=region_type)


def _covering_text_boxes(image: np.ndarray) -> list[OCRTextBox]:
    boxes = []
    for index, y in enumerate(range(18, image.shape[0] - 20, 18), start=1):
        boxes.append(OCRTextBox(text=f"line {index}", bbox=(12, y - 8, image.shape[1] - 24, 14), confidence=0.95, level="line"))
    return boxes


def test_ocr_geometry_overlap_helpers():
    candidate = (10, 10, 100, 100)
    boxes = [
        OCRTextBox("inside", (20, 20, 40, 30), confidence=0.9),
        OCRTextBox("outside", (200, 200, 20, 20), confidence=0.2),
    ]

    assert compute_text_box_count(candidate, boxes) == 1
    assert compute_bbox_overlap_ratio(candidate, boxes) == 0.12
    assert compute_text_box_coverage(candidate, boxes) == 0.75
    assert compute_mean_text_confidence(candidate, boxes) == 0.9

    mask = build_text_mask((140, 140), boxes, dilation_px=0)
    assert compute_text_mask_overlap(candidate, mask) == 0.12


def test_plain_paragraph_high_ocr_overlap_is_rejected_as_text():
    image = _paragraph()
    diagnostic = _analyzer().text_filter.candidate_diagnostics(
        image,
        _full_region(image),
        ocr_text_boxes=_covering_text_boxes(image),
    )

    assert diagnostic["rejection_reason"] in {"high_ocr_text_overlap", "title_text_like", "regular_text_projection"}
    assert diagnostic["region_role"] == "rejected_text"
    assert diagnostic["features"]["ocr_text_overlap_ratio"] >= 0.5


def test_title_like_high_ocr_overlap_is_rejected_as_text():
    image = np.full((90, 420), 255, dtype=np.uint8)
    cv2.putText(image, "DRAEGER NOTES", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.45, 0, 3)
    boxes = [OCRTextBox("DRAEGER NOTES", (0, 0, image.shape[1], image.shape[0]), confidence=0.96, level="line")]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] in {"high_ocr_text_overlap", "title_text_like", "regular_text_projection"}
    assert diagnostic["region_role"] == "rejected_text"


def test_vertical_text_high_ocr_overlap_is_rejected():
    image = _vertical_text()
    boxes = [
        OCRTextBox("vertical", (45, 10, 55, image.shape[0] - 20), confidence=0.9, level="block"),
    ]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] in {"high_ocr_text_overlap", "rotated_text_like", "bad_aspect_ratio", "regular_text_projection"}
    assert diagnostic["region_role"] in {"rejected_text", "rejected_noise"}


def test_large_figure_low_ocr_overlap_is_accepted():
    image = _diagram()
    boxes = [OCRTextBox("caption", (10, image.shape[0] - 16, 80, 12), confidence=0.9)]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] is None
    assert diagnostic["accepted_reason"] in {"low_ocr_overlap_visual_region", "figure_like", "sparse_symbol"}
    assert diagnostic["features"]["ocr_text_overlap_ratio"] <= 0.1


def test_photo_like_region_low_ocr_overlap_is_accepted():
    image = _photo_like()
    boxes = [OCRTextBox("caption", (5, image.shape[0] - 12, 60, 10), confidence=0.9)]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image, "photo"), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] is None
    assert diagnostic["accepted_reason"] is not None


def test_labeled_diagram_partial_ocr_overlap_is_preserved_as_mixed():
    image = _diagram(labeled=True)
    boxes = [
        OCRTextBox("A", (62, 62, 48, 46), confidence=0.93),
        OCRTextBox("B", (image.shape[1] - 116, image.shape[0] - 116, 50, 46), confidence=0.91),
    ]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] is None
    assert diagnostic["region_role"] in {"image", "mixed", "uncertain"}
    assert diagnostic["features"]["ocr_text_box_count"] == 2


def test_sparse_arrows_with_low_ocr_overlap_remain_accepted():
    image = np.full((220, 320), 255, dtype=np.uint8)
    cv2.arrowedLine(image, (25, 45), (285, 45), 0, 3)
    cv2.arrowedLine(image, (285, 75), (40, 175), 0, 3)
    boxes = [OCRTextBox("small label", (12, 190, 80, 12), confidence=0.88)]

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["rejection_reason"] is None
    assert diagnostic["scores"]["sparse_symbol_score"] >= 0.65


def test_broad_mixed_crop_high_ocr_overlap_weak_visual_is_rejected_or_uncertain():
    image = _paragraph(width=520, height=320)
    cv2.rectangle(image, (410, 220), (500, 292), 0, 2)
    boxes = _covering_text_boxes(image)

    diagnostic = _analyzer().text_filter.candidate_diagnostics(image, _full_region(image), ocr_text_boxes=boxes)

    assert diagnostic["region_role"] in {"rejected_text", "uncertain"}
    if diagnostic["rejection_reason"]:
        assert "ocr" in diagnostic["rejection_reason"] or "text" in diagnostic["rejection_reason"]


def test_no_ocr_boxes_preserves_visual_only_decision():
    image = _diagram()
    region = _full_region(image)
    analyzer = _analyzer()

    without_argument = analyzer.text_filter.candidate_diagnostics(image, region)
    with_empty_boxes = analyzer.text_filter.candidate_diagnostics(image, region, ocr_text_boxes=[])

    assert with_empty_boxes["rejection_reason"] == without_argument["rejection_reason"]
    assert with_empty_boxes["accepted_reason"] == without_argument["accepted_reason"]
    assert with_empty_boxes["features"]["ocr_text_boxes_available"] is False
