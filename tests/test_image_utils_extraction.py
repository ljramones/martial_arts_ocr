from __future__ import annotations

import cv2
import numpy as np

from utils.image.layout.analyzer import LayoutAnalyzer
from utils.image.ops.extract import save_region_crops
from utils.image.regions.core_types import ImageRegion


def _synthetic_page() -> np.ndarray:
    image = np.full((420, 520), 255, dtype=np.uint8)
    for y in range(30, 130, 18):
        cv2.line(image, (30, y), (250, y), 0, 2)
    cv2.rectangle(image, (300, 80), (470, 220), 0, 3)
    cv2.line(image, (310, 210), (460, 90), 0, 2)
    cv2.circle(image, (60, 360), 2, 0, -1)
    return image


def test_image_region_supports_legacy_xy_constructor_and_to_dict():
    region = ImageRegion(x=10, y=20, width=30, height=40, region_type="diagram", confidence=0.75)

    assert region.bbox == (10, 20, 40, 60)
    assert region.x == 10
    assert region.y == 20
    assert region.confidence == 0.75
    assert region.to_dict()["width"] == 30


def test_layout_analyzer_detects_diagram_and_ignores_tiny_noise():
    analyzer = LayoutAnalyzer(
        {
            "enabled_detectors": ["contours"],
            "contours_always": True,
            "filter_text_like": False,
            "contour_min_area": 5000,
            "contour_max_area_ratio": 0.5,
            "contour_left_bias_xmax": 1.0,
            "contour_topk": 5,
        }
    )

    regions = analyzer.detect_image_regions(_synthetic_page())

    assert len(regions) == 1
    region = regions[0]
    assert region.region_type == "diagram"
    assert 285 <= region.x <= 305
    assert 65 <= region.y <= 85
    assert 170 <= region.width <= 200
    assert 140 <= region.height <= 170


def test_region_crops_are_saved_with_metadata(tmp_path):
    image = _synthetic_page()
    region = ImageRegion(x=300, y=80, width=170, height=140, region_type="diagram", confidence=0.9)

    saved = save_region_crops(image, [region], tmp_path, prefix="diagram")

    assert len(saved) == 1
    crop_path = tmp_path / "diagram_001.png"
    assert crop_path.exists()
    assert saved[0]["image_path"] == str(crop_path)
    assert saved[0]["region"]["region_type"] == "diagram"
    assert saved[0]["width"] == 170
    assert saved[0]["height"] == 140
