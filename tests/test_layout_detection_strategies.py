from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.image.layout.strategies import (
    ClassicalLayoutStrategy,
    DocLayoutYOLOStrategy,
    GenericYOLOLayoutStrategy,
    LayoutParserStrategy,
    PaddleLayoutStrategy,
)
from utils.image.layout.strategy import LayoutDetectionResult


def _synthetic_diagram_page() -> np.ndarray:
    image = np.full((420, 520), 255, dtype=np.uint8)
    for y in range(30, 130, 18):
        cv2.line(image, (30, y), (250, y), 0, 2)
    cv2.rectangle(image, (300, 80), (470, 220), 0, 3)
    cv2.line(image, (310, 210), (460, 90), 0, 2)
    return image


def test_classical_layout_strategy_wraps_current_detector():
    strategy = ClassicalLayoutStrategy(
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

    result = strategy.detect(_synthetic_diagram_page())

    assert isinstance(result, LayoutDetectionResult)
    assert result.strategy_name == "classical_opencv"
    assert result.available is True
    assert len(result.regions) == 1
    assert result.regions[0].region_type == "diagram"
    assert "accepted" in result.metadata
    assert result.to_dict()["regions"][0]["region_type"] == "diagram"


def test_optional_yolo_strategies_skip_without_model_paths(monkeypatch):
    image = _synthetic_diagram_page()
    monkeypatch.setattr(DocLayoutYOLOStrategy, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(GenericYOLOLayoutStrategy, "is_available", classmethod(lambda cls: True))

    doclayout_result = DocLayoutYOLOStrategy().detect(image)
    generic_result = GenericYOLOLayoutStrategy().detect(image)

    assert doclayout_result.available is False
    assert "model_path" in doclayout_result.skipped_reason
    assert generic_result.available is False
    assert "model_path" in generic_result.skipped_reason


def test_optional_layoutparser_strategy_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(LayoutParserStrategy, "is_available", classmethod(lambda cls: False))

    result = LayoutParserStrategy().detect(_synthetic_diagram_page())

    assert result.available is False
    assert result.skipped_reason == "layoutparser is not installed"


def test_optional_paddle_strategy_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(PaddleLayoutStrategy, "is_available", classmethod(lambda cls: False))

    result = PaddleLayoutStrategy().detect(_synthetic_diagram_page())

    assert result.available is False
    assert result.skipped_reason == "paddleocr is not installed"


def test_layoutparser_strategy_maps_fake_model_output(monkeypatch):
    @dataclass
    class FakeBox:
        coordinates: tuple[int, int, int, int]

    @dataclass
    class FakeBlock:
        block: FakeBox
        type: str
        score: float

    class FakeModel:
        def detect(self, image):
            return [FakeBlock(FakeBox((10, 20, 110, 120)), "Figure", 0.92)]

    monkeypatch.setattr(LayoutParserStrategy, "is_available", classmethod(lambda cls: True))

    result = LayoutParserStrategy({"model": FakeModel()}).detect(_synthetic_diagram_page())

    assert result.available is True
    assert len(result.regions) == 1
    assert result.regions[0].region_type == "figure"
    assert result.regions[0].metadata["label"] == "figure"
