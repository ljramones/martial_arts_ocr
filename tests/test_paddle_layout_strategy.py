from __future__ import annotations

import importlib

import numpy as np

from utils.image.layout.strategies.paddle_layout import PaddleLayoutStrategy


def test_paddle_strategy_imports_are_lazy(monkeypatch):
    original_import = importlib.import_module
    imported: list[str] = []

    def tracking_import(name, package=None):
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    PaddleLayoutStrategy()

    assert "paddleocr" not in imported


def test_paddle_strategy_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(PaddleLayoutStrategy, "is_available", classmethod(lambda cls: False))

    result = PaddleLayoutStrategy().detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert result.available is False
    assert result.skipped_reason == "paddleocr is not installed"


def test_paddle_strategy_maps_fake_ppstructure_output(monkeypatch):
    class FakeEngine:
        def predict(self, image):
            return [
                {
                    "layout_det_res": {
                        "boxes": [
                            {
                                "label": "image",
                                "score": 0.97,
                                "coordinate": [10, 20, 110, 120],
                            },
                            {
                                "label": "table",
                                "score": 0.8,
                                "coordinate": [120, 20, 220, 120],
                            },
                            {
                                "label": "figure_title",
                                "score": 0.9,
                                "coordinate": [10, 130, 110, 145],
                            },
                        ]
                    }
                }
            ]

    monkeypatch.setattr(PaddleLayoutStrategy, "is_available", classmethod(lambda cls: True))

    result = PaddleLayoutStrategy({"engine": FakeEngine()}).detect(np.zeros((240, 260, 3), dtype=np.uint8))

    assert result.available is True
    assert len(result.regions) == 3
    assert result.regions[0].bbox == (10, 20, 110, 120)
    assert result.regions[0].region_type == "figure"
    assert result.regions[0].metadata["layout_label"] == "image"
    assert result.regions[1].region_type == "table"
    assert result.regions[2].region_type == "text"
    assert result.regions[2].metadata["layout_label"] == "title"
