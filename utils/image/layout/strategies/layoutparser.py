from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from utils.image.layout.strategy import LayoutDetectionResult, skipped_result
from utils.image.regions.core_types import ImageRegion


class LayoutParserStrategy:
    """Optional LayoutParser adapter.

    A model object or model factory must be provided explicitly. The adapter
    does not download Detectron2/LayoutParser weights.
    """

    name = "layoutparser"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("layoutparser") is not None

    def detect(self, image: np.ndarray, *, ocr_text_boxes: list[Any] | None = None) -> LayoutDetectionResult:
        if not self.is_available():
            return skipped_result(self.name, "layoutparser is not installed")
        model = self.config.get("model")
        model_factory = self.config.get("model_factory")
        if model is None and model_factory is not None:
            model = model_factory()
        if model is None:
            return skipped_result(self.name, "no LayoutParser model or model_factory configured")

        layout = model.detect(image)
        regions: list[ImageRegion] = []
        for index, block in enumerate(layout):
            block_type = str(getattr(block, "type", "unknown")).lower()
            coords = getattr(getattr(block, "block", block), "coordinates", None)
            if coords is None:
                continue
            x1, y1, x2, y2 = [int(value) for value in coords]
            regions.append(
                ImageRegion(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    region_type=_layout_label_to_region_type(block_type),
                    confidence=getattr(block, "score", None),
                    id=f"layoutparser_{index:03d}",
                    metadata={"label": block_type, "strategy": self.name},
                )
            )
        return LayoutDetectionResult(self.name, regions=regions)


def _layout_label_to_region_type(label: str) -> str:
    if "figure" in label or "image" in label:
        return "figure"
    if "table" in label:
        return "table"
    if "text" in label or "title" in label or "caption" in label or "list" in label:
        return "text"
    return "unknown"
