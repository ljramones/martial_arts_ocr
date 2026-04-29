from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from utils.image.layout.strategy import LayoutDetectionResult, skipped_result
from utils.image.regions.core_types import ImageRegion


class PaddleLayoutStrategy:
    """Optional PaddleOCR PP-Structure layout adapter."""

    name = "paddle_ppstructure"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def detect(self, image: np.ndarray, *, ocr_text_boxes: list[Any] | None = None) -> LayoutDetectionResult:
        if not self.is_available():
            return skipped_result(self.name, "paddleocr is not installed")

        from paddleocr import PPStructure

        engine = self.config.get("engine")
        if engine is None:
            engine = PPStructure(
                show_log=False,
                layout=True,
                ocr=bool(self.config.get("ocr", False)),
            )
        raw = engine(image)
        regions: list[ImageRegion] = []
        for index, item in enumerate(raw or []):
            bbox = item.get("bbox") or item.get("layout", {}).get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
            label = str(item.get("type", item.get("label", "unknown"))).lower()
            regions.append(
                ImageRegion(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    region_type=_layout_label_to_region_type(label),
                    confidence=item.get("score"),
                    id=f"paddle_{index:03d}",
                    metadata={"label": label, "strategy": self.name},
                )
            )
        return LayoutDetectionResult(self.name, regions=regions, metadata={"raw_count": len(raw or [])})


def _layout_label_to_region_type(label: str) -> str:
    if "figure" in label or "image" in label:
        return "figure"
    if "table" in label:
        return "table"
    if "text" in label or "title" in label or "caption" in label:
        return "text"
    return "unknown"
