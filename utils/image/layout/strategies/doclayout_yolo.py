from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from utils.image.layout.strategy import LayoutDetectionResult, skipped_result
from utils.image.regions.core_types import ImageRegion


class DocLayoutYOLOStrategy:
    """Optional DocLayout-YOLO style adapter.

    This adapter intentionally does not download models. Provide a local model
    path through config to run it.
    """

    name = "doclayout_yolo"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("ultralytics") is not None

    def detect(self, image: np.ndarray) -> LayoutDetectionResult:
        model_path = self.config.get("model_path") or self.config.get(f"{self.name}_model_path")
        if not self.is_available():
            return skipped_result(self.name, "ultralytics is not installed")
        if not model_path:
            return skipped_result(self.name, f"no {self.name} model_path configured")
        if not Path(model_path).exists():
            return skipped_result(self.name, f"model_path does not exist: {model_path}")

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        result = model.predict(
            source=image,
            imgsz=int(self.config.get("imgsz", 1024)),
            conf=float(self.config.get("conf", 0.25)),
            iou=float(self.config.get("iou", 0.60)),
            verbose=False,
        )[0]
        names = getattr(model, "names", {}) or {}
        regions: list[ImageRegion] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), score, cls_idx in zip(boxes, confs, classes):
                label = str(names.get(int(cls_idx), int(cls_idx))).lower()
                regions.append(
                    ImageRegion(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                        region_type=_layout_label_to_region_type(label),
                        confidence=float(score),
                        metadata={"label": label, "strategy": self.name},
                    )
                )
        return LayoutDetectionResult(self.name, regions=regions, metadata={"model_path": str(model_path)})


def _layout_label_to_region_type(label: str) -> str:
    if "figure" in label or "image" in label or "photo" in label:
        return "figure"
    if "table" in label:
        return "table"
    if "title" in label or "text" in label or "caption" in label or "list" in label:
        return "text"
    return "unknown"
