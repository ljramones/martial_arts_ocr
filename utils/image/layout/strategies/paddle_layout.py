from __future__ import annotations

import importlib.util
import os
from pathlib import Path
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

        cache_dir = self.config.get("cache_dir") or self.config.get("model_dir")
        if cache_dir:
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(cache_dir)))

        engine = self.config.get("engine")
        if engine is None:
            try:
                engine = _build_paddle_engine(self.config)
            except Exception as exc:
                return skipped_result(self.name, f"paddle layout engine unavailable: {exc}")
        try:
            raw = _run_paddle_engine(engine, image)
        except Exception as exc:
            return skipped_result(self.name, f"paddle layout inference failed: {exc}")
        regions: list[ImageRegion] = []
        for index, item in enumerate(_flatten_items(raw)):
            bbox = _bbox(item)
            if not bbox or len(bbox) < 4:
                continue
            x, y, width, height = bbox
            label = str(item.get("type") or item.get("label") or item.get("category") or item.get("class") or "unknown").lower()
            regions.append(
                ImageRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    region_type=_layout_label_to_region_type(label),
                    confidence=_score(item),
                    id=f"paddle_{index:03d}",
                    metadata={
                        "label": label,
                        "raw_label": label,
                        "layout_label": _layout_label_to_layout_label(label),
                        "strategy": self.name,
                        "layout_backend": self.name,
                    },
                )
            )
        return LayoutDetectionResult(
            self.name,
            regions=regions,
            metadata={
                "raw_count": len(raw or []) if isinstance(raw, list) else 1,
                "api_variant": engine.__class__.__name__,
            },
        )


def _build_paddle_engine(config: dict[str, Any]) -> Any:
    import paddleocr

    if hasattr(paddleocr, "PPStructureV3"):
        return paddleocr.PPStructureV3(**dict(config.get("constructor_kwargs", {})))
    if hasattr(paddleocr, "PPStructure"):
        return paddleocr.PPStructure(
            show_log=False,
            layout=True,
            ocr=bool(config.get("ocr", False)),
        )
    raise RuntimeError("paddleocr exposes neither PPStructureV3 nor PPStructure")


def _run_paddle_engine(engine: Any, image: np.ndarray) -> Any:
    if hasattr(engine, "predict"):
        return engine.predict(image)
    return engine(image)


def _flatten_items(raw: Any) -> list[dict[str, Any]]:
    raw = _unwrap(raw)
    if isinstance(raw, list):
        items: list[dict[str, Any]] = []
        for item in raw:
            items.extend(_flatten_items(item))
        return items
    if isinstance(raw, dict):
        for key in (
            "layout_det_res",
            "region_det_res",
            "layout",
            "res",
            "boxes",
            "regions",
            "results",
            "table_res_list",
            "seal_res_list",
            "chart_res_list",
            "formula_res_list",
        ):
            value = raw.get(key)
            if isinstance(value, (list, dict)):
                return _flatten_items(value)
        return [raw]
    return []


def _unwrap(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, type(None))):
        return value
    for attr in ("json", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                continue
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


def _bbox(item: dict[str, Any]) -> list[int] | None:
    for key in ("bbox", "box", "coordinate", "coordinates"):
        normalized = _normalize_bbox(item.get(key))
        if normalized is not None:
            return normalized
    layout = item.get("layout")
    if isinstance(layout, dict):
        return _bbox(layout)
    return None


def _normalize_bbox(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if {"x", "y", "width", "height"}.issubset(value):
            return [int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"])]
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return _xyxy_to_xywh([value["x1"], value["y1"], value["x2"], value["y2"]])
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        if all(isinstance(point, (list, tuple)) for point in value[:4]):
            xs = [float(point[0]) for point in value[:4]]
            ys = [float(point[1]) for point in value[:4]]
            return [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
        return _xyxy_to_xywh(value[:4])
    return None


def _xyxy_to_xywh(values: Any) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in values[:4]]
    if x2 > x1 and y2 > y1:
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    return [int(x1), int(y1), int(x2), int(y2)]


def _score(item: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "prob"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _layout_label_to_region_type(label: str) -> str:
    layout_label = _layout_label_to_layout_label(label)
    if layout_label in {"figure", "image", "photo", "diagram"}:
        return "figure"
    if layout_label == "table":
        return "table"
    if layout_label in {"text", "title", "caption"}:
        return "text"
    return "unknown"


def _layout_label_to_layout_label(label: str) -> str:
    normalized = label.lower()
    if normalized in {"figure", "image", "photo", "diagram"}:
        return normalized
    if "table" in normalized:
        return "table"
    if "caption" in normalized:
        return "caption"
    if "title" in normalized:
        return "title"
    if "text" in normalized or "paragraph" in normalized:
        return "text"
    return "unknown"
