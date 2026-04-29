"""OCR text-box geometry helpers.

This module uses an explicit ``(x, y, width, height)`` box convention. Helpers
accept common object/dict shapes at the edges, but all returned boxes are
normalized to that convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


BBoxXYWH = tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRTextBox:
    """OCR text geometry in pixel coordinates.

    ``bbox`` is always ``(x, y, width, height)``.
    """

    text: str
    bbox: BBoxXYWH
    confidence: float | None = None
    language: str | None = None
    level: Literal["word", "line", "block"] = "word"

    def to_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bbox
        return {
            "text": self.text,
            "bbox": {"x": x, "y": y, "width": width, "height": height},
            "confidence": self.confidence,
            "language": self.language,
            "level": self.level,
        }


def normalize_bbox(value: Any, *, convention: str = "xywh") -> BBoxXYWH:
    """Normalize a bbox-like value to ``(x, y, width, height)``.

    Supported inputs:
    - tuple/list ``(x, y, width, height)`` when ``convention="xywh"``
    - tuple/list ``(x1, y1, x2, y2)`` when ``convention="xyxy"``
    - dict/object with ``x/y/width/height``
    - dict/object with ``x1/y1/x2/y2`` or ``bbox``
    - utility image regions with ``x/y/width/height`` properties
    """
    if isinstance(value, OCRTextBox):
        return value.bbox

    bbox = _value(value, "bbox")
    if bbox is not None and bbox is not value:
        if isinstance(bbox, dict):
            return normalize_bbox(bbox, convention=convention)
        if len(bbox) >= 4:
            return _normalize_sequence(bbox[:4], convention=convention)

    x = _value(value, "x")
    y = _value(value, "y")
    width = _value(value, "width")
    height = _value(value, "height")
    if None not in (x, y, width, height):
        return _positive_xywh(int(x), int(y), int(width), int(height))

    x1 = _value(value, "x1")
    y1 = _value(value, "y1")
    x2 = _value(value, "x2")
    y2 = _value(value, "y2")
    if None not in (x1, y1, x2, y2):
        return _normalize_sequence((x1, y1, x2, y2), convention="xyxy")

    polygon = _value(value, "polygon", _value(value, "points"))
    if polygon is not None:
        return bbox_from_polygon(polygon)

    if isinstance(value, (tuple, list)) and len(value) >= 4:
        return _normalize_sequence(value[:4], convention=convention)

    raise ValueError(f"Unsupported bbox value: {value!r}")


def text_box_from_value(value: Any, *, convention: str = "xywh") -> OCRTextBox:
    if isinstance(value, OCRTextBox):
        return value
    return OCRTextBox(
        text=str(_value(value, "text", "")),
        bbox=normalize_bbox(value, convention=convention),
        confidence=_float_or_none(_value(value, "confidence", _value(value, "score"))),
        language=_value(value, "language"),
        level=str(_value(value, "level", "word")),
    )


def bbox_from_polygon(points: Any) -> BBoxXYWH:
    """Convert a polygon/list of points into its enclosing ``xywh`` bbox."""
    xs: list[float] = []
    ys: list[float] = []
    for point in points or []:
        if isinstance(point, dict):
            x = _value(point, "x")
            y = _value(point, "y")
        else:
            if len(point) < 2:
                continue
            x, y = point[0], point[1]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        raise ValueError(f"Unsupported polygon value: {points!r}")
    x1, y1 = int(round(min(xs))), int(round(min(ys)))
    x2, y2 = int(round(max(xs))), int(round(max(ys)))
    return _positive_xywh(x1, y1, x2 - x1, y2 - y1)


def normalize_text_boxes(values: list[Any] | tuple[Any, ...] | None, *, convention: str = "xywh") -> list[OCRTextBox]:
    boxes: list[OCRTextBox] = []
    for value in values or []:
        try:
            box = text_box_from_value(value, convention=convention)
        except (TypeError, ValueError):
            continue
        if bbox_area(box.bbox) > 0:
            boxes.append(box)
    return boxes


def bbox_area(bbox: Any, *, convention: str = "xywh") -> int:
    _x, _y, width, height = normalize_bbox(bbox, convention=convention)
    return max(0, int(width)) * max(0, int(height))


def bbox_intersection_area(first: Any, second: Any, *, convention: str = "xywh") -> int:
    ax, ay, aw, ah = normalize_bbox(first, convention=convention)
    bx, by, bw, bh = normalize_bbox(second, convention=convention)
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_bbox_overlap_ratio(candidate_bbox: Any, text_boxes: list[Any] | tuple[Any, ...] | None) -> float:
    """Return candidate area covered by intersecting OCR text boxes."""
    candidate_area = max(1, bbox_area(candidate_bbox))
    intersections = sum(bbox_intersection_area(candidate_bbox, box) for box in normalize_text_boxes(text_boxes))
    return min(1.0, float(intersections) / float(candidate_area))


def compute_text_box_coverage(candidate_bbox: Any, text_boxes: list[Any] | tuple[Any, ...] | None) -> float:
    """Return intersecting OCR text area covered by the candidate region."""
    boxes = normalize_text_boxes(text_boxes)
    text_area = sum(bbox_area(box.bbox) for box in boxes)
    if text_area <= 0:
        return 0.0
    intersections = sum(bbox_intersection_area(candidate_bbox, box.bbox) for box in boxes)
    return min(1.0, float(intersections) / float(text_area))


def compute_text_box_count(candidate_bbox: Any, text_boxes: list[Any] | tuple[Any, ...] | None) -> int:
    return sum(1 for box in normalize_text_boxes(text_boxes) if bbox_intersection_area(candidate_bbox, box.bbox) > 0)


def compute_mean_text_confidence(candidate_bbox: Any, text_boxes: list[Any] | tuple[Any, ...] | None) -> float | None:
    confidences = [
        float(box.confidence)
        for box in normalize_text_boxes(text_boxes)
        if box.confidence is not None and bbox_intersection_area(candidate_bbox, box.bbox) > 0
    ]
    if not confidences:
        return None
    return float(sum(confidences) / len(confidences))


def build_text_mask(
    image_size: tuple[int, int],
    text_boxes: list[Any] | tuple[Any, ...] | None,
    *,
    dilation_px: int = 4,
) -> np.ndarray:
    """Build a uint8 mask from OCR boxes.

    ``image_size`` is ``(height, width)``. Text pixels are 255, background is 0.
    """
    height, width = int(image_size[0]), int(image_size[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in normalize_text_boxes(text_boxes):
        x, y, bw, bh = box.bbox
        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + bw))
        y2 = max(0, min(height, y + bh))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    if dilation_px > 0 and np.any(mask):
        import cv2

        kernel_size = max(1, int(dilation_px) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def compute_text_mask_overlap(candidate_bbox: Any, text_mask: np.ndarray | None) -> float:
    """Return candidate area covered by a text mask."""
    if text_mask is None:
        return 0.0
    x, y, width, height = normalize_bbox(candidate_bbox)
    x1 = max(0, min(text_mask.shape[1], x))
    y1 = max(0, min(text_mask.shape[0], y))
    x2 = max(0, min(text_mask.shape[1], x + width))
    y2 = max(0, min(text_mask.shape[0], y + height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = text_mask[y1:y2, x1:x2]
    return float(np.mean(roi > 0)) if roi.size else 0.0


def _normalize_sequence(values: Any, *, convention: str) -> BBoxXYWH:
    a, b, c, d = [int(round(float(value))) for value in values]
    if convention == "xyxy":
        return _positive_xywh(a, b, c - a, d - b)
    if convention != "xywh":
        raise ValueError(f"Unsupported bbox convention: {convention}")
    return _positive_xywh(a, b, c, d)


def _positive_xywh(x: int, y: int, width: int, height: int) -> BBoxXYWH:
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)
    return int(x), int(y), max(0, int(width)), max(0, int(height))


def _value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
