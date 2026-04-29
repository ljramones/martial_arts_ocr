"""Adapters from extraction utilities into canonical document models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from martial_arts_ocr.pipeline.document_models import (
    BoundingBox,
    DocumentResult,
    ImageRegion,
    PageResult,
    TextRegion,
)


def bounding_box_from_extraction(value: Any) -> BoundingBox | None:
    """Normalize utility region shapes into a canonical bounding box."""
    region = _value(value, "region", value)
    bbox = _value(region, "bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
        return BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))

    x = _value(region, "x", _value(region, "x1"))
    y = _value(region, "y", _value(region, "y1"))
    width = _value(region, "width")
    height = _value(region, "height")
    if None in (x, y, width, height):
        return None
    return BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))


def text_region_from_extraction(value: Any, *, index: int = 1) -> TextRegion:
    """Convert text extraction output into a canonical text region."""
    text = _value(value, "text", _value(value, "text_content", value if isinstance(value, str) else ""))
    return TextRegion(
        region_id=str(_value(value, "region_id", _value(value, "id", f"text_{index}"))),
        text=str(text or ""),
        bbox=bounding_box_from_extraction(value),
        confidence=_float_or_none(_value(value, "confidence", _value(value, "score"))),
        language=_value(value, "language"),
        reading_order=int(_value(value, "reading_order", index) or index),
        metadata=_metadata(value),
    )


def image_region_from_extraction(value: Any, *, index: int = 1) -> ImageRegion:
    """Convert image extraction output into a canonical image region."""
    image_path = _value(value, "image_path")
    region = _value(value, "region", value)
    metadata = _metadata(value)
    region_metadata = _metadata(region)
    if region_metadata:
        metadata.update(region_metadata)
    return ImageRegion(
        region_id=str(_value(value, "region_id", _value(region, "id", f"image_{index}"))),
        image_path=Path(image_path) if image_path else None,
        bbox=bounding_box_from_extraction(value),
        region_type=_region_type(_value(value, "image_type", _value(region, "region_type", "image"))),
        caption=_value(value, "caption", _value(value, "description")),
        confidence=_float_or_none(_value(value, "confidence", _value(region, "score"))),
        reading_order=int(_value(value, "reading_order", index) or index),
        metadata=metadata,
    )


def page_result_from_extractions(
    *,
    page_number: int = 1,
    source_width: int | None = None,
    source_height: int | None = None,
    raw_text: str = "",
    text_items: list[Any] | None = None,
    image_items: list[Any] | None = None,
    confidence: float | None = None,
) -> PageResult:
    """Build a page result from extracted text/image utility records."""
    text_regions = [
        text_region_from_extraction(item, index=index)
        for index, item in enumerate(text_items or [], start=1)
    ]
    image_regions = [
        image_region_from_extraction(item, index=index)
        for index, item in enumerate(image_items or [], start=1)
    ]
    return PageResult(
        page_number=page_number,
        width=source_width,
        height=source_height,
        text_regions=text_regions,
        image_regions=image_regions,
        raw_text=raw_text,
        confidence=confidence,
    )


def document_result_from_extractions(
    *,
    source_path: str | Path,
    document_id: int | None = None,
    pages: list[PageResult],
    detected_languages: list[str] | None = None,
    confidence: float | None = None,
) -> DocumentResult:
    """Build a document result from canonical page extraction results."""
    return DocumentResult(
        document_id=document_id,
        source_path=Path(source_path),
        pages=pages,
        detected_languages=detected_languages or [],
        confidence=confidence,
    )


def _value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _metadata(value: Any) -> dict[str, Any]:
    metadata = _value(value, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _region_type(value: Any) -> str:
    allowed = {"image", "diagram", "table", "seal", "unknown"}
    region_type = str(value or "image").lower()
    if region_type == "figure":
        return "diagram"
    return region_type if region_type in allowed else "image"
