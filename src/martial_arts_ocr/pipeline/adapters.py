"""Adapters from legacy processor output into canonical pipeline models."""

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


def document_result_from_ocr_output(
    ocr_output: Any,
    *,
    document_id: int | None,
    source_path: Path,
    language_hint: str | None = None,
) -> DocumentResult:
    """Convert current loose OCR outputs into a canonical document result."""

    if isinstance(ocr_output, DocumentResult):
        return ocr_output

    pages_value = _value(ocr_output, "pages")
    pages = _pages_from_any(pages_value) if pages_value else []

    if not pages:
        pages = [_page_from_flat_output(ocr_output)]

    detected_languages = _detected_languages(ocr_output)
    confidence = _float_or_none(
        _value(ocr_output, "confidence", _value(ocr_output, "overall_confidence"))
    )
    metadata = {
        "source_shape": type(ocr_output).__name__,
        "legacy": _safe_legacy_dict(ocr_output),
        "ocr_engine": _value(_value(ocr_output, "best_ocr_result"), "engine", "unknown"),
        "processing_time": _float_or_none(_value(ocr_output, "processing_time")),
        "quality_score": _float_or_none(_value(ocr_output, "quality_score")),
        "has_japanese": _value(ocr_output, "japanese_result") is not None,
        "text_statistics": _value(ocr_output, "text_statistics"),
    }

    return DocumentResult(
        document_id=document_id,
        source_path=source_path,
        pages=pages,
        language_hint=language_hint,
        detected_languages=detected_languages,
        confidence=confidence,
        metadata=metadata,
    )


def _pages_from_any(pages_value: Any) -> list[PageResult]:
    if isinstance(pages_value, dict):
        pages_iterable = pages_value.values()
    else:
        pages_iterable = pages_value or []
    return [_page_from_page_output(page, index + 1) for index, page in enumerate(pages_iterable)]


def _page_from_flat_output(ocr_output: Any) -> PageResult:
    text = _text_from_any(ocr_output)
    confidence = _float_or_none(
        _value(ocr_output, "confidence", _value(ocr_output, "overall_confidence"))
    )
    width, height = _dimensions_from_any(ocr_output)
    return PageResult(
        page_number=1,
        width=width,
        height=height,
        text_regions=_text_regions_from_any(_value(ocr_output, "text_regions"), text),
        image_regions=_image_regions_from_any(
            _value(ocr_output, "image_regions"),
            _value(ocr_output, "extracted_images"),
        ),
        raw_text=text,
        confidence=confidence,
        metadata={"legacy_page": _safe_legacy_dict(ocr_output)},
    )


def _page_from_page_output(page_output: Any, page_number: int) -> PageResult:
    text = _text_from_any(page_output)
    confidence = _float_or_none(
        _value(page_output, "confidence", _value(page_output, "overall_confidence"))
    )
    width, height = _dimensions_from_any(page_output)
    return PageResult(
        page_number=int(_value(page_output, "page_number", page_number) or page_number),
        width=width,
        height=height,
        text_regions=_text_regions_from_any(_value(page_output, "text_regions"), text),
        image_regions=_image_regions_from_any(
            _value(page_output, "image_regions"),
            _value(page_output, "extracted_images"),
        ),
        raw_text=text,
        confidence=confidence,
        metadata={"legacy_page": _safe_legacy_dict(page_output)},
    )


def _text_from_any(value: Any) -> str:
    direct_text = _value(
        value,
        "text",
        _value(value, "cleaned_text", _value(value, "raw_text", None)),
    )
    if direct_text is not None:
        return str(direct_text)

    best_ocr_result = _value(value, "best_ocr_result")
    best_text = _value(best_ocr_result, "text")
    if best_text is not None:
        return str(best_text)

    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _text_regions_from_any(regions_value: Any, fallback_text: str) -> list[TextRegion]:
    regions = list(regions_value or [])
    if not regions and fallback_text:
        return [TextRegion(region_id="text_1", text=fallback_text, reading_order=1)]

    text_regions: list[TextRegion] = []
    for index, region in enumerate(regions, start=1):
        text = _value(region, "text", _value(region, "text_content", ""))
        text_regions.append(
            TextRegion(
                region_id=str(_value(region, "id", _value(region, "region_id", f"text_{index}"))),
                text=str(text or ""),
                bbox=_bbox_from_any(region),
                confidence=_float_or_none(
                    _value(region, "confidence", _value(region, "score"))
                ),
                language=_value(region, "language"),
                reading_order=index,
                metadata=_metadata_from_any(region),
            )
        )
    return text_regions


def _image_regions_from_any(regions_value: Any, extracted_images_value: Any = None) -> list[ImageRegion]:
    regions = list(regions_value or [])
    extracted_images = list(extracted_images_value or [])
    image_regions: list[ImageRegion] = []

    for index, region in enumerate(regions, start=1):
        region_type = str(_value(region, "region_type", "image") or "image")
        image_regions.append(
            ImageRegion(
                region_id=str(_value(region, "id", _value(region, "region_id", f"image_{index}"))),
                bbox=_bbox_from_any(region),
                region_type=_canonical_region_type(region_type),
                confidence=_float_or_none(
                    _value(region, "confidence", _value(region, "score"))
                ),
                reading_order=index,
                metadata=_metadata_from_any(region),
            )
        )

    offset = len(image_regions)
    for index, image in enumerate(extracted_images, start=1):
        region = _value(image, "region", image)
        image_path = _value(image, "image_path")
        image_regions.append(
            ImageRegion(
                region_id=str(_value(region, "id", _value(image, "region_id", f"image_{offset + index}"))),
                image_path=Path(image_path) if image_path else None,
                bbox=_bbox_from_any(region),
                region_type=_canonical_region_type(_value(image, "image_type", "image")),
                caption=_value(image, "description", _value(image, "caption")),
                confidence=_float_or_none(
                    _value(image, "confidence", _value(region, "score"))
                ),
                reading_order=offset + index,
                metadata=_metadata_from_any(image),
            )
        )
    return image_regions


def _bbox_from_any(value: Any) -> BoundingBox | None:
    region = _value(value, "region", value)
    bbox = _value(region, "bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(n) for n in bbox[:4]]
        return BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))

    x = _value(region, "x", _value(region, "x1"))
    y = _value(region, "y", _value(region, "y1"))
    width = _value(region, "width")
    height = _value(region, "height")
    if x is None or y is None or width is None or height is None:
        return None
    return BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))


def _dimensions_from_any(value: Any) -> tuple[int | None, int | None]:
    metadata = _value(value, "processing_metadata", {}) or {}
    dimensions = metadata.get("image_dimensions", {}) if isinstance(metadata, dict) else {}
    width = _value(value, "width", dimensions.get("width"))
    height = _value(value, "height", dimensions.get("height"))
    return _int_or_none(width), _int_or_none(height)


def _detected_languages(value: Any) -> list[str]:
    detected = _value(value, "detected_languages")
    if detected:
        return [str(language) for language in detected]

    languages: list[str] = []
    for segment in _value(value, "language_segments", []) or []:
        language = _value(segment, "language")
        if language and language not in languages:
            languages.append(str(language))
    return languages


def _metadata_from_any(value: Any) -> dict[str, Any]:
    metadata = _value(value, "metadata", {})
    if isinstance(metadata, dict):
        return dict(metadata)
    return {"metadata": metadata}


def _safe_legacy_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _json_safe(dict(value))
    if hasattr(value, "to_dict"):
        try:
            legacy = value.to_dict()
            return _json_safe(legacy if isinstance(legacy, dict) else {"value": legacy})
        except Exception:
            return {"repr": repr(value)}
    return {"repr": repr(value)}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return repr(value)
    return repr(value)


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


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _canonical_region_type(value: Any) -> str:
    allowed = {"image", "diagram", "table", "seal", "unknown"}
    region_type = str(value or "image").lower()
    return region_type if region_type in allowed else "image"
