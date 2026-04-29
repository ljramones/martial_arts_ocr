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
from utils.text.geometry import bbox_from_polygon


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
    ocr_text_regions = ocr_text_regions_from_ocr_output(ocr_output)
    text_regions = _text_regions_from_any(_value(ocr_output, "text_regions"), text)
    if ocr_text_regions:
        text_regions = ocr_text_regions
    metadata = {"legacy_page": _safe_legacy_dict(ocr_output)}
    if ocr_text_regions:
        metadata["ocr_text_boxes"] = [region.to_dict() for region in ocr_text_regions]
    return PageResult(
        page_number=1,
        width=width,
        height=height,
        text_regions=text_regions,
        image_regions=_image_regions_from_any(
            _value(ocr_output, "image_regions"),
            _value(ocr_output, "extracted_images"),
        ),
        raw_text=text,
        confidence=confidence,
        metadata=metadata,
    )


def _page_from_page_output(page_output: Any, page_number: int) -> PageResult:
    text = _text_from_any(page_output)
    confidence = _float_or_none(
        _value(page_output, "confidence", _value(page_output, "overall_confidence"))
    )
    width, height = _dimensions_from_any(page_output)
    ocr_text_regions = ocr_text_regions_from_ocr_output(page_output)
    text_regions = _text_regions_from_any(_value(page_output, "text_regions"), text)
    if ocr_text_regions:
        text_regions = ocr_text_regions
    metadata = {"legacy_page": _safe_legacy_dict(page_output)}
    if ocr_text_regions:
        metadata["ocr_text_boxes"] = [region.to_dict() for region in ocr_text_regions]
    return PageResult(
        page_number=int(_value(page_output, "page_number", page_number) or page_number),
        width=width,
        height=height,
        text_regions=text_regions,
        image_regions=_image_regions_from_any(
            _value(page_output, "image_regions"),
            _value(page_output, "extracted_images"),
        ),
        raw_text=text,
        confidence=confidence,
        metadata=metadata,
    )


def ocr_text_regions_from_ocr_output(value: Any, *, engine: str | None = None) -> list[TextRegion]:
    """Extract OCR engine word/line boxes into canonical text regions."""
    boxes = ocr_text_boxes_from_ocr_output(value, engine=engine)
    regions: list[TextRegion] = []
    for index, box in enumerate(boxes, start=1):
        bbox = _bbox_from_any(box)
        if bbox is None:
            continue
        metadata = {
            "source": box.get("source", "ocr_engine"),
            "engine": box.get("engine", engine or "unknown"),
            "ocr_level": box.get("ocr_level", box.get("level", "word")),
        }
        if "polygon" in box:
            metadata["polygon"] = box["polygon"]
        regions.append(
            TextRegion(
                region_id=str(box.get("region_id") or box.get("id") or f"ocr_{metadata['ocr_level']}_{index}"),
                text=str(box.get("text", "")),
                bbox=bbox,
                confidence=_float_or_none(box.get("confidence", box.get("conf", box.get("score")))),
                language=box.get("language"),
                reading_order=index,
                metadata=metadata,
            )
        )
    return regions


def ocr_text_boxes_from_ocr_output(value: Any, *, engine: str | None = None) -> list[dict[str, Any]]:
    """Normalize common OCR output box shapes.

    Supported shapes include OCRResult.bounding_boxes, dicts with
    ``ocr_text_boxes``/``words``/``lines``/``blocks``, Tesseract TSV-style rows,
    EasyOCR tuples, and existing TextRegion-like objects.
    """
    candidates: list[Any] = []
    explicit_engine = engine or _value(value, "engine")

    for key in ("ocr_text_boxes", "bounding_boxes", "words", "lines", "blocks"):
        items = _value(value, key)
        if items:
            candidates.extend(_iter_items(items))

    best_ocr_result = _value(value, "best_ocr_result")
    if best_ocr_result is not None and best_ocr_result is not value:
        candidates.extend(ocr_text_boxes_from_ocr_output(best_ocr_result, engine=_value(best_ocr_result, "engine", explicit_engine)))

    for result in _value(value, "ocr_results", []) or []:
        candidates.extend(ocr_text_boxes_from_ocr_output(result, engine=_value(result, "engine", explicit_engine)))

    normalized: list[dict[str, Any]] = []
    for item in candidates:
        box = _ocr_box_from_any(item, engine=explicit_engine)
        if box is not None:
            normalized.append(box)
    return normalized


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
    if isinstance(bbox, dict):
        x = _value(bbox, "x")
        y = _value(bbox, "y")
        width = _value(bbox, "width")
        height = _value(bbox, "height")
        if None not in (x, y, width, height):
            return BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
    elif bbox and len(bbox) >= 4:
        if _value(region, "bbox_convention") == "xywh":
            x, y, width, height = [int(n) for n in bbox[:4]]
            return BoundingBox(x=x, y=y, width=max(0, width), height=max(0, height))
        x1, y1, x2, y2 = [int(n) for n in bbox[:4]]
        return BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))

    polygon = _value(region, "polygon", _value(region, "points"))
    if polygon:
        x, y, width, height = bbox_from_polygon(polygon)
        return BoundingBox(x=x, y=y, width=width, height=height)

    x = _value(region, "x", _value(region, "x1"))
    y = _value(region, "y", _value(region, "y1"))
    if x is None:
        x = _value(region, "left")
    if y is None:
        y = _value(region, "top")
    width = _value(region, "width")
    height = _value(region, "height")
    if x is None or y is None or width is None or height is None:
        return None
    return BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))


def _ocr_box_from_any(value: Any, *, engine: str | None = None) -> dict[str, Any] | None:
    if isinstance(value, TextRegion):
        if value.bbox is None:
            return None
        return {
            "text": value.text,
            "x": value.bbox.x,
            "y": value.bbox.y,
            "width": value.bbox.width,
            "height": value.bbox.height,
            "confidence": value.confidence,
            "language": value.language,
            "source": value.metadata.get("source", "ocr_engine"),
            "engine": value.metadata.get("engine", engine or "unknown"),
            "ocr_level": value.metadata.get("ocr_level", value.metadata.get("level", "word")),
        }

    if isinstance(value, (tuple, list)) and len(value) >= 3 and isinstance(value[0], (tuple, list)):
        polygon, text, confidence = value[0], value[1], value[2]
        x, y, width, height = bbox_from_polygon(polygon)
        return _box_dict(
            text=text,
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=confidence,
            engine=engine or "easyocr",
            ocr_level="line",
            polygon=polygon,
        )

    text = _value(value, "text", "")
    conf = _value(value, "confidence", _value(value, "conf", _value(value, "score")))
    ocr_level = _ocr_level_from_any(_value(value, "ocr_level", _value(value, "level", "word")))
    source = _value(value, "source", "ocr_engine")
    item_engine = _value(value, "engine", engine or "unknown")

    try:
        bbox = _bbox_from_any(value)
    except Exception:
        bbox = None
    if bbox is None:
        return None
    return _box_dict(
        text=text,
        x=bbox.x,
        y=bbox.y,
        width=bbox.width,
        height=bbox.height,
        confidence=conf,
        language=_value(value, "language"),
        source=source,
        engine=item_engine,
        ocr_level=ocr_level,
        polygon=_value(value, "polygon", _value(value, "points")),
    )


def _box_dict(
    *,
    text: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    confidence: Any = None,
    language: Any = None,
    source: str = "ocr_engine",
    engine: str = "unknown",
    ocr_level: str = "word",
    polygon: Any = None,
) -> dict[str, Any]:
    box = {
        "text": str(text or ""),
        "x": int(x),
        "y": int(y),
        "width": max(0, int(width)),
        "height": max(0, int(height)),
        "confidence": _ocr_confidence(confidence),
        "language": language,
        "source": source,
        "engine": engine,
        "ocr_level": ocr_level,
        "bbox_convention": "xywh",
    }
    if polygon is not None:
        box["polygon"] = polygon
    return box


def _ocr_confidence(value: Any) -> float | None:
    confidence = _float_or_none(value)
    if confidence is None:
        return None
    if confidence > 1.0 and confidence <= 100.0:
        return confidence / 100.0
    return confidence


def _ocr_level_from_any(value: Any) -> str:
    if isinstance(value, int):
        return {
            1: "block",
            2: "block",
            3: "block",
            4: "line",
            5: "word",
        }.get(value, "word")
    text = str(value or "word").lower()
    if text.isdigit():
        return _ocr_level_from_any(int(text))
    return text if text in {"word", "line", "block"} else "word"


def _iter_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        if {"text", "left", "top", "width", "height"}.issubset(value.keys()):
            texts = value.get("text", [])
            length = len(texts) if isinstance(texts, list) else 0
            return [
                {
                    "text": value.get("text", [""] * length)[index],
                    "left": value.get("left", [0] * length)[index],
                    "top": value.get("top", [0] * length)[index],
                    "width": value.get("width", [0] * length)[index],
                    "height": value.get("height", [0] * length)[index],
                    "conf": value.get("conf", [None] * length)[index],
                    "level": value.get("level", ["word"] * length)[index],
                    "engine": value.get("engine", "tesseract"),
                }
                for index in range(length)
            ]
        return list(value.values())
    return list(value or [])


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
