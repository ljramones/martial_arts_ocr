"""Canonical text hierarchy helpers for OCR output."""

from __future__ import annotations

from dataclasses import replace

from martial_arts_ocr.pipeline.document_models import BoundingBox, PageResult, TextRegion


def group_word_regions_into_lines(
    word_regions: list[TextRegion],
    *,
    y_tolerance: int | None = None,
) -> list[TextRegion]:
    """Group OCR word regions into deterministic line-level text regions."""

    words = [
        region for region in word_regions
        if region.bbox is not None and (region.metadata or {}).get("ocr_level") == "word"
    ]
    if not words:
        return []

    ordered = sorted(words, key=lambda region: (_center_y(region), region.bbox.x if region.bbox else 0))
    tolerance = y_tolerance if y_tolerance is not None else _default_y_tolerance(ordered)
    line_groups: list[list[TextRegion]] = []

    for word in ordered:
        center_y = _center_y(word)
        for group in line_groups:
            if abs(center_y - _group_center_y(group)) <= tolerance:
                group.append(word)
                break
        else:
            line_groups.append([word])

    lines: list[TextRegion] = []
    for index, group in enumerate(line_groups, start=1):
        sorted_group = sorted(group, key=lambda region: region.bbox.x if region.bbox else 0)
        text = " ".join(region.text for region in sorted_group if region.text)
        bbox = _union_bbox(sorted_group)
        confidences = [region.confidence for region in sorted_group if region.confidence is not None]
        confidence = sum(confidences) / len(confidences) if confidences else None
        language = _common_language(sorted_group)
        source_ids = [region.region_id for region in sorted_group]
        lines.append(
            TextRegion(
                region_id=f"ocr_line_{index}",
                text=text,
                bbox=bbox,
                confidence=confidence,
                language=language,
                reading_order=index,
                metadata={
                    "source": "ocr_normalization",
                    "ocr_level": "line",
                    "derived_from_count": len(source_ids),
                    "derived_from": source_ids,
                    "source_engines": sorted(
                        {
                            str(region.metadata.get("engine"))
                            for region in sorted_group
                            if region.metadata.get("engine")
                        }
                    ),
                },
            )
        )
    return lines


def build_readable_page_text(line_regions: list[TextRegion]) -> str:
    """Build compact page text from line-level regions."""

    ordered = sorted(
        [region for region in line_regions if region.text],
        key=lambda region: (
            region.reading_order if region.reading_order is not None else 10**9,
            region.bbox.y if region.bbox else 0,
            region.bbox.x if region.bbox else 0,
        ),
    )
    return "\n".join(region.text for region in ordered)


def compact_ocr_metadata_for_page(page: PageResult) -> dict:
    """Return compact OCR hierarchy metadata for a page."""

    word_count = sum(
        1 for region in page.text_regions
        if (region.metadata or {}).get("ocr_level") == "word"
    )
    line_count = sum(
        1 for region in page.text_regions
        if (region.metadata or {}).get("ocr_level") == "line"
    )
    return {
        "readable_text": page.metadata.get("readable_text", page.combined_text()),
        "ocr_word_count": word_count,
        "ocr_line_count": line_count,
    }


def add_readable_lines_to_page(page: PageResult) -> PageResult:
    """Add derived line regions and compact readable text metadata to a page."""

    word_regions = [
        region for region in page.text_regions
        if (region.metadata or {}).get("ocr_level") == "word"
    ]
    if not word_regions:
        metadata = dict(page.metadata)
        metadata.setdefault("readable_text", page.combined_text())
        metadata.setdefault("ocr_word_count", 0)
        metadata.setdefault("ocr_line_count", 0)
        return replace(page, metadata=metadata)

    line_regions = group_word_regions_into_lines(word_regions)
    readable_text = build_readable_page_text(line_regions)
    existing_non_line_regions = [
        region for region in page.text_regions
        if (region.metadata or {}).get("source") != "ocr_normalization"
    ]
    metadata = dict(page.metadata)
    metadata["readable_text"] = readable_text
    metadata["ocr_word_count"] = len(word_regions)
    metadata["ocr_line_count"] = len(line_regions)
    return replace(
        page,
        text_regions=[*line_regions, *existing_non_line_regions],
        raw_text=page.raw_text or readable_text,
        metadata=metadata,
    )


def _center_y(region: TextRegion) -> float:
    bbox = region.bbox
    return float((bbox.y if bbox else 0) + (bbox.height if bbox else 0) / 2)


def _group_center_y(group: list[TextRegion]) -> float:
    return sum(_center_y(region) for region in group) / max(1, len(group))


def _default_y_tolerance(words: list[TextRegion]) -> int:
    heights = [region.bbox.height for region in words if region.bbox is not None]
    if not heights:
        return 8
    return max(4, int(round(sum(heights) / len(heights) * 0.6)))


def _union_bbox(regions: list[TextRegion]) -> BoundingBox | None:
    boxes = [region.bbox for region in regions if region.bbox is not None]
    if not boxes:
        return None
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x + box.width for box in boxes)
    y2 = max(box.y + box.height for box in boxes)
    return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _common_language(regions: list[TextRegion]) -> str | None:
    languages = [region.language for region in regions if region.language]
    if not languages:
        return None
    return max(set(languages), key=languages.count)
