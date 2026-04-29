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
    median_height = _median_word_height(ordered)
    tolerance = y_tolerance if y_tolerance is not None else _default_y_tolerance(ordered)
    line_groups: list[list[TextRegion]] = []

    for word in ordered:
        best_index: int | None = None
        best_score: float | None = None
        for index, group in enumerate(line_groups):
            if not _belongs_to_line(word, group, tolerance=tolerance, median_height=median_height):
                continue
            score = abs(_center_y(word) - _group_center_y(group))
            if best_score is None or score < best_score:
                best_index = index
                best_score = score
        if best_index is None:
            line_groups.append([word])
        else:
            line_groups[best_index].append(word)

    lines: list[TextRegion] = []
    for index, group in enumerate(line_groups, start=1):
        sorted_group = sorted(group, key=lambda region: region.bbox.x if region.bbox else 0)
        text = _join_words_with_spacing(sorted_group, median_height=median_height)
        bbox = _union_bbox(sorted_group)
        confidences = [region.confidence for region in sorted_group if region.confidence is not None]
        confidence = sum(confidences) / len(confidences) if confidences else None
        language = _common_language(sorted_group)
        source_ids = [region.region_id for region in sorted_group]
        reading_order_uncertain = _line_reading_order_uncertain(sorted_group, median_height=median_height)
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
                    "line_grouping_method": "adaptive_center_overlap_v1",
                    "reading_order_uncertain": reading_order_uncertain,
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
    return max(4, int(round(_median(heights) * 0.55)))


def _median_word_height(words: list[TextRegion]) -> float:
    heights = [region.bbox.height for region in words if region.bbox is not None]
    return _median(heights) if heights else 12.0


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _belongs_to_line(
    word: TextRegion,
    group: list[TextRegion],
    *,
    tolerance: int,
    median_height: float,
) -> bool:
    word_box = word.bbox
    group_box = _union_bbox(group)
    if word_box is None or group_box is None:
        return False

    center_delta = abs(_center_y(word) - _group_center_y(group))
    if center_delta <= tolerance:
        return True

    overlap_ratio = _vertical_overlap_ratio(word_box, group_box)
    if overlap_ratio >= 0.65 and center_delta <= max(tolerance + 2, median_height * 0.6):
        return True

    return False


def _vertical_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    top = max(a.y, b.y)
    bottom = min(a.y + a.height, b.y + b.height)
    overlap = max(0, bottom - top)
    return overlap / max(1, min(a.height, b.height))


def _join_words_with_spacing(words: list[TextRegion], *, median_height: float) -> str:
    text = ""
    previous_box: BoundingBox | None = None
    for word in words:
        if not word.text:
            continue
        if not text:
            text = word.text
        elif previous_box is not None and word.bbox is not None:
            gap = word.bbox.x - (previous_box.x + previous_box.width)
            if gap > max(40, median_height * 3.0):
                text += "  " + word.text
            else:
                text += " " + word.text
        else:
            text += " " + word.text
        previous_box = word.bbox or previous_box
    return text


def _line_reading_order_uncertain(words: list[TextRegion], *, median_height: float) -> bool:
    if len(words) <= 1:
        return False
    sorted_words = sorted(words, key=lambda region: region.bbox.x if region.bbox else 0)
    boxes = [region.bbox for region in sorted_words if region.bbox is not None]
    if len(boxes) <= 1:
        return False
    large_gaps = [
        boxes[index].x - (boxes[index - 1].x + boxes[index - 1].width)
        for index in range(1, len(boxes))
    ]
    return any(gap > max(120, median_height * 8.0) for gap in large_gaps)


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
