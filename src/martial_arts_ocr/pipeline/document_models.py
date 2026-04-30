"""Canonical internal document result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RegionType = Literal["text", "image", "diagram", "table", "seal", "unknown"]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space rectangle for document regions."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class TextRegion:
    """Text-bearing region on a page."""

    region_id: str
    text: str
    bbox: BoundingBox | None = None
    confidence: float | None = None
    language: str | None = None
    reading_order: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "type": "text",
            "text": self.text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": self.confidence,
            "language": self.language,
            "reading_order": self.reading_order,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ImageRegion:
    """Image-bearing region on a page."""

    region_id: str
    image_path: Path | None = None
    bbox: BoundingBox | None = None
    region_type: RegionType = "image"
    caption: str | None = None
    confidence: float | None = None
    reading_order: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "type": self.region_type,
            "image_path": str(self.image_path) if self.image_path else None,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "caption": self.caption,
            "confidence": self.confidence,
            "reading_order": self.reading_order,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PageResult:
    """Canonical page-level result consumed by pipeline boundaries."""

    page_number: int
    width: int | None = None
    height: int | None = None
    text_regions: list[TextRegion] = field(default_factory=list)
    image_regions: list[ImageRegion] = field(default_factory=list)
    raw_text: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        if self.raw_text:
            return self.raw_text
        return "\n".join(region.text for region in self.text_regions if region.text)

    def line_regions(self) -> list[TextRegion]:
        return [
            region for region in self.text_regions
            if (region.metadata or {}).get("ocr_level") == "line"
        ]

    def word_regions(self) -> list[TextRegion]:
        return [
            region for region in self.text_regions
            if (region.metadata or {}).get("ocr_level") == "word"
        ]

    def text_summary(self) -> dict[str, Any]:
        line_regions = self.line_regions()
        word_regions = self.word_regions()
        grouping_methods = sorted(
            {
                str(region.metadata.get("line_grouping_method"))
                for region in line_regions
                if region.metadata.get("line_grouping_method")
            }
        )
        return {
            "raw_text": self.raw_text,
            "readable_text": self.metadata.get("readable_text", self.combined_text()),
            "word_count": self.metadata.get("ocr_word_count", len(word_regions)),
            "line_count": self.metadata.get("ocr_line_count", len(line_regions)),
            "line_grouping_method": grouping_methods[0] if len(grouping_methods) == 1 else grouping_methods,
            "reading_order_uncertain": any(
                bool(region.metadata.get("reading_order_uncertain"))
                for region in line_regions
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text_regions": [region.to_dict() for region in self.text_regions],
            "line_regions": [region.to_dict() for region in self.line_regions()],
            "word_regions": [region.to_dict() for region in self.word_regions()],
            "image_regions": [region.to_dict() for region in self.image_regions],
            "raw_text": self.raw_text,
            "readable_text": self.metadata.get("readable_text"),
            "text_summary": self.text_summary(),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DocumentResult:
    """Canonical document-level result for OCR and downstream processing."""

    document_id: int | None
    source_path: Path
    pages: list[PageResult] = field(default_factory=list)
    language_hint: str | None = None
    detected_languages: list[str] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        return "\n\n".join(page.combined_text() for page in self.pages if page.combined_text())

    def text_summary(self) -> dict[str, Any]:
        pages = [page.text_summary() for page in self.pages]
        return {
            "page_count": len(self.pages),
            "word_count": sum(int(page.get("word_count") or 0) for page in pages),
            "line_count": sum(int(page.get("line_count") or 0) for page in pages),
            "readable_text": "\n\n".join(
                str(page.get("readable_text") or "")
                for page in pages
                if page.get("readable_text")
            ),
            "reading_order_uncertain": any(
                bool(page.get("reading_order_uncertain")) for page in pages
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": str(self.source_path),
            "pages": [page.to_dict() for page in self.pages],
            "text_summary": self.text_summary(),
            "language_hint": self.language_hint,
            "detected_languages": list(self.detected_languages),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }
