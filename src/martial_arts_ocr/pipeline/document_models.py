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

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text_regions": [region.to_dict() for region in self.text_regions],
            "image_regions": [region.to_dict() for region in self.image_regions],
            "raw_text": self.raw_text,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": str(self.source_path),
            "pages": [page.to_dict() for page in self.pages],
            "language_hint": self.language_hint,
            "detected_languages": list(self.detected_languages),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }
