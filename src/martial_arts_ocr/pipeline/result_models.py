"""Pipeline request and result models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineRequest:
    """Input for a document processing run."""

    document_id: int
    image_path: Path
    language_hint: str | None = None
    preserve_layout: bool = True
    extract_images: bool = True


@dataclass(frozen=True)
class PipelineResult:
    """Structured result for a document processing run."""

    document_id: int
    status: str
    output_dir: Path
    html_path: Path | None = None
    json_path: Path | None = None
    text_path: Path | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    payload: Any | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed" and self.error is None

    def to_legacy_dict(self) -> dict[str, Any]:
        metadata = self.metadata or {}
        return {
            "success": self.success,
            "error": self.error,
            "ocr_engine": metadata.get("ocr_engine", "unknown"),
            "confidence": metadata.get("confidence"),
            "has_japanese": metadata.get("has_japanese", False),
            "document_id": self.document_id,
            "output_dir": str(self.output_dir),
        }
