"""Pipeline request and result models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineRequest:
    """Input for a document processing run."""

    image_path: Path
    document_id: int | None = None
    language_hint: str | None = None
    preserve_layout: bool = True
    extract_images: bool = True


@dataclass(frozen=True)
class PipelineResult:
    """Normalized wrapper around the existing OCR processing result."""

    request: PipelineRequest
    status: str
    payload: Any | None = None

