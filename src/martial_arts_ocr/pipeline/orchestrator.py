"""Canonical pipeline orchestration seam."""

from __future__ import annotations

from typing import Protocol

from martial_arts_ocr.pipeline.result_models import PipelineRequest, PipelineResult


class _DocumentProcessor(Protocol):
    def process_document(self, image_path: str, document_id: int | None = None): ...


class WorkflowOrchestrator:
    """Thin orchestration layer around the current OCR processor.

    The class is intentionally small for this stabilization pass. It gives Flask,
    scripts, and future batch tooling one API to call while the existing OCR
    internals remain in place.
    """

    def __init__(self, processor: _DocumentProcessor | None = None) -> None:
        self._processor = processor

    def process_document(self, request: PipelineRequest) -> PipelineResult:
        image_path = request.image_path.expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        processor = self._processor or self._build_default_processor()
        payload = processor.process_document(str(image_path), document_id=request.document_id)
        return PipelineResult(request=request, status="completed", payload=payload)

    @staticmethod
    def _build_default_processor() -> _DocumentProcessor:
        from processors.ocr_processor import OCRProcessor

        return OCRProcessor()

