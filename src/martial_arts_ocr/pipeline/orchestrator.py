"""Canonical document processing orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from martial_arts_ocr.pipeline.adapters import document_result_from_ocr_output
from martial_arts_ocr.pipeline.document_models import DocumentResult
from martial_arts_ocr.pipeline.result_models import PipelineRequest, PipelineResult

logger = logging.getLogger(__name__)


class _DocumentProcessor(Protocol):
    def process_document(self, image_path: str, document_id: int | None = None): ...


class _PageReconstructor(Protocol):
    def reconstruct_page(self, processing_result: Any, image_path: str): ...


class WorkflowOrchestrator:
    """Coordinate document processing, persistence, and artifact writing."""

    def __init__(
        self,
        processor: _DocumentProcessor | None = None,
        page_reconstructor: _PageReconstructor | None = None,
        session_factory: Callable | None = None,
        processed_path_factory: Callable[[str], Path] | None = None,
        document_model: Any | None = None,
        page_model: Any | None = None,
        db_processing_result_model: Any | None = None,
        persist: bool = True,
    ) -> None:
        if session_factory is None:
            from martial_arts_ocr.db.database import get_db_session

            session_factory = get_db_session
        if processed_path_factory is None:
            from martial_arts_ocr.config import get_processed_path

            processed_path_factory = get_processed_path
        if document_model is None or page_model is None or db_processing_result_model is None:
            from martial_arts_ocr.db.models import Document, Page, ProcessingResult as DBProcessingResult

            document_model = document_model or Document
            page_model = page_model or Page
            db_processing_result_model = db_processing_result_model or DBProcessingResult
        self._processor = processor
        self._page_reconstructor = page_reconstructor
        self._session_factory = session_factory
        self._processed_path_factory = processed_path_factory
        self._document_model = document_model
        self._page_model = page_model
        self._db_processing_result_model = db_processing_result_model
        self._persist = persist

    def process_document(self, request: PipelineRequest) -> PipelineResult:
        image_path = request.image_path.expanduser()
        output_dir = self._processed_path_factory(f"doc_{request.document_id}")

        if not image_path.exists():
            error = f"Input image does not exist: {image_path}"
            if self._persist:
                self._mark_document_failed(request.document_id, error)
            return PipelineResult(
                document_id=request.document_id,
                status="failed",
                output_dir=output_dir,
                error=error,
            )

        if self._persist:
            self._mark_document_processing(request.document_id)

        try:
            processor = self._processor or self._build_default_processor()
            processing_result = processor.process_document(str(image_path), document_id=request.document_id)
            document_result = document_result_from_ocr_output(
                processing_result,
                document_id=request.document_id,
                source_path=image_path,
                language_hint=request.language_hint,
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            page_id = self._persist_processing_result(request, processing_result) if self._persist else None
            paths = self._write_artifacts(output_dir, request, processing_result, document_result)

            metadata = {
                "ocr_engine": self._best_engine(processing_result),
                "confidence": document_result.confidence,
                "has_japanese": getattr(processing_result, "japanese_result", None) is not None,
                "page_id": page_id,
                "detected_languages": document_result.detected_languages,
            }
            if self._persist:
                self._mark_document_completed(request.document_id, metadata["ocr_engine"])

            return PipelineResult(
                document_id=request.document_id,
                status="completed",
                output_dir=output_dir,
                html_path=paths.get("html_path"),
                json_path=paths.get("json_path"),
                text_path=paths.get("text_path"),
                metadata=metadata,
                payload=document_result,
            )

        except Exception as exc:
            logger.error("Document processing failed for %s: %s", request.document_id, exc, exc_info=True)
            if self._persist:
                self._mark_document_failed(request.document_id, str(exc))
            return PipelineResult(
                document_id=request.document_id,
                status="failed",
                output_dir=output_dir,
                error=str(exc),
            )

    @staticmethod
    def _build_default_processor() -> _DocumentProcessor:
        from martial_arts_ocr.ocr.processor import OCRProcessor

        return OCRProcessor()

    @staticmethod
    def _build_default_reconstructor() -> _PageReconstructor:
        from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor

        return PageReconstructor()

    def _persist_processing_result(self, request: PipelineRequest, processing_result: Any) -> int:
        with self._session_factory() as session:
            page = self._page_model(
                document_id=request.document_id,
                page_number=1,
                image_path=str(request.image_path),
                image_width=self._image_dimension(processing_result, "width"),
                image_height=self._image_dimension(processing_result, "height"),
                processing_time=getattr(processing_result, "processing_time", 0.0),
                ocr_confidence=getattr(processing_result, "overall_confidence", None),
                text_regions=self._regions_to_dicts(getattr(processing_result, "text_regions", [])),
                image_regions=self._regions_to_dicts(getattr(processing_result, "image_regions", [])),
            )
            session.add(page)
            session.flush()

            japanese_result = getattr(processing_result, "japanese_result", None)
            best_ocr_result = getattr(processing_result, "best_ocr_result", None)
            db_result = self._db_processing_result_model(
                document_id=request.document_id,
                page_id=page.id,
                ocr_engine_used=self._best_engine(processing_result),
                processing_time=getattr(processing_result, "processing_time", 0.0),
                raw_ocr_text=getattr(processing_result, "raw_text", None),
                cleaned_text=getattr(processing_result, "cleaned_text", None),
                ocr_confidence=getattr(processing_result, "overall_confidence", None),
                ocr_metadata=getattr(best_ocr_result, "metadata", None),
                has_japanese=japanese_result is not None,
                japanese_segments=[seg.to_dict() for seg in japanese_result.segments] if japanese_result else None,
                language_analysis=japanese_result.language_analysis if japanese_result else None,
                martial_arts_terms=japanese_result.martial_arts_terms if japanese_result else None,
                overall_romaji=japanese_result.overall_romaji if japanese_result else None,
                overall_translation=japanese_result.overall_translation if japanese_result else None,
                japanese_confidence=japanese_result.confidence_score if japanese_result else None,
                japanese_metadata=japanese_result.processing_metadata if japanese_result else None,
                html_content=getattr(processing_result, "html_content", None),
                markdown_content=getattr(processing_result, "markdown_content", None),
                extracted_images=getattr(processing_result, "extracted_images", None),
                text_statistics=getattr(processing_result, "text_statistics", None),
                quality_score=getattr(processing_result, "quality_score", None),
                confidence_breakdown={
                    "ocr_confidence": getattr(processing_result, "overall_confidence", None),
                    "quality_score": getattr(processing_result, "quality_score", None),
                    "japanese_confidence": japanese_result.confidence_score if japanese_result else None,
                },
            )
            session.add(db_result)
            session.commit()
            return int(page.id)

    def _write_artifacts(
        self,
        output_dir: Path,
        request: PipelineRequest,
        processing_result: Any,
        document_result: DocumentResult,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}

        try:
            reconstructor = self._page_reconstructor or self._build_default_reconstructor()
            reconstructed_page = reconstructor.reconstruct_page(processing_result, str(request.image_path))
            page_data_path = output_dir / "page_data.json"
            page_data_path.write_text(
                json.dumps(reconstructed_page.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["page_data_path"] = page_data_path
        except Exception as exc:
            logger.warning("Page reconstruction failed for doc %s: %s", request.document_id, exc)

        legacy_data = self._processing_result_to_dict(processing_result)
        combined_text = document_result.combined_text()
        json_data = document_result.to_dict()
        json_data["processing_date"] = datetime.now().isoformat()
        json_data["legacy_processing_result"] = legacy_data
        json_data["raw_text"] = legacy_data.get("raw_text", combined_text)
        json_data["cleaned_text"] = legacy_data.get("cleaned_text", combined_text)
        json_data["text"] = combined_text
        if "html_content" in legacy_data:
            json_data["html_content"] = legacy_data["html_content"]
        if "overall_confidence" in legacy_data:
            json_data["overall_confidence"] = legacy_data["overall_confidence"]
        json_path = output_dir / "data.json"
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json_path"] = json_path

        html = getattr(processing_result, "html_content", None)
        if html:
            html_path = output_dir / "page_1.html"
            html_path.write_text(html, encoding="utf-8")
            paths["html_path"] = html_path

        text = combined_text
        if text:
            text_path = output_dir / "text.txt"
            text_path.write_text(text, encoding="utf-8")
            paths["text_path"] = text_path

        return paths

    def _mark_document_processing(self, document_id: int) -> None:
        with self._session_factory() as session:
            doc = session.get(self._document_model, document_id)
            if doc:
                doc.status = "processing"
                session.commit()

    def _mark_document_completed(self, document_id: int, ocr_engine: str) -> None:
        with self._session_factory() as session:
            doc = session.get(self._document_model, document_id)
            if doc:
                doc.status = "completed"
                doc.processing_date = datetime.now()
                doc.ocr_engine = ocr_engine
                session.commit()

    def _mark_document_failed(self, document_id: int, error: str) -> None:
        with self._session_factory() as session:
            doc = session.get(self._document_model, document_id)
            if doc:
                doc.status = "failed"
                doc.error_message = error
                session.commit()

    @staticmethod
    def _regions_to_dicts(regions: list[Any]) -> list[dict[str, Any]]:
        out = []
        for region in regions or []:
            out.append(region.to_dict() if hasattr(region, "to_dict") else dict(region))
        return out

    @staticmethod
    def _image_dimension(processing_result: Any, key: str) -> Any:
        metadata = getattr(processing_result, "processing_metadata", {}) or {}
        return metadata.get("image_dimensions", {}).get(key)

    @staticmethod
    def _best_engine(processing_result: Any) -> str:
        best_ocr_result = getattr(processing_result, "best_ocr_result", None)
        return getattr(best_ocr_result, "engine", "unknown")

    @staticmethod
    def _processing_result_to_dict(processing_result: Any) -> dict[str, Any]:
        if hasattr(processing_result, "to_dict"):
            return processing_result.to_dict()
        if isinstance(processing_result, dict):
            return dict(processing_result)
        return {"repr": repr(processing_result)}
