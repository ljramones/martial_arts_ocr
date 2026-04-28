"""Canonical document processing orchestration."""

from __future__ import annotations

import json
import logging
import html as html_lib
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
        db_context: Any | None = None,
        extraction_service: Any | None = None,
        persist: bool = True,
    ) -> None:
        if db_context is not None and session_factory is None:
            session_factory = db_context.get_db_session
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
        self._extraction_service = extraction_service
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
            document_result = self._process_to_document_result(processor, request, image_path)

            output_dir.mkdir(parents=True, exist_ok=True)
            document_result = self._enrich_with_extraction(document_result, output_dir=output_dir)
            page_id = self._persist_document_result(request, document_result) if self._persist else None
            paths = self._write_artifacts(output_dir, request, document_result)

            metadata = {
                "ocr_engine": self._document_ocr_engine(document_result),
                "confidence": document_result.confidence,
                "has_japanese": self._document_has_japanese(document_result),
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

    def _process_to_document_result(
        self,
        processor: _DocumentProcessor,
        request: PipelineRequest,
        image_path: Path,
    ) -> DocumentResult:
        if hasattr(processor, "process_to_document_result"):
            document_result = processor.process_to_document_result(
                image_path,
                document_id=request.document_id,
            )
            if isinstance(document_result, DocumentResult):
                return document_result
            return document_result_from_ocr_output(
                document_result,
                document_id=request.document_id,
                source_path=image_path,
                language_hint=request.language_hint,
            )

        legacy_result = processor.process_document(str(image_path), document_id=request.document_id)
        return document_result_from_ocr_output(
            legacy_result,
            document_id=request.document_id,
            source_path=image_path,
            language_hint=request.language_hint,
        )

    def _enrich_with_extraction(self, document_result: DocumentResult, *, output_dir: Path) -> DocumentResult:
        if self._extraction_service is None:
            return document_result
        return self._extraction_service.enrich_document_result(document_result, output_dir=output_dir)

    def _persist_document_result(self, request: PipelineRequest, document_result: DocumentResult) -> int:
        page_result = document_result.pages[0] if document_result.pages else None
        combined_text = document_result.combined_text()
        processing_time = float(document_result.metadata.get("processing_time") or 0.0)
        confidence = document_result.confidence or (page_result.confidence if page_result else None)

        with self._session_factory() as session:
            page = self._page_model(
                document_id=request.document_id,
                page_number=page_result.page_number if page_result else 1,
                image_path=str(request.image_path),
                image_width=page_result.width if page_result else None,
                image_height=page_result.height if page_result else None,
                processing_time=processing_time,
                ocr_confidence=confidence,
                text_regions=[region.to_dict() for region in page_result.text_regions] if page_result else [],
                image_regions=[region.to_dict() for region in page_result.image_regions] if page_result else [],
            )
            session.add(page)
            session.flush()

            legacy_data = self._document_legacy_data(document_result)
            html_content = self._legacy_field(document_result, "html_content")
            markdown_content = self._legacy_field(document_result, "markdown_content")
            db_result = self._db_processing_result_model(
                document_id=request.document_id,
                page_id=page.id,
                ocr_engine_used=self._document_ocr_engine(document_result),
                processing_time=processing_time,
                raw_ocr_text=page_result.raw_text if page_result else combined_text,
                cleaned_text=combined_text,
                ocr_confidence=confidence,
                ocr_metadata=document_result.metadata,
                has_japanese=self._document_has_japanese(document_result),
                japanese_segments=legacy_data.get("japanese_segments"),
                language_analysis=legacy_data.get("language_analysis"),
                martial_arts_terms=legacy_data.get("martial_arts_terms"),
                overall_romaji=legacy_data.get("overall_romaji"),
                overall_translation=legacy_data.get("overall_translation"),
                japanese_confidence=legacy_data.get("japanese_confidence"),
                japanese_metadata=legacy_data.get("japanese_metadata"),
                html_content=html_content,
                markdown_content=markdown_content,
                extracted_images=[region.to_dict() for region in page_result.image_regions] if page_result else [],
                text_statistics=document_result.metadata.get("text_statistics"),
                quality_score=document_result.metadata.get("quality_score"),
                confidence_breakdown={
                    "ocr_confidence": confidence,
                    "quality_score": document_result.metadata.get("quality_score"),
                    "japanese_confidence": legacy_data.get("japanese_confidence"),
                },
            )
            session.add(db_result)
            session.commit()
            return int(page.id)

    def _write_artifacts(
        self,
        output_dir: Path,
        request: PipelineRequest,
        document_result: DocumentResult,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}

        try:
            reconstructor = self._page_reconstructor or self._build_default_reconstructor()
            reconstructed_page = reconstructor.reconstruct_page(document_result, str(request.image_path))
            page_data_path = output_dir / "page_data.json"
            page_data_path.write_text(
                json.dumps(reconstructed_page.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["page_data_path"] = page_data_path
        except Exception as exc:
            logger.warning("Page reconstruction failed for doc %s: %s", request.document_id, exc)

        legacy_data = self._document_legacy_data(document_result)
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

        html = self._legacy_field(document_result, "html_content")
        if not html and "page_data_path" in paths:
            try:
                page_data = json.loads(paths["page_data_path"].read_text(encoding="utf-8"))
                html = page_data.get("html_content")
            except Exception:
                html = None
        if not html and combined_text:
            html = f"<html><body><pre>{html_lib.escape(combined_text)}</pre></body></html>"
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
    def _document_legacy_data(document_result: DocumentResult) -> dict[str, Any]:
        legacy = document_result.metadata.get("legacy", {})
        return legacy if isinstance(legacy, dict) else {}

    @classmethod
    def _legacy_field(cls, document_result: DocumentResult, key: str) -> Any:
        legacy = cls._document_legacy_data(document_result)
        return legacy.get(key)

    @classmethod
    def _document_ocr_engine(cls, document_result: DocumentResult) -> str:
        engine = document_result.metadata.get("ocr_engine")
        if engine and engine != "unknown":
            return str(engine)

        legacy = cls._document_legacy_data(document_result)
        best_ocr_result = legacy.get("best_ocr_result")
        if isinstance(best_ocr_result, dict) and best_ocr_result.get("engine"):
            return str(best_ocr_result["engine"])
        if legacy.get("ocr_engine_used"):
            return str(legacy["ocr_engine_used"])
        return "unknown"

    @staticmethod
    def _document_has_japanese(document_result: DocumentResult) -> bool:
        has_japanese = document_result.metadata.get("has_japanese")
        if has_japanese is not None:
            return bool(has_japanese)
        return any(language.lower().startswith("ja") for language in document_result.detected_languages)
