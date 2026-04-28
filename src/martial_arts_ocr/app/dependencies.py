"""Application-scoped dependency container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from martial_arts_ocr.db.context import DatabaseContext
from martial_arts_ocr.db.models import Document, Page, ProcessingResult
from martial_arts_ocr.pipeline import WorkflowOrchestrator


ProcessorFactory = Callable[[], object]


@dataclass
class AppDependencies:
    """Lazy runtime dependencies attached to a Flask app instance."""

    db_context: DatabaseContext
    data_dir: Path
    upload_dir: Path
    processed_dir: Path
    orchestrator: WorkflowOrchestrator | None = None
    ocr_processor_factory: ProcessorFactory | None = None
    content_extractor_factory: ProcessorFactory | None = None
    japanese_processor_factory: ProcessorFactory | None = None
    page_reconstructor_factory: ProcessorFactory | None = None
    ocr_processor: object | None = None
    content_extractor: object | None = None
    japanese_processor: object | None = None
    page_reconstructor: object | None = None

    def get_ocr_processor(self):
        if self.ocr_processor is None and self.ocr_processor_factory:
            self.ocr_processor = self.ocr_processor_factory()
        return self.ocr_processor

    def get_content_extractor(self):
        if self.content_extractor is None and self.content_extractor_factory:
            self.content_extractor = self.content_extractor_factory()
        return self.content_extractor

    def get_japanese_processor(self):
        if self.japanese_processor is None and self.japanese_processor_factory:
            self.japanese_processor = self.japanese_processor_factory()
        return self.japanese_processor

    def get_page_reconstructor(self):
        if self.page_reconstructor is None and self.page_reconstructor_factory:
            self.page_reconstructor = self.page_reconstructor_factory()
        return self.page_reconstructor

    def get_orchestrator(self):
        if self.orchestrator is None:
            self.orchestrator = WorkflowOrchestrator(
                processor=self.get_ocr_processor(),
                page_reconstructor=self.get_page_reconstructor(),
                db_context=self.db_context,
                processed_path_factory=lambda name: self.processed_dir / name,
                document_model=Document,
                page_model=Page,
                db_processing_result_model=ProcessingResult,
            )
        return self.orchestrator
