from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from martial_arts_ocr.pipeline.document_models import BoundingBox, DocumentResult, ImageRegion, PageResult, TextRegion
from martial_arts_ocr.pipeline.extraction_service import ExtractionService, ExtractionServiceOptions
from utils.image.regions.core_types import ImageRegion as UtilityImageRegion


class FakeOCRResult:
    engine = "fake"
    metadata = {"source": "test"}

    def to_dict(self):
        return {"engine": self.engine, "metadata": self.metadata}


class FakeProcessingResult:
    document_id = None
    page_id = None
    ocr_results = []
    best_ocr_result = FakeOCRResult()
    raw_text = "Sample OCR text"
    cleaned_text = "Sample OCR text"
    text_regions = []
    image_regions = []
    extracted_images = []
    japanese_result = None
    language_segments = [{"text": "Sample OCR text", "language": "en"}]
    text_statistics = {"character_count": 15}
    overall_confidence = 0.95
    quality_score = 0.9
    processing_time = 0.01
    html_content = "<html>Sample OCR text</html>"
    markdown_content = "Sample OCR text"
    processing_metadata = {"image_dimensions": {"width": 8, "height": 8}}

    def to_dict(self):
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "overall_confidence": self.overall_confidence,
            "processing_metadata": self.processing_metadata,
        }


class FakeProcessor:
    def process_document(self, image_path: str, document_id: int | None = None):
        result = FakeProcessingResult()
        result.document_id = document_id
        return result


class FakeCanonicalProcessor:
    process_document_called = False

    def process_to_document_result(self, image_path, document_id: int | None = None):
        return DocumentResult(
            document_id=document_id,
            source_path=Path(image_path),
            pages=[PageResult(page_number=1, width=8, height=8, raw_text="Canonical OCR text", confidence=0.97)],
            detected_languages=["en"],
            confidence=0.97,
            metadata={"ocr_engine": "canonical", "processing_time": 0.02, "quality_score": 0.91},
        )

    def process_document(self, image_path: str, document_id: int | None = None):
        self.process_document_called = True
        raise AssertionError("orchestrator should prefer process_to_document_result")


class FakeReadableTextProcessor:
    def process_to_document_result(self, image_path, document_id: int | None = None):
        return DocumentResult(
            document_id=document_id,
            source_path=Path(image_path),
            pages=[
                PageResult(
                    page_number=1,
                    raw_text="raw OCR text",
                    text_regions=[
                        TextRegion(
                            region_id="line_1",
                            text="readable OCR text",
                            bbox=BoundingBox(x=1, y=2, width=30, height=10),
                            metadata={
                                "source": "ocr_normalization",
                                "ocr_level": "line",
                                "line_grouping_method": "adaptive_center_overlap_v1",
                                "reading_order_uncertain": False,
                            },
                        ),
                        TextRegion(
                            region_id="word_1",
                            text="readable",
                            bbox=BoundingBox(x=1, y=2, width=20, height=10),
                            metadata={"source": "ocr_engine", "ocr_level": "word"},
                        ),
                    ],
                    metadata={
                        "readable_text": "readable OCR text",
                        "ocr_word_count": 1,
                        "ocr_line_count": 1,
                    },
                )
            ],
            metadata={"ocr_engine": "canonical"},
        )


class FakeReadableTextWithLegacyHtmlProcessor(FakeReadableTextProcessor):
    def process_to_document_result(self, image_path, document_id: int | None = None):
        result = super().process_to_document_result(image_path, document_id=document_id)
        return DocumentResult(
            document_id=result.document_id,
            source_path=result.source_path,
            pages=result.pages,
            detected_languages=result.detected_languages,
            confidence=result.confidence,
            metadata={
                **result.metadata,
                "legacy": {"html_content": "<html>legacy WORD_ONLY_DEBUG</html>"},
            },
        )


class FakePage:
    def to_dict(self):
        return {"pages": [], "source": "fake"}


class FakeReconstructor:
    last_processing_result = None

    def reconstruct_page(self, processing_result, image_path: str):
        self.last_processing_result = processing_result
        return FakePage()


class RecordingExtractionService:
    def __init__(self):
        self.calls = []

    def enrich_document_result(self, document_result, *, output_dir):
        self.calls.append((document_result, output_dir))
        return DocumentResult(
            document_id=document_result.document_id,
            source_path=document_result.source_path,
            pages=[
                PageResult(
                    page_number=1,
                    width=document_result.pages[0].width if document_result.pages else None,
                    height=document_result.pages[0].height if document_result.pages else None,
                    raw_text=document_result.combined_text(),
                    image_regions=[
                        ImageRegion(
                            region_id="recorded_image",
                            bbox=BoundingBox(x=1, y=2, width=3, height=4),
                            region_type="diagram",
                        )
                    ],
                    confidence=document_result.confidence,
                    metadata={"image_extraction": {"enabled": True}},
                )
            ],
            detected_languages=document_result.detected_languages,
            confidence=document_result.confidence,
            metadata={**document_result.metadata, "image_extraction": {"enabled": True, "status": "completed"}},
        )


class FakeLayoutAnalyzer:
    def detect_image_regions_with_diagnostics(self, image, ocr_text_boxes=None):
        return {
            "accepted_regions": [
                UtilityImageRegion(x=20, y=24, width=80, height=70, region_type="diagram", confidence=0.91)
            ],
            "accepted": [],
            "rejected": [],
            "consolidation": [],
        }


def _write_synthetic_image(path: Path) -> None:
    image = np.full((180, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 24), (100, 94), (0, 0, 0), 3)
    cv2.imwrite(str(path), image)


def _fresh_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASK_ENV", "testing")
    monkeypatch.setenv("MARTIAL_ARTS_OCR_DATA_DIR", str(tmp_path / "data"))
    for module_name in (
        "app",
        "database",
        "models",
        "config",
        "martial_arts_ocr.app.flask_app",
        "martial_arts_ocr.config",
        "martial_arts_ocr.db.database",
        "martial_arts_ocr.db.models",
        "martial_arts_ocr.pipeline",
        "martial_arts_ocr.pipeline.orchestrator",
        "martial_arts_ocr.pipeline.result_models",
    ):
        sys.modules.pop(module_name, None)
    return importlib.import_module("app")


def _create_document(runtime, filename: str) -> int:
    with runtime.get_db_session() as session:
        doc = runtime.Document(
            filename=filename,
            original_filename=filename,
            file_size=1,
            upload_date=datetime.now(),
            status="uploaded",
        )
        session.add(doc)
        session.commit()
        return int(doc.id)


def test_orchestrator_success_writes_db_and_artifacts(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "canonical_scan.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(
        processor=FakeProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    )
    result = orchestrator.process_document(
        PipelineRequest(document_id=document_id, image_path=image_path)
    )

    assert result.success is True
    assert result.output_dir == tmp_path / "data" / "processed" / f"doc_{document_id}"
    assert isinstance(result.payload, DocumentResult)
    assert result.payload.combined_text() == "Sample OCR text"
    assert isinstance(orchestrator._page_reconstructor.last_processing_result, DocumentResult)
    assert result.html_path.exists()
    assert result.json_path.exists()
    assert result.text_path.exists()
    assert result.text_path.read_text(encoding="utf-8") == "Sample OCR text"

    artifact_data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert artifact_data["document_id"] == document_id
    assert artifact_data["pages"][0]["raw_text"] == "Sample OCR text"
    assert artifact_data["cleaned_text"] == "Sample OCR text"
    assert artifact_data["legacy_processing_result"]["cleaned_text"] == "Sample OCR text"

    with runtime.get_db_session() as session:
        doc = session.get(runtime.Document, document_id)
        page = session.query(runtime.Page).filter_by(document_id=document_id).first()
        db_result = session.query(runtime.ProcessingResult).filter_by(document_id=document_id).first()

        assert doc.status == "completed"
        assert doc.ocr_engine == "fake"
        assert page.ocr_confidence == 0.95
        assert db_result.cleaned_text == "Sample OCR text"


def test_orchestrator_prefers_canonical_processor_and_persists_document_result(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)
    processor = FakeCanonicalProcessor()

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(
        processor=processor,
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    )

    result = orchestrator.process_document(
        PipelineRequest(document_id=document_id, image_path=image_path)
    )

    assert result.success is True
    assert processor.process_document_called is False
    assert result.payload.combined_text() == "Canonical OCR text"
    assert result.metadata["ocr_engine"] == "canonical"
    assert result.text_path.read_text(encoding="utf-8") == "Canonical OCR text"

    with runtime.get_db_session() as session:
        doc = session.get(runtime.Document, document_id)
        page = session.query(runtime.Page).filter_by(document_id=document_id).first()
        db_result = session.query(runtime.ProcessingResult).filter_by(document_id=document_id).first()

        assert doc.ocr_engine == "canonical"
        assert page.image_width == 8
        assert page.ocr_confidence == 0.97
        assert db_result.cleaned_text == "Canonical OCR text"
        assert db_result.ocr_engine_used == "canonical"


def test_orchestrator_missing_image_returns_failed_result(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    missing_path = tmp_path / "missing.png"
    document_id = _create_document(runtime, missing_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeProcessor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    ).process_document(
        PipelineRequest(document_id=document_id, image_path=missing_path)
    )

    assert result.success is False
    assert result.status == "failed"
    assert "does not exist" in result.error

    with runtime.get_db_session() as session:
        doc = session.get(runtime.Document, document_id)
        assert doc.status == "failed"
        assert "does not exist" in doc.error_message


def test_orchestrator_uses_extraction_service_when_injected(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan_extraction_injected.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)
    extraction_service = RecordingExtractionService()

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeCanonicalProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
        extraction_service=extraction_service,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    assert len(extraction_service.calls) == 1
    assert extraction_service.calls[0][1] == tmp_path / "data" / "processed" / f"doc_{document_id}"
    assert result.payload.metadata["image_extraction"]["status"] == "completed"


def test_orchestrator_enabled_extraction_writes_image_regions_to_data_json(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan_extraction_enabled.png"

    _write_synthetic_image(image_path)
    document_id = _create_document(runtime, image_path.name)
    extraction_service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True),
        layout_analyzer_factory=lambda: FakeLayoutAnalyzer(),
    )

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeCanonicalProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
        extraction_service=extraction_service,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    artifact_data = json.loads(result.json_path.read_text(encoding="utf-8"))
    image_regions = artifact_data["pages"][0]["image_regions"]
    assert len(image_regions) == 1
    assert image_regions[0]["image_path"]
    assert (result.output_dir / "image_regions" / "image_region_001.png").exists()


def test_orchestrator_without_extraction_service_keeps_image_regions_empty(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan_extraction_disabled.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeCanonicalProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    artifact_data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert artifact_data["pages"][0]["image_regions"] == []


def test_orchestrator_data_json_exposes_readable_text_summary(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan_readable_text.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeReadableTextProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    artifact_data = json.loads(result.json_path.read_text(encoding="utf-8"))
    page = artifact_data["pages"][0]
    assert artifact_data["text_summary"]["readable_text"] == "readable OCR text"
    assert page["text_summary"]["word_count"] == 1
    assert page["text_summary"]["line_count"] == 1
    assert page["line_regions"][0]["text"] == "readable OCR text"
    assert page["word_regions"][0]["text"] == "readable"
    assert result.text_path.read_text(encoding="utf-8") == "readable OCR text"


def test_orchestrator_page_html_prefers_canonical_reconstruction(monkeypatch, tmp_path):
    runtime = _fresh_runtime(monkeypatch, tmp_path)
    image_path = tmp_path / "scan_readable_text_with_legacy_html.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    result = WorkflowOrchestrator(
        processor=FakeReadableTextWithLegacyHtmlProcessor(),
        session_factory=runtime.get_db_session,
        processed_path_factory=lambda name: tmp_path / "data" / "processed" / name,
        document_model=runtime.Document,
        page_model=runtime.Page,
        db_processing_result_model=runtime.ProcessingResult,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    html = result.html_path.read_text(encoding="utf-8")
    assert "readable OCR text" in html
    assert "legacy WORD_ONLY_DEBUG" not in html
