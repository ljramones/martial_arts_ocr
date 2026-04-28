from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path


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


class FakePage:
    def to_dict(self):
        return {"pages": [], "source": "fake"}


class FakeReconstructor:
    def reconstruct_page(self, processing_result, image_path: str):
        return FakePage()


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
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(runtime, image_path.name)

    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator
    from martial_arts_ocr.pipeline.document_models import DocumentResult

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
