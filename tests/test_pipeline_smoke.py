from pathlib import Path

import pytest

from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator


class FakeProcessor:
    def process_document(self, image_path: str, document_id: int | None = None):
        return {"image_path": image_path, "document_id": document_id, "text": "ok"}


def test_pipeline_request_model(tmp_path):
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"not a real image")

    request = PipelineRequest(image_path=image_path, document_id=7, language_hint="en")

    assert request.image_path == image_path
    assert request.document_id == 7
    assert request.preserve_layout is True
    assert request.extract_images is True


def test_orchestrator_with_injected_processor(tmp_path):
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"not a real image")

    orchestrator = WorkflowOrchestrator(processor=FakeProcessor())
    result = orchestrator.process_document(PipelineRequest(image_path=image_path, document_id=3))

    assert result.status == "completed"
    assert result.payload["document_id"] == 3
    assert Path(result.payload["image_path"]) == image_path


def test_orchestrator_rejects_missing_file():
    orchestrator = WorkflowOrchestrator(processor=FakeProcessor())

    with pytest.raises(FileNotFoundError):
        orchestrator.process_document(PipelineRequest(image_path=Path("missing.png")))

