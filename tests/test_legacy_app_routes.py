from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

from PIL import Image


def _png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class FakeOCRResult:
    engine = "fake"
    metadata = {"source": "route-test"}

    def to_dict(self):
        return {"engine": self.engine, "metadata": self.metadata}


class FakeProcessingResult:
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
        return {"cleaned_text": self.cleaned_text}


class FakeProcessor:
    def process_document(self, image_path: str, document_id: int | None = None):
        return FakeProcessingResult()


class FakePage:
    def to_dict(self):
        return {"pages": []}


class FakeReconstructor:
    def reconstruct_page(self, processing_result, image_path: str):
        return FakePage()


def _import_legacy_app(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("FLASK_ENV", "testing")
    monkeypatch.setenv("MARTIAL_ARTS_OCR_DATA_DIR", str(data_dir))
    monkeypatch.delenv("USE_YOLO_FIGURE", raising=False)

    for module_name in (
        "app",
        "database",
        "models",
        "config",
        "martial_arts_ocr",
        "martial_arts_ocr.app",
        "martial_arts_ocr.app.flask_app",
        "martial_arts_ocr.config",
        "martial_arts_ocr.db.database",
        "martial_arts_ocr.db.models",
        "martial_arts_ocr.pipeline",
        "martial_arts_ocr.pipeline.orchestrator",
        "martial_arts_ocr.pipeline.result_models",
    ):
        sys.modules.pop(module_name, None)

    legacy_app = importlib.import_module("app")
    legacy_app.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return legacy_app, data_dir


def test_legacy_app_health_index_and_gallery(monkeypatch, tmp_path):
    legacy_app, _data_dir = _import_legacy_app(monkeypatch, tmp_path)
    client = legacy_app.app.test_client()

    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/gallery").status_code == 200


def test_upload_process_view_and_download_routes_use_data_dir(monkeypatch, tmp_path):
    legacy_app, data_dir = _import_legacy_app(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy_app, "_kickoff_processing_async", lambda _document_id: None)
    client = legacy_app.app.test_client()

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(_png_bytes()), "scan.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    document_id = payload["documentId"]

    uploaded_files = list((data_dir / "uploads").glob("scan_*.png"))
    assert uploaded_files, "upload route should save files under data/uploads"

    from martial_arts_ocr.pipeline import WorkflowOrchestrator

    legacy_app.workflow_orchestrator = WorkflowOrchestrator(
        processor=FakeProcessor(),
        page_reconstructor=FakeReconstructor(),
        session_factory=legacy_app.get_db_session,
        processed_path_factory=lambda name: data_dir / "processed" / name,
        document_model=legacy_app.Document,
        page_model=legacy_app.Page,
        db_processing_result_model=legacy_app.ProcessingResult,
    )

    process_response = client.get(f"/process/{document_id}", follow_redirects=True)
    assert process_response.status_code == 200
    assert (data_dir / "processed" / f"doc_{document_id}" / "page_1.html").exists()

    view_response = client.get(f"/view/{document_id}")
    assert view_response.status_code == 200

    download_response = client.get(f"/download/{document_id}")
    assert download_response.status_code == 200
    assert b"Sample OCR text" in download_response.data


def test_optional_processor_failures_return_json_errors(monkeypatch, tmp_path):
    legacy_app, _data_dir = _import_legacy_app(monkeypatch, tmp_path)
    missing = legacy_app.UnavailableProcessor("JapaneseProcessor", RuntimeError("missing optional dep"))
    monkeypatch.setattr(legacy_app, "japanese_processor", missing)
    monkeypatch.setattr(
        legacy_app,
        "ocr_processor",
        legacy_app.UnavailableProcessor("OCRProcessor", RuntimeError("missing OCR engine")),
    )
    client = legacy_app.app.test_client()

    engine_response = client.get("/api/engines/status")
    assert engine_response.status_code == 200
    assert engine_response.get_json()["available"] is False

    romanize_response = client.post("/api/romanize", json={"text": "日本語"})
    assert romanize_response.status_code == 503
    assert romanize_response.get_json()["error"] == "Romanization failed"

    translate_response = client.post("/api/translate", json={"text": "日本語"})
    assert translate_response.status_code == 503
    assert translate_response.get_json()["error"] == "Translation failed"
