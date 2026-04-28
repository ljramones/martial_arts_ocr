from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from martial_arts_ocr.pipeline import PipelineResult


def _png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class FakeOrchestrator:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.requests = []

    def process_document(self, request):
        self.requests.append(request)
        output_dir = self.processed_dir / f"doc_{request.document_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "page_1.html"
        html_path.write_text("<html>ok</html>", encoding="utf-8")
        return PipelineResult(
            document_id=request.document_id,
            status="completed",
            output_dir=output_dir,
            html_path=html_path,
        )


def _create_app(tmp_path, name: str):
    from martial_arts_ocr.app.flask_app import create_app

    data_dir = tmp_path / name / "data"
    orchestrator = FakeOrchestrator(data_dir / "runtime" / "processed")
    app = create_app(
        {
            "TESTING": True,
            "DATA_DIR": data_dir,
            "DATABASE_PATH": data_dir / "app.db",
            "WTF_CSRF_ENABLED": False,
        },
        orchestrator=orchestrator,
    )
    return app, data_dir, orchestrator


def test_create_app_returns_isolated_instances(tmp_path):
    first_app, first_data_dir, first_orchestrator = _create_app(tmp_path, "first")
    second_app, second_data_dir, second_orchestrator = _create_app(tmp_path, "second")

    first_deps = first_app.extensions["martial_arts_ocr"]
    second_deps = second_app.extensions["martial_arts_ocr"]

    assert first_app is not second_app
    assert first_deps.db_context is not second_deps.db_context
    assert first_deps.get_orchestrator() is first_orchestrator
    assert second_deps.get_orchestrator() is second_orchestrator
    assert first_deps.upload_dir == first_data_dir / "runtime" / "uploads"
    assert second_deps.upload_dir == second_data_dir / "runtime" / "uploads"
    assert first_deps.extraction_service.options.enable_image_regions is False
    assert second_deps.extraction_service.options.enable_image_regions is False


def test_create_app_can_enable_extraction_service_for_review_mode(tmp_path):
    from martial_arts_ocr.app.flask_app import create_app

    data_dir = tmp_path / "review" / "data"
    app = create_app(
        {
            "TESTING": True,
            "DATA_DIR": data_dir,
            "DATABASE_PATH": data_dir / "app.db",
            "ENABLE_IMAGE_REGION_EXTRACTION": True,
        }
    )

    deps = app.extensions["martial_arts_ocr"]
    assert deps.extraction_service.options.enable_image_regions is True


def test_create_app_parses_string_extraction_flags(tmp_path):
    from martial_arts_ocr.app.flask_app import create_app

    data_dir = tmp_path / "string_flags" / "data"
    app = create_app(
        {
            "TESTING": True,
            "DATA_DIR": data_dir,
            "DATABASE_PATH": data_dir / "app.db",
            "ENABLE_IMAGE_REGION_EXTRACTION": "false",
            "IMAGE_REGION_EXTRACTION_SAVE_CROPS": "true",
        }
    )

    deps = app.extensions["martial_arts_ocr"]
    assert deps.extraction_service.options.enable_image_regions is False
    assert deps.extraction_service.options.save_crops is True


def test_upload_and_process_routes_do_not_share_state(monkeypatch, tmp_path):
    from martial_arts_ocr.app import flask_app
    from martial_arts_ocr.db.models import Document

    first_app, first_data_dir, first_orchestrator = _create_app(tmp_path, "first")
    second_app, second_data_dir, second_orchestrator = _create_app(tmp_path, "second")
    monkeypatch.setattr(flask_app, "_kickoff_processing_async", lambda _document_id: None)

    first_client = first_app.test_client()
    second_client = second_app.test_client()

    first_response = first_client.post(
        "/upload",
        data={"file": (io.BytesIO(_png_bytes()), "scan.png")},
        content_type="multipart/form-data",
    )
    second_response = second_client.post(
        "/upload",
        data={"file": (io.BytesIO(_png_bytes()), "scan.png")},
        content_type="multipart/form-data",
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_id = first_response.get_json()["documentId"]
    second_id = second_response.get_json()["documentId"]
    assert list((first_data_dir / "runtime" / "uploads").glob("scan_*.png"))
    assert list((second_data_dir / "runtime" / "uploads").glob("scan_*.png"))

    assert first_client.get(f"/process/{first_id}").status_code == 302
    assert second_client.get(f"/process/{second_id}").status_code == 302
    assert first_orchestrator.requests[0].document_id == first_id
    assert second_orchestrator.requests[0].document_id == second_id

    with first_app.extensions["martial_arts_ocr"].db_context.get_db_session() as session:
        assert session.query(Document).count() == 1
    with second_app.extensions["martial_arts_ocr"].db_context.get_db_session() as session:
        assert session.query(Document).count() == 1


def test_legacy_import_still_exposes_app_object():
    import app

    assert app.app
    assert app.app.test_client().get("/healthz").status_code == 200
