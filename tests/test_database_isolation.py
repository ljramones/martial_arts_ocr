from datetime import datetime
from pathlib import Path

from martial_arts_ocr.db.context import DatabaseConfig, DatabaseContext
from martial_arts_ocr.db.models import Document
from martial_arts_ocr.pipeline.document_models import DocumentResult, PageResult


def _create_document(context: DatabaseContext, filename: str) -> int:
    with context.get_db_session() as session:
        doc = Document(
            filename=filename,
            original_filename=filename,
            file_size=1,
            upload_date=datetime.now(),
            status="uploaded",
        )
        session.add(doc)
        session.commit()
        return int(doc.id)


def _count_documents(context: DatabaseContext) -> int:
    with context.get_db_session() as session:
        return session.query(Document).count()


def test_database_contexts_do_not_share_state(tmp_path):
    first = DatabaseContext(DatabaseConfig(database_path=tmp_path / "one.db"))
    second = DatabaseContext(DatabaseConfig(database_path=tmp_path / "two.db"))
    first.init_db()
    second.init_db()

    _create_document(first, "first.png")

    assert _count_documents(first) == 1
    assert _count_documents(second) == 0


def test_flask_app_factory_accepts_isolated_database_and_data_dir(tmp_path):
    from martial_arts_ocr.app import flask_app
    from martial_arts_ocr.config import get_upload_path
    from martial_arts_ocr.db import database

    previous_url = database.get_database_context().config.url
    data_dir = tmp_path / "app-data"
    db_path = data_dir / "isolated.db"

    try:
        app = flask_app.create_app(
            {
                "TESTING": True,
                "DATA_DIR": data_dir,
                "DATABASE_PATH": db_path,
            }
        )

        assert app.config["DATA_DIR"] == str(data_dir)
        assert app.config["DATABASE_URL"] == f"sqlite:///{db_path}"
        deps = app.extensions["martial_arts_ocr"]
        assert deps.upload_dir / "scan.png" == data_dir / "runtime" / "uploads" / "scan.png"

        _create_document(deps.db_context, "factory.png")
        assert _count_documents(deps.db_context) == 1
        assert db_path.exists()
    finally:
        database.configure_database(database_url=previous_url)
        database.init_db()
        flask_app.workflow_orchestrator = None


def test_legacy_database_import_exposes_reconfiguration_api():
    import database

    assert database.init_db
    assert database.configure_database
    assert database.get_database_context


def test_workflow_orchestrator_accepts_database_context(tmp_path):
    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    class FakeProcessor:
        def process_to_document_result(self, image_path, document_id=None):
            return DocumentResult(
                document_id=document_id,
                source_path=Path(image_path),
                pages=[PageResult(page_number=1, raw_text="isolated text", confidence=0.93)],
                confidence=0.93,
                metadata={"ocr_engine": "fake"},
            )

    context = DatabaseContext(DatabaseConfig(database_path=tmp_path / "pipeline.db"))
    context.init_db()
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"fake image bytes")
    document_id = _create_document(context, image_path.name)

    result = WorkflowOrchestrator(
        processor=FakeProcessor(),
        db_context=context,
        processed_path_factory=lambda name: tmp_path / "processed" / name,
    ).process_document(PipelineRequest(document_id=document_id, image_path=image_path))

    assert result.success is True
    with context.get_db_session() as session:
        doc = session.get(Document, document_id)
        assert doc.status == "completed"
        assert doc.ocr_engine == "fake"
