def test_package_imports():
    import martial_arts_ocr
    from martial_arts_ocr.app import create_app
    from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

    assert martial_arts_ocr.__version__
    assert create_app
    assert PipelineRequest
    assert WorkflowOrchestrator


def test_flask_factory_smoke():
    from martial_arts_ocr.app import create_app

    app = create_app({"TESTING": True})
    assert app.testing is True

    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}


def test_compatibility_imports():
    from martial_arts_ocr.config import get_config
    from martial_arts_ocr.pipeline.result_models import PipelineResult

    assert get_config("testing").TESTING is True
    assert PipelineResult
