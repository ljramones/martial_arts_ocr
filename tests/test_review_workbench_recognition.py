from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from martial_arts_ocr.pipeline.document_models import BoundingBox, ImageRegion


def _write_image(path: Path, size=(180, 120)):
    Image.new("RGB", size, "white").save(path)


def _create_review_app(tmp_path):
    from martial_arts_ocr.app.flask_app import create_app

    data_dir = tmp_path / "data"
    scans = tmp_path / "scans"
    scans.mkdir()
    app = create_app(
        {
            "TESTING": True,
            "DATA_DIR": data_dir,
            "DATABASE_PATH": data_dir / "app.db",
            "REVIEW_ALLOWED_ROOTS": [tmp_path],
            "WTF_CSRF_ENABLED": False,
        }
    )
    return app, data_dir, scans


def test_recognition_endpoint_imports_detected_regions_without_ocr(tmp_path):
    app, data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page.png", size=(180, 120))
    calls = []

    class FakeRecognitionService:
        def enrich_document_result(self, document, *, output_dir):
            calls.append({"document": document, "output_dir": output_dir})
            page = replace(
                document.pages[0],
                image_regions=[
                    ImageRegion(
                        region_id="fake_001",
                        region_type="image",
                        bbox=BoundingBox(10, 12, 40, 50),
                        confidence=0.91,
                        metadata={
                            "detector": "fake_layout",
                            "mixed_region": False,
                            "needs_review": False,
                            "layout_fusion_applied": False,
                            "region_role": "figure",
                        },
                    ),
                    ImageRegion(
                        region_id="fake_002",
                        region_type="image",
                        bbox=BoundingBox(70, 20, 30, 35),
                        confidence=0.45,
                        metadata={
                            "detector": "fake_layout",
                            "mixed_region": True,
                            "needs_review": True,
                        },
                    ),
                ],
            )
            return replace(document, pages=[page])

    app.config["REVIEW_RECOGNITION_SERVICE"] = FakeRecognitionService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "recognize"})

    response = client.post("/api/review/projects/recognize/pages/page_001/recognize")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["detected_count"] == 2
    assert len(calls) == 1
    assert calls[0]["document"].metadata["ocr_executed"] is False
    page = payload["page"]
    assert page["regions"][0]["region_id"] == "det_001"
    assert page["regions"][0]["detected_type"] == "image"
    assert page["regions"][0]["detected_bbox"] == [10, 12, 40, 50]
    assert page["regions"][0]["effective_bbox"] == [10, 12, 40, 50]
    assert page["regions"][0]["metadata"]["detector"] == "fake_layout"
    assert page["regions"][0]["metadata"]["region_role"] == "figure"
    assert page["regions"][1]["detected_type"] == "unknown_needs_review"
    assert page["regions"][1]["needs_review"] is True
    assert page["recognition"]["rerun_behavior"] == "replaced_unreviewed_machine_detection_regions"

    saved = json.loads(
        (data_dir / "runtime" / "review_projects" / "recognize" / "project_state.json").read_text(encoding="utf-8")
    )
    assert saved["pages"][0]["regions"][0]["source"] == "machine_detection"


def test_recognition_rerun_preserves_manual_and_reviewed_regions(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page.png", size=(180, 120))
    run = {"count": 0}

    class FakeRecognitionService:
        def enrich_document_result(self, document, *, output_dir):
            run["count"] += 1
            if run["count"] == 1:
                image_regions = [
                    ImageRegion(
                        region_id="first_001",
                        region_type="image",
                        bbox=BoundingBox(10, 10, 30, 30),
                        metadata={"detector": "first"},
                    ),
                    ImageRegion(
                        region_id="first_002",
                        region_type="image",
                        bbox=BoundingBox(60, 10, 20, 20),
                        metadata={"detector": "first"},
                    ),
                ]
            else:
                image_regions = [
                    ImageRegion(
                        region_id="second_001",
                        region_type="diagram",
                        bbox=BoundingBox(100, 30, 40, 40),
                        metadata={"detector": "second"},
                    )
                ]
            return replace(document, pages=[replace(document.pages[0], image_regions=image_regions)])

    app.config["REVIEW_RECOGNITION_SERVICE"] = FakeRecognitionService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "rerun"})
    client.post("/api/review/projects/rerun/pages/page_001/recognize")
    client.post(
        "/api/review/projects/rerun/pages/page_001/regions",
        json={"reviewed_type": "caption_label", "bbox": [5, 5, 20, 20]},
    )
    client.patch(
        "/api/review/projects/rerun/pages/page_001/regions/det_001",
        json={"reviewed_type": "diagram", "reviewed_bbox": [8, 8, 34, 34]},
    )

    rerun_response = client.post("/api/review/projects/rerun/pages/page_001/recognize")

    assert rerun_response.status_code == 200
    regions = {region["region_id"]: region for region in rerun_response.get_json()["page"]["regions"]}
    assert regions["det_001"]["reviewed_type"] == "diagram"
    assert regions["det_001"]["detected_bbox"] == [10, 10, 30, 30]
    assert regions["r_001"]["source"] == "manual"
    assert regions["det_002"]["detected_type"] == "diagram"
    assert regions["det_002"]["metadata"]["detector"] == "second"
    assert len(regions) == 3
