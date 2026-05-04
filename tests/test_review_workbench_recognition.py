from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from martial_arts_ocr.pipeline.document_models import BoundingBox, ImageRegion


def _write_image(path: Path, size=(180, 120)):
    Image.new("RGB", size, "white").save(path)


def _write_three_panel_row(path: Path) -> None:
    page = np.full((360, 760), 255, dtype=np.uint8)
    for x in (55, 305, 555):
        cv2.rectangle(page, (x + 8, 103), (x + 131, 236), 0, 4)
        cv2.circle(page, (x + 70, 170), 30, 0, 4)
        cv2.line(page, (x + 25, 220), (x + 115, 122), 0, 4)
        cv2.arrowedLine(page, (x + 105, 210), (x + 70, 170), 0, 3, tipLength=0.16)
    cv2.imwrite(str(path), page)


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
            return replace(
                document,
                pages=[page],
                metadata={
                    **document.metadata,
                    "image_extraction": {
                        "accepted_count": 2,
                        "raw_candidate_count": 4,
                        "raw_candidates": [
                            {
                                "bbox": (10, 12, 50, 62),
                                "x": 10,
                                "y": 12,
                                "width": 40,
                                "height": 50,
                                "region_type": "image",
                                "confidence": 0.91,
                                "metadata": {"detector": "fake_layout"},
                            },
                            {
                                "bbox": (70, 20, 100, 55),
                                "x": 70,
                                "y": 20,
                                "width": 30,
                                "height": 35,
                                "region_type": "image",
                                "confidence": 0.45,
                                "metadata": {"detector": "fake_layout"},
                            },
                        ],
                        "accepted": [
                            {
                                "bbox": (10, 12, 50, 62),
                                "x": 10,
                                "y": 12,
                                "width": 40,
                                "height": 50,
                                "region_type": "image",
                                "confidence": 0.91,
                                "metadata": {"detector": "fake_layout"},
                            }
                        ],
                        "rejected": [
                            {
                                "region": {
                                    "bbox": (120, 25, 170, 70),
                                    "x": 120,
                                    "y": 25,
                                    "width": 50,
                                    "height": 45,
                                    "region_type": "diagram",
                                },
                                "rejection_reason": "text_like_components",
                            }
                        ],
                        "consolidation": [
                            {
                                "reason": "contained_suppression",
                                "suppressed": {
                                    "bbox": (130, 30, 145, 45),
                                    "x": 130,
                                    "y": 30,
                                    "width": 15,
                                    "height": 15,
                                    "region_type": "diagram",
                                },
                                "kept": {
                                    "bbox": (120, 25, 170, 70),
                                    "x": 120,
                                    "y": 25,
                                    "width": 50,
                                    "height": 45,
                                    "region_type": "diagram",
                                },
                            }
                        ],
                        "detector_diagnostics": [
                            {
                                "detector": "contours",
                                "topk": 6,
                                "topk_suppressed": [
                                    {
                                        "bbox": (150, 80, 190, 115),
                                        "x": 150,
                                        "y": 80,
                                        "width": 40,
                                        "height": 35,
                                        "region_type": "diagram",
                                    }
                                ],
                            }
                        ],
                    },
                },
            )

    app.config["REVIEW_RECOGNITION_SERVICE"] = FakeRecognitionService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "recognize"})

    response = client.post("/api/review/projects/recognize/pages/page_001/recognize")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["detected_count"] == 2
    assert payload["recognition_diagnostics"]["raw_candidate_count"] == 4
    assert payload["recognition_diagnostics"]["accepted_count"] == 2
    assert payload["recognition_diagnostics"]["rejected_count"] == 1
    assert payload["recognition_diagnostics"]["suppressed_count"] == 2
    assert payload["recognition_diagnostics"]["imported_count"] == 2
    candidate = payload["recognition_diagnostics"]["candidates"][0]
    assert candidate["candidate_id"] == "cand_001"
    assert candidate["stage"] == "raw"
    assert candidate["reason"] == "detected"
    assert candidate["bbox"] == [10, 12, 40, 50]
    assert len(calls) == 1
    assert calls[0]["document"].metadata["ocr_executed"] is False
    page = payload["page"]
    assert page["orientation"]["effective_rotation_degrees"] == 0
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
    assert saved["pages"][0]["recognition_diagnostics"]["imported_count"] == 2


def test_recognition_reports_rejected_candidates_without_importing_them(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page.png", size=(180, 120))

    class EmptyRecognitionService:
        def enrich_document_result(self, document, *, output_dir):
            return replace(
                document,
                metadata={
                    **document.metadata,
                    "image_extraction": {
                        "rejected": [
                            {
                                "region": {"bbox": [20, 25, 70, 65]},
                                "rejection_reason": "text_like_components",
                            }
                        ]
                    },
                },
            )

    app.config["REVIEW_RECOGNITION_SERVICE"] = EmptyRecognitionService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "diagnostic"})

    response = client.post("/api/review/projects/diagnostic/pages/page_001/recognize")

    assert response.status_code == 200
    page = response.get_json()["page"]
    assert response.get_json()["detected_count"] == 0
    assert response.get_json()["rejected_count"] == 1
    assert page["regions"] == []


def test_recognition_runs_against_effective_oriented_page(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page.png", size=(180, 120))
    calls = []

    class FakeRecognitionService:
        def enrich_document_result(self, document, *, output_dir):
            with Image.open(document.source_path) as image:
                calls.append(
                    {
                        "source_size": image.size,
                        "page_size": (document.pages[0].width, document.pages[0].height),
                        "orientation_degrees": document.metadata["orientation_degrees"],
                    }
                )
            page = replace(
                document.pages[0],
                image_regions=[
                    ImageRegion(
                        region_id="oriented_001",
                        region_type="image",
                        bbox=BoundingBox(5, 6, 30, 40),
                        metadata={"detector": "fake_oriented"},
                    )
                ],
            )
            return replace(document, pages=[page])

    app.config["REVIEW_RECOGNITION_SERVICE"] = FakeRecognitionService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "oriented_recognition"})
    client.patch(
        "/api/review/projects/oriented_recognition/pages/page_001/orientation",
        json={"reviewed_rotation_degrees": 90},
    )

    response = client.post("/api/review/projects/oriented_recognition/pages/page_001/recognize")

    assert response.status_code == 200
    assert calls == [
        {
            "source_size": (120, 180),
            "page_size": (120, 180),
            "orientation_degrees": 90,
        }
    ]
    page = response.get_json()["page"]
    assert page["regions"][0]["detected_bbox"] == [5, 6, 30, 40]
    assert page["recognition"]["orientation_degrees"] == 90


def test_recognition_imports_multi_figure_row_proposals(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_three_panel_row(scans / "three_panels.png")
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "multirow"})

    response = client.post("/api/review/projects/multirow/pages/page_001/recognize")

    assert response.status_code == 200
    page = response.get_json()["page"]
    assert len(page["regions"]) == 3
    assert response.get_json()["detected_count"] == len(page["regions"])
    diagnostics = response.get_json()["recognition_diagnostics"]
    row_detector = [
        detector for detector in diagnostics["detector_diagnostics"]
        if detector["detector"] == "multi_figure_rows"
    ][0]
    assert row_detector["returned_count"] == 3
    assert diagnostics["imported_count"] == len(page["regions"])


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
    assert regions["r_001"]["source"] == "reviewer_manual"
    assert regions["det_002"]["detected_type"] == "diagram"
    assert regions["det_002"]["metadata"]["detector"] == "second"
    assert len(regions) == 3
