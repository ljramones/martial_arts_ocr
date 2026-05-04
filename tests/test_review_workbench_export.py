from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from martial_arts_ocr.review.region_ocr_service import RegionOCRResult, RegionOCRRoute


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


def _write_marker_image(path: Path, size=(90, 60)):
    image = Image.new("RGB", size, "white")
    for x in range(size[0]):
        for y in range(size[1]):
            if 10 <= x < 40 and 12 <= y < 35:
                image.putpixel((x, y), (30, 70, 180))
            elif 45 <= x < 75 and 20 <= y < 45:
                image.putpixel((x, y), (180, 70, 30))
    image.save(path)


def test_review_page_export_writes_json_markdown_text_and_crops(tmp_path):
    app, data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "page.png", size=(100, 80))

    class FakeRegionOCRService:
        def run(self, *, image_path, bbox, region_type, region_id):
            text_by_region = {
                "r_001": "Lower raw text",
                "r_002": "Upper raw text",
            }
            return RegionOCRResult(
                text=text_by_region.get(region_id, ""),
                cleaned_text=text_by_region.get(region_id, ""),
                confidence=0.8,
                route=RegionOCRRoute(language="eng", psm=6),
                status="ok",
            )

    app.config["REVIEW_REGION_OCR_SERVICE"] = FakeRegionOCRService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_route"})
    client.post(
        "/api/review/projects/export_route/pages/page_001/regions",
        json={"reviewed_type": "english_text", "reviewed_bbox": [5, 50, 40, 20]},
    )
    client.post(
        "/api/review/projects/export_route/pages/page_001/regions",
        json={"reviewed_type": "english_text", "reviewed_bbox": [5, 5, 40, 20]},
    )
    client.post(
        "/api/review/projects/export_route/pages/page_001/regions",
        json={"reviewed_type": "diagram", "reviewed_bbox": [10, 12, 30, 23]},
    )
    client.post(
        "/api/review/projects/export_route/pages/page_001/regions",
        json={"reviewed_type": "ignore", "reviewed_bbox": [0, 0, 15, 15]},
    )
    client.post("/api/review/projects/export_route/pages/page_001/regions/r_001/ocr")
    client.post("/api/review/projects/export_route/pages/page_001/regions/r_002/ocr")
    client.patch(
        "/api/review/projects/export_route/pages/page_001/ocr_attempts/ocr_001",
        json={"review_status": "edited", "reviewed_text": "Lower reviewed text"},
    )
    client.patch(
        "/api/review/projects/export_route/pages/page_001/ocr_attempts/ocr_002",
        json={"review_status": "accepted", "reviewed_text": "Upper reviewed text"},
    )

    response = client.post("/api/review/projects/export_route/pages/page_001/export")

    assert response.status_code == 200
    payload = response.get_json()
    export_dir = Path(payload["export_dir"])
    assert export_dir.exists()
    files = payload["files"]
    for key in ["project_state_snapshot", "page_review_json", "page_review_markdown", "page_text", "crops_dir"]:
        assert Path(files[key]).exists()

    page_review = json.loads(Path(files["page_review_json"]).read_text(encoding="utf-8"))
    assert page_review["project_id"] == "export_route"
    assert page_review["page_id"] == "page_001"
    assert len(page_review["regions"]) == 4

    lower = next(region for region in page_review["regions"] if region["region_id"] == "r_001")
    assert lower["ocr"]["raw_text"] == "Lower raw text"
    assert lower["ocr"]["reviewed_text"] == "Lower reviewed text"
    assert lower["ocr"]["preferred_text"] == "Lower reviewed text"
    assert lower["ocr"]["source_text_mutated"] is False

    diagram = next(region for region in page_review["regions"] if region["region_id"] == "r_003")
    assert diagram["crop_path"] == "crops/region_r_003.png"
    with Image.open(export_dir / diagram["crop_path"]) as crop:
        assert crop.size == (30, 23)

    ignored = next(region for region in page_review["regions"] if region["region_id"] == "r_004")
    assert ignored["ignored"] is True
    assert ignored["crop_path"] is None

    text_export = Path(files["page_text"]).read_text(encoding="utf-8")
    assert text_export == "Upper reviewed text\n\nLower reviewed text\n"

    markdown = Path(files["page_review_markdown"]).read_text(encoding="utf-8")
    assert "# Page Review Export: page_001" in markdown
    assert "Upper reviewed text" in markdown
    assert "Lower raw text" in markdown


def test_review_page_export_uses_effective_oriented_image_for_crops(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "page.png", size=(90, 60))
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_oriented"})
    client.patch(
        "/api/review/projects/export_oriented/pages/page_001/orientation",
        json={"reviewed_rotation_degrees": 90},
    )
    client.post(
        "/api/review/projects/export_oriented/pages/page_001/regions",
        json={"reviewed_type": "diagram", "reviewed_bbox": [5, 7, 11, 13]},
    )

    response = client.post("/api/review/projects/export_oriented/pages/page_001/export")

    assert response.status_code == 200
    payload = response.get_json()
    review = json.loads(Path(payload["files"]["page_review_json"]).read_text(encoding="utf-8"))
    region = review["regions"][0]
    assert review["orientation"]["effective_rotation_degrees"] == 90
    with Image.open(Path(payload["export_dir"]) / region["crop_path"]) as crop:
        assert crop.size == (11, 13)


def test_review_page_export_does_not_require_ocr_attempts(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "page.png", size=(80, 50))
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_no_ocr"})
    client.post(
        "/api/review/projects/export_no_ocr/pages/page_001/regions",
        json={"reviewed_type": "english_text", "reviewed_bbox": [2, 3, 20, 10]},
    )

    response = client.post("/api/review/projects/export_no_ocr/pages/page_001/export")

    assert response.status_code == 200
    payload = response.get_json()
    review = json.loads(Path(payload["files"]["page_review_json"]).read_text(encoding="utf-8"))
    assert review["regions"][0]["ocr"]["latest_attempt_id"] is None
    assert review["regions"][0]["ocr"]["preferred_text"] == ""
    assert Path(payload["files"]["page_text"]).read_text(encoding="utf-8") == ""
