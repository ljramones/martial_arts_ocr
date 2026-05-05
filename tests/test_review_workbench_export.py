from __future__ import annotations

import json
import zipfile
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


def test_review_project_export_v2_writes_multi_page_bundle_and_html(tmp_path):
    app, data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "page_a.png", size=(120, 90))
    _write_marker_image(scans / "page_b.png", size=(110, 80))

    class FakeRegionOCRService:
        def run(self, *, image_path, bbox, region_type, region_id):
            text_by_region = {
                "r_001": "Page one raw",
                "r_001_b": "unused",
            }
            return RegionOCRResult(
                text=text_by_region.get(region_id, f"{region_id} raw"),
                cleaned_text=text_by_region.get(region_id, f"{region_id} cleaned"),
                confidence=0.8,
                route=RegionOCRRoute(language="eng", psm=6),
                status="ok",
            )

    app.config["REVIEW_REGION_OCR_SERVICE"] = FakeRegionOCRService()
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_v2"})
    client.post(
        "/api/review/projects/export_v2/pages/page_001/regions",
        json={"reviewed_type": "english_text", "reviewed_bbox": [5, 5, 45, 20]},
    )
    client.post(
        "/api/review/projects/export_v2/pages/page_001/regions",
        json={"reviewed_type": "diagram", "reviewed_bbox": [10, 30, 30, 25]},
    )
    client.post(
        "/api/review/projects/export_v2/pages/page_001/regions",
        json={"reviewed_type": "ignore", "reviewed_bbox": [70, 5, 20, 20]},
    )
    client.post(
        "/api/review/projects/export_v2/pages/page_002/regions",
        json={"reviewed_type": "caption_label", "reviewed_bbox": [4, 8, 35, 18]},
    )
    client.post("/api/review/projects/export_v2/pages/page_001/regions/r_001/ocr")
    client.patch(
        "/api/review/projects/export_v2/pages/page_001/ocr_attempts/ocr_001",
        json={"review_status": "edited", "reviewed_text": "Page one reviewed"},
    )
    client.post("/api/review/projects/export_v2/pages/page_002/regions/r_001/ocr")
    client.patch(
        "/api/review/projects/export_v2/pages/page_002/ocr_attempts/ocr_001",
        json={"review_status": "accepted", "reviewed_text": "Page two caption"},
    )
    state_path = data_dir / "runtime" / "review_projects" / "export_v2" / "project_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    page_001 = next(page for page in state["pages"] if page["page_id"] == "page_001")
    diagram = next(region for region in page_001["regions"] if region["region_id"] == "r_002")
    diagram["metadata"]["needs_review"] = True
    diagram["review_status"] = "unreviewed"
    diagram["status"] = "detected"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    response = client.post(
        "/api/review/projects/export_v2/export",
        json={"page_selection": {"mode": "all"}, "formats": ["review_bundle", "html", "docx"]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    export_dir = Path(payload["export_path"])
    assert payload["page_ids"] == ["page_001", "page_002"]
    assert Path(payload["manifest_path"]).exists()
    assert (export_dir / "project_state_snapshot.json").exists()
    assert (export_dir / "document_export_model.json").exists()

    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_version"] == 2
    assert manifest["page_selection"]["mode"] == "all"
    assert manifest["formats"] == ["review_bundle", "html", "docx"]
    assert manifest["source_text_mutated"] is False
    assert manifest["artifact_paths"]["docx"]["document"].endswith("document.docx")

    page_001_json = export_dir / "review_bundle" / "pages" / "page_001_review.json"
    page_002_json = export_dir / "review_bundle" / "pages" / "page_002_review.json"
    assert page_001_json.exists()
    assert page_002_json.exists()
    page_001_review = json.loads(page_001_json.read_text(encoding="utf-8"))
    text_region = next(region for region in page_001_review["regions"] if region["region_id"] == "r_001")
    assert text_region["ocr"]["raw_text"] == "Page one raw"
    assert text_region["ocr"]["reviewed_text"] == "Page one reviewed"
    assert text_region["ocr"]["preferred_text"] == "Page one reviewed"
    assert text_region["ocr"]["source_text_mutated"] is False

    ignored = next(region for region in page_001_review["regions"] if region["region_id"] == "r_003")
    assert ignored["ignored"] is True
    assert ignored["crop_path"] is None

    crop = export_dir / "review_bundle" / "crops" / "page_001_region_r_002.png"
    assert crop.exists()
    with Image.open(crop) as image:
        assert image.size == (30, 25)

    page_001_text = (export_dir / "review_bundle" / "pages" / "page_001_text.txt").read_text(encoding="utf-8")
    assert page_001_text == "Page one reviewed\n"
    page_002_text = (export_dir / "review_bundle" / "pages" / "page_002_text.txt").read_text(encoding="utf-8")
    assert page_002_text == "Page two caption\n"

    html = (export_dir / "html" / "document.html").read_text(encoding="utf-8")
    assert "Workbench Review Export: export_v2" in html
    assert "Source text mutated" in html
    assert "source_text_mutated=false" in html
    assert "<nav class=\"toc\"" in html
    assert "Page 1 - page_a.png" in html
    assert "Page 2 - page_b.png" in html
    assert "reading_order_uncertain" in html
    assert "needs_review" in html
    assert "Reviewed / Display Text" in html
    assert "Page one reviewed" in html
    assert "Page two caption" in html
    assert "Raw / cleaned OCR evidence" in html
    assert "Page one raw" in html
    assert "OCR route:" in html
    assert "Crop asset:" in html
    assert "assets/page_001_region_r_002.png" in html
    assert (export_dir / "html" / "assets" / "page_001_region_r_002.png").exists()

    docx_path = export_dir / "docx" / "document.docx"
    assert docx_path.exists()
    assert payload["formats"]["docx"] == str(docx_path)
    with zipfile.ZipFile(docx_path) as archive:
        names = set(archive.namelist())
        assert "word/document.xml" in names
        assert "word/media/page_001_region_r_002.png" in names
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Workbench Review Export: export_v2" in document_xml
    assert "Page one reviewed" in document_xml
    assert "Page two caption" in document_xml
    assert "Raw / cleaned OCR evidence" in document_xml
    assert "source_text_mutated=false" in document_xml


def test_review_project_export_v2_supports_selected_and_range_modes(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "a.png", size=(40, 30))
    _write_marker_image(scans / "b.png", size=(40, 30))
    _write_marker_image(scans / "c.png", size=(40, 30))
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_v2_select"})

    selected_response = client.post(
        "/api/review/projects/export_v2_select/export",
        json={"page_selection": {"mode": "selected", "page_ids": ["page_002"]}, "formats": ["review_bundle"]},
    )
    assert selected_response.status_code == 200
    assert selected_response.get_json()["page_ids"] == ["page_002"]

    range_response = client.post(
        "/api/review/projects/export_v2_select/export",
        json={
            "page_selection": {"mode": "range", "range": {"start": "page_003", "end": "page_001"}},
            "formats": ["html"],
        },
    )
    assert range_response.status_code == 200
    assert range_response.get_json()["page_ids"] == ["page_001", "page_002", "page_003"]


def test_review_project_export_v2_rejects_unsupported_formats(tmp_path):
    app, _data_dir, scans = _create_review_app(tmp_path)
    _write_marker_image(scans / "page.png", size=(40, 30))
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "export_v2_bad"})

    response = client.post(
        "/api/review/projects/export_v2_bad/export",
        json={"page_selection": {"mode": "all"}, "formats": ["pdf"]},
    )

    assert response.status_code == 400
