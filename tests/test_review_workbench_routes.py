from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_image(path: Path, size=(90, 70)):
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


def test_review_page_route_loads(tmp_path):
    app, _data_dir, _scans = _create_review_app(tmp_path)
    client = app.test_client()

    response = client.get("/review")

    assert response.status_code == 200
    assert b"Local Research Review Workbench" in response.data


def test_review_project_routes_create_list_image_and_reload(tmp_path):
    app, data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page_001.png", size=(120, 80))
    _write_image(scans / "page_002.jpg", size=(100, 60))
    client = app.test_client()

    create_response = client.post(
        "/api/review/projects",
        json={"source_folder": str(scans), "project_id": "route_test"},
    )

    assert create_response.status_code == 200
    project = create_response.get_json()
    assert project["project_id"] == "route_test"
    assert len(project["pages"]) == 2
    assert project["pages"][0]["page_id"] == "page_001"

    state_path = data_dir / "runtime" / "review_projects" / "route_test" / "project_state.json"
    assert state_path.exists()

    load_response = client.get("/api/review/projects/route_test")
    assert load_response.status_code == 200
    assert load_response.get_json()["source_folder"] == str(scans.resolve())

    pages_response = client.get("/api/review/projects/route_test/pages")
    assert pages_response.status_code == 200
    assert len(pages_response.get_json()["pages"]) == 2

    page_response = client.get("/api/review/projects/route_test/pages/page_001")
    assert page_response.status_code == 200
    assert page_response.get_json()["width"] == 120

    image_response = client.get("/api/review/projects/route_test/pages/page_001/image")
    assert image_response.status_code == 200
    assert image_response.data.startswith(b"\x89PNG")


def test_review_region_routes_add_update_ignore_delete_and_preserve_detected_fields(tmp_path):
    app, data_dir, scans = _create_review_app(tmp_path)
    _write_image(scans / "page.png", size=(200, 120))
    client = app.test_client()
    client.post("/api/review/projects", json={"source_folder": str(scans), "project_id": "regions"})

    add_response = client.post(
        "/api/review/projects/regions/pages/page_001/regions",
        json={
            "reviewed_type": "modern_japanese_vertical",
            "reviewed_bbox": [15, 20, 40, 80],
            "notes": "sidebar",
        },
    )

    assert add_response.status_code == 201
    region = add_response.get_json()["region"]
    assert region["region_id"] == "r_001"
    assert region["detected_type"] is None
    assert region["effective_type"] == "modern_japanese_vertical"
    assert region["effective_bbox"] == [15, 20, 40, 80]

    update_response = client.patch(
        "/api/review/projects/regions/pages/page_001/regions/r_001",
        json={
            "reviewed_type": "caption_label",
            "reviewed_bbox": [10, 12, 50, 60],
            "notes": "label correction",
        },
    )

    assert update_response.status_code == 200
    updated = update_response.get_json()["region"]
    assert updated["reviewed_type"] == "caption_label"
    assert updated["effective_type"] == "caption_label"
    assert updated["reviewed_bbox"] == [10, 12, 50, 60]
    assert updated["detected_bbox"] is None
    assert updated["status"] == "reviewed"

    ignore_response = client.patch(
        "/api/review/projects/regions/pages/page_001/regions/r_001",
        json={"reviewed_type": "ignore"},
    )
    assert ignore_response.status_code == 200
    ignored = ignore_response.get_json()["region"]
    assert ignored["status"] == "ignored"
    assert ignored["ignored"] is True

    delete_response = client.delete("/api/review/projects/regions/pages/page_001/regions/r_001")
    assert delete_response.status_code == 200
    assert delete_response.get_json()["page"]["regions"] == []

    saved = json.loads(
        (data_dir / "runtime" / "review_projects" / "regions" / "project_state.json").read_text(encoding="utf-8")
    )
    assert saved["pages"][0]["regions"] == []


def test_review_project_rejects_disallowed_source_folder(tmp_path):
    app, _data_dir, _scans = _create_review_app(tmp_path)
    outside = tmp_path.parent / "outside_review_source"
    outside.mkdir(exist_ok=True)
    _write_image(outside / "page.png")
    client = app.test_client()

    response = client.post(
        "/api/review/projects",
        json={"source_folder": str(outside), "project_id": "blocked"},
    )

    assert response.status_code == 403
    assert "outside configured review roots" in response.get_json()["error"]
