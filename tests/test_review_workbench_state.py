from __future__ import annotations

import json

from PIL import Image

from martial_arts_ocr.review.workbench_state import ReviewWorkbenchStore


def _write_image(path, size=(120, 80)):
    Image.new("RGB", size, "white").save(path)


def test_create_project_lists_pages_and_writes_project_state(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page_b.jpg", size=(200, 100))
    _write_image(source / "page_a.png", size=(120, 80))
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")

    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="example")

    assert state["project_id"] == "example"
    assert state["source_folder"] == str(source.resolve())
    assert [page["filename"] for page in state["pages"]] == ["page_a.png", "page_b.jpg"]
    assert state["pages"][0]["width"] == 120
    assert state["pages"][0]["height"] == 80
    assert store.project_path("example").exists()

    saved = json.loads(store.project_path("example").read_text(encoding="utf-8"))
    assert saved["metadata"]["schema_version"] == 1
    assert saved["metadata"]["local_only"] is True


def test_add_update_ignore_delete_region_preserves_detected_fields(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(200, 160))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="example")
    page_id = state["pages"][0]["page_id"]

    region = store.add_region(
        state,
        page_id,
        {
            "reviewed_type": "modern_japanese_vertical",
            "reviewed_bbox": [10, 20, 50, 90],
            "notes": "manual sidebar",
        },
    )

    assert region["region_id"] == "r_001"
    assert region["detected_type"] is None
    assert region["detected_bbox"] is None
    assert region["reviewed_type"] == "modern_japanese_vertical"
    assert region["effective_type"] == "modern_japanese_vertical"
    assert region["effective_bbox"] == [10, 20, 50, 90]
    assert region["source"] == "manual"

    # Simulate a machine-detected region, then verify reviewer edits do not
    # overwrite detected evidence.
    page = store.get_page(state, page_id)
    page["regions"].append(
        {
            "region_id": "r_002",
            "detected_type": "image",
            "reviewed_type": None,
            "effective_type": "image",
            "detected_bbox": [100, 30, 40, 40],
            "reviewed_bbox": None,
            "effective_bbox": [100, 30, 40, 40],
            "status": "detected",
            "source": "detector",
            "notes": "",
        }
    )
    store.save_project(state)

    updated = store.update_region(
        state,
        page_id,
        "r_002",
        {
            "reviewed_type": "caption_label",
            "reviewed_bbox": [95, 25, 55, 45],
            "notes": "diagram label",
        },
    )

    assert updated["detected_type"] == "image"
    assert updated["detected_bbox"] == [100, 30, 40, 40]
    assert updated["reviewed_type"] == "caption_label"
    assert updated["reviewed_bbox"] == [95, 25, 55, 45]
    assert updated["effective_type"] == "caption_label"
    assert updated["effective_bbox"] == [95, 25, 55, 45]
    assert updated["status"] == "reviewed"
    assert updated["source"] == "reviewer_override"

    ignored = store.update_region(state, page_id, "r_002", {"reviewed_type": "ignore"})
    assert ignored["status"] == "ignored"
    assert ignored["ignored"] is True
    assert ignored["effective_type"] == "ignore"

    store.delete_region(state, page_id, "r_001")
    assert [region["region_id"] for region in store.get_page(state, page_id)["regions"]] == ["r_002"]


def test_reloads_project_state_from_disk(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png")
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="example")
    store.add_region(state, "page_001", {"reviewed_type": "english_text"})

    loaded = store.load_project("example")

    assert loaded["project_id"] == "example"
    assert loaded["pages"][0]["regions"][0]["effective_type"] == "english_text"


def test_import_detected_regions_preserves_reviewer_work_and_replaces_unreviewed_machine_regions(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(200, 160))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="example")
    page_id = state["pages"][0]["page_id"]

    page = store.import_detected_regions(
        state,
        page_id,
        [
            {
                "region_type": "image",
                "bbox": [10, 12, 40, 50],
                "confidence": 0.82,
                "metadata": {"detector": "fake_detector"},
            },
            {
                "region_type": "image",
                "bbox": [70, 20, 30, 35],
                "metadata": {"needs_review": True, "mixed_region": True},
            },
        ],
    )

    assert [region["region_id"] for region in page["regions"]] == ["det_001", "det_002"]
    first = page["regions"][0]
    assert first["detected_type"] == "image"
    assert first["detected_bbox"] == [10, 12, 40, 50]
    assert first["effective_bbox"] == [10, 12, 40, 50]
    assert first["metadata"]["detector"] == "fake_detector"
    assert first["source"] == "machine_detection"
    assert page["regions"][1]["detected_type"] == "unknown_needs_review"

    manual = store.add_region(state, page_id, {"reviewed_type": "caption_label", "bbox": [5, 5, 20, 20]})
    reviewed = store.update_region(
        state,
        page_id,
        "det_001",
        {"reviewed_type": "diagram", "reviewed_bbox": [8, 10, 45, 55]},
    )

    rerun_page = store.import_detected_regions(
        state,
        page_id,
        [
            {
                "region_type": "photo",
                "bbox": [100, 70, 25, 30],
                "metadata": {"detector": "second_run"},
            }
        ],
    )

    regions = {region["region_id"]: region for region in rerun_page["regions"]}
    assert manual["region_id"] in regions
    assert regions["det_001"]["reviewed_type"] == "diagram"
    assert regions["det_001"]["detected_bbox"] == reviewed["detected_bbox"]
    assert regions["det_002"]["detected_type"] == "photo"
    assert regions["det_002"]["metadata"]["detector"] == "second_run"
    assert rerun_page["recognition"]["rerun_behavior"] == "replaced_unreviewed_machine_detection_regions"


def test_source_folder_must_be_under_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    _write_image(outside / "page.png")

    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[allowed])

    try:
        store.create_project(outside, project_id="blocked")
    except PermissionError as exc:
        assert "outside configured review roots" in str(exc)
    else:
        raise AssertionError("Expected PermissionError")
