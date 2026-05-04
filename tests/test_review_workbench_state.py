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
    assert state["pages"][0]["effective_width"] == 120
    assert state["pages"][0]["effective_height"] == 80
    assert state["pages"][0]["orientation"]["effective_rotation_degrees"] == 0
    assert store.project_path("example").exists()

    saved = json.loads(store.project_path("example").read_text(encoding="utf-8"))
    assert saved["metadata"]["schema_version"] == 1
    assert saved["metadata"]["local_only"] is True


def test_update_page_orientation_marks_existing_regions_stale(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(200, 120))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="orientation")
    page_id = state["pages"][0]["page_id"]
    store.add_region(state, page_id, {"reviewed_type": "caption_label", "bbox": [10, 10, 40, 30]})

    page = store.update_page_orientation(
        state,
        page_id,
        detected_rotation_degrees=90,
        detected_confidence=0.94,
        source="orientation_cnn",
        status="detected",
        metadata={"model_used": "convnext"},
    )

    assert page["orientation"]["detected_rotation_degrees"] == 90
    assert page["orientation"]["effective_rotation_degrees"] == 90
    assert page["orientation"]["detected_confidence"] == 0.94
    assert page["effective_width"] == 120
    assert page["effective_height"] == 200
    assert page["regions_stale"] is True
    assert page["regions"][0]["stale"] is True
    assert page["regions"][0]["metadata"]["stale_reason"] == "orientation_changed"

    reviewed = store.update_page_orientation(
        state,
        page_id,
        reviewed_rotation_degrees=180,
        source="reviewer_override",
        status="reviewed",
    )

    assert reviewed["orientation"]["reviewed_rotation_degrees"] == 180
    assert reviewed["orientation"]["effective_rotation_degrees"] == 180
    assert reviewed["effective_width"] == 200
    assert reviewed["effective_height"] == 120


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
    assert region["source"] == "reviewer_manual"
    assert region["review_status"] == "manually_added"
    assert region["training_feedback"]["label"] == "manually_added"
    assert region["training_feedback"]["feedback_type"] == "missed_positive"
    assert region["training_feedback"]["target_type"] == "modern_japanese_vertical"
    assert region["metadata"]["feedback_type"] == "missed_positive"
    assert region["metadata"]["manually_added"] is True

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
    assert updated["review_status"] == "resized"
    assert updated["training_feedback"]["label"] == "resized_positive"

    ignored = store.update_region(state, page_id, "r_002", {"reviewed_type": "ignore"})
    assert ignored["status"] == "ignored"
    assert ignored["ignored"] is True
    assert ignored["effective_type"] == "ignore"
    assert ignored["review_status"] == "rejected"
    assert ignored["training_feedback"]["label"] == "false_positive"

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


def test_duplicate_region_creates_reviewed_manual_copy_and_preserves_original(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(220, 140))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="duplicate")
    page_id = state["pages"][0]["page_id"]
    page = store.import_detected_regions(
        state,
        page_id,
        [
            {
                "region_type": "diagram",
                "bbox": [80, 20, 50, 60],
                "metadata": {"detector": "fake_detector"},
            }
        ],
    )
    original = dict(page["regions"][0])

    duplicate = store.duplicate_region(state, page_id, "det_001", direction="right")

    assert duplicate["region_id"] == "r_001"
    assert duplicate["source"] == "reviewer_manual_duplicate"
    assert duplicate["status"] == "reviewed"
    assert duplicate["detected_bbox"] is None
    assert duplicate["reviewed_bbox"] == [130, 20, 50, 60]
    assert duplicate["effective_bbox"] == [130, 20, 50, 60]
    assert duplicate["reviewed_type"] == "diagram"
    assert duplicate["effective_type"] == "diagram"
    assert duplicate["metadata"]["duplicated_from_region_id"] == "det_001"
    assert duplicate["metadata"]["duplicated_from_detector"] == "fake_detector"
    assert duplicate["review_status"] == "manually_added"
    assert duplicate["training_feedback"]["label"] == "manually_added"
    assert duplicate["training_feedback"]["related_machine_regions"] == ["det_001"]
    assert store.get_region(store.get_page(state, page_id), "det_001") == original

    loaded = store.load_project("duplicate")
    loaded_duplicate = store.get_region(store.get_page(loaded, page_id), "r_001")
    assert loaded_duplicate["source"] == "reviewer_manual_duplicate"

    updated_duplicate = store.update_region(
        loaded,
        page_id,
        "r_001",
        {"reviewed_bbox": [135, 25, 45, 55]},
    )
    assert updated_duplicate["source"] == "reviewer_manual_duplicate"
    assert updated_duplicate["reviewed_bbox"] == [135, 25, 45, 55]


def test_duplicate_region_left_right_clamps_to_page_bounds(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(120, 90))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="duplicate_clamp")
    page_id = state["pages"][0]["page_id"]
    store.add_region(state, page_id, {"reviewed_type": "image", "bbox": [5, 10, 40, 50]})

    left = store.duplicate_region(state, page_id, "r_001", direction="left")
    right = store.duplicate_region(state, page_id, "r_001", direction="right")

    assert left["reviewed_bbox"] == [0, 10, 40, 50]
    assert right["reviewed_bbox"] == [45, 10, 40, 50]


def test_multiple_drawn_manual_regions_and_ignore_region_state(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(160, 120))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="manual_draw")
    page_id = state["pages"][0]["page_id"]

    first = store.add_region(state, page_id, {"reviewed_type": "image", "bbox": [10, 10, 30, 40]})
    second = store.add_region(state, page_id, {"reviewed_type": "english_text", "bbox": [60, 15, 70, 20]})
    ignored = store.add_region(state, page_id, {"reviewed_type": "ignore", "bbox": [0, 0, 12, 12]})

    assert [first["region_id"], second["region_id"], ignored["region_id"]] == ["r_001", "r_002", "r_003"]
    assert first["source"] == "reviewer_manual"
    assert first["detected_bbox"] is None
    assert first["reviewed_bbox"] == [10, 10, 30, 40]
    assert first["effective_bbox"] == [10, 10, 30, 40]
    assert first["metadata"]["feedback_type"] == "missed_positive"
    assert second["source"] == "reviewer_manual"
    assert ignored["status"] == "ignored"
    assert ignored["ignored"] is True
    assert ignored["effective_type"] == "ignore"
    assert ignored["review_status"] == "ignored"

    loaded = store.load_project("manual_draw")
    loaded_regions = loaded["pages"][0]["regions"]
    assert [region["source"] for region in loaded_regions] == ["reviewer_manual", "reviewer_manual", "reviewer_manual"]


def test_add_region_ocr_attempt_records_page_and_region_links(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(120, 90))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="ocr_attempt")
    page_id = state["pages"][0]["page_id"]
    region = store.add_region(state, page_id, {"reviewed_type": "english_text", "bbox": [5, 10, 60, 20]})

    attempt = store.add_region_ocr_attempt(
        state,
        page_id,
        region["region_id"],
        {
            "text": "Daito-ryu",
            "cleaned_text": "Daito-ryu",
            "confidence": 0.83,
            "route": {"engine": "tesseract", "language": "eng", "psm": 6},
            "status": "ok",
            "source_text_mutated": False,
        },
    )

    assert attempt["attempt_id"] == "ocr_001"
    assert attempt["region_id"] == "r_001"
    assert attempt["region_type"] == "english_text"
    assert attempt["bbox"] == [5, 10, 60, 20]
    assert attempt["orientation_degrees"] == 0
    page = store.get_page(state, page_id)
    assert page["ocr_attempts"][0]["text"] == "Daito-ryu"
    assert page["ocr_attempts"][0]["review_status"] == "unreviewed"
    assert page["ocr_attempts"][0]["source_text_mutated"] is False
    stored_region = store.get_region(page, "r_001")
    assert stored_region["ocr_attempt_ids"] == ["ocr_001"]
    assert stored_region["last_ocr_attempt_id"] == "ocr_001"

    loaded = store.load_project("ocr_attempt")
    assert loaded["pages"][0]["ocr_attempts"][0]["attempt_id"] == "ocr_001"


def test_update_ocr_attempt_review_preserves_raw_text_and_records_reviewed_text(tmp_path):
    source = tmp_path / "scans"
    source.mkdir()
    _write_image(source / "page.png", size=(120, 90))
    store = ReviewWorkbenchStore(tmp_path / "runtime" / "review_projects", allowed_roots=[tmp_path])
    state = store.create_project(source, project_id="ocr_review")
    page_id = state["pages"][0]["page_id"]
    region = store.add_region(state, page_id, {"reviewed_type": "english_text", "bbox": [5, 10, 60, 20]})
    attempt = store.add_region_ocr_attempt(
        state,
        page_id,
        region["region_id"],
        {
            "text": "Le opmgageet",
            "cleaned_text": "Le opmgageet",
            "confidence": 0.42,
            "route": {"engine": "tesseract", "language": "eng", "psm": 6},
            "status": "ok",
            "source_text_mutated": False,
        },
    )

    reviewed = store.update_ocr_attempt_review(
        state,
        page_id,
        attempt["attempt_id"],
        {
            "reviewed_text": "[Question.]\nAaaa yes.",
            "review_status": "edited",
        },
    )

    assert reviewed["text"] == "Le opmgageet"
    assert reviewed["cleaned_text"] == "Le opmgageet"
    assert reviewed["reviewed_text"] == "[Question.]\nAaaa yes."
    assert reviewed["review_status"] == "edited"
    assert reviewed["source_text_mutated"] is False
    assert "reviewed_at" in reviewed

    accepted = store.update_ocr_attempt_review(state, page_id, attempt["attempt_id"], {"review_status": "accepted"})
    assert accepted["review_status"] == "accepted"
    assert accepted["reviewed_text"] == "[Question.]\nAaaa yes."

    rejected = store.update_ocr_attempt_review(state, page_id, attempt["attempt_id"], {"review_status": "rejected"})
    assert rejected["review_status"] == "rejected"
    assert rejected["text"] == "Le opmgageet"

    try:
        store.update_ocr_attempt_review(state, page_id, attempt["attempt_id"], {"review_status": "final_truth"})
    except ValueError as exc:
        assert "review_status must be" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("invalid OCR review status should raise")


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
    assert first["review_status"] == "unreviewed"
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
