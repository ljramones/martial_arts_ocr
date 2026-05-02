from __future__ import annotations

import csv
import json

from PIL import Image

from experiments import review_japanese_region_ocr as helper


def test_parser_exposes_routing_profile_flags():
    help_text = helper.build_parser().format_help()

    assert "--profile" in help_text
    assert "--use-routing-profiles" in help_text
    assert "--no-use-routing-profiles" in help_text


def test_region_profiles_encode_expected_routes():
    vertical = helper.REGION_PROFILES["vertical_modern_japanese"]
    mixed_page_text = helper.REGION_PROFILES["mixed_english_japanese_page_text"]
    mixed_parentheticals = helper.REGION_PROFILES["mixed_japanese_parentheticals"]
    calligraphy = helper.REGION_PROFILES["stylized_calligraphy"]

    assert vertical.routes[0].language == "jpn_vert"
    assert vertical.routes[0].psm == "5"
    assert "upscale_2x" in vertical.routes[0].preprocess_profiles

    assert ("eng+jpn", "6") in {
        (route.language, route.psm) for route in mixed_page_text.routes
    }
    assert ("eng+jpn", "11") in {
        (route.language, route.psm) for route in mixed_page_text.routes
    }
    assert mixed_page_text.routes[0].language == "eng+jpn"

    assert mixed_parentheticals.routes[0].language == "jpn"
    assert mixed_parentheticals.routes[0].role == "primary_term_recovery"
    assert helper.REGION_PROFILES["mixed_english_japanese"] is mixed_page_text

    assert calligraphy.needs_review_default is True


def test_select_region_profile_uses_manifest_type_and_override():
    sample = {"id": "region_1", "japanese_region_type": "vertical_modern_japanese"}

    selected = helper._select_region_profile(
        sample,
        profile_override=None,
        use_routing_profiles=True,
    )
    overridden = helper._select_region_profile(
        sample,
        profile_override="horizontal_modern_japanese",
        use_routing_profiles=True,
    )
    disabled = helper._select_region_profile(
        sample,
        profile_override=None,
        use_routing_profiles=False,
    )

    assert selected.name == "vertical_modern_japanese"
    assert overridden.name == "horizontal_modern_japanese"
    assert disabled.name == "unknown_japanese_like"


def test_routed_processing_records_selected_profile_and_attempt_metadata(tmp_path, monkeypatch):
    image_path = tmp_path / "japanese.png"
    Image.new("RGB", (80, 60), "white").save(image_path)

    def fake_run(_image, *, language: str, psm: str):
        if language == "jpn_vert" and str(psm) == "5":
            return "忍者 伊賀 甲賀", 0.01
        return "noise", 0.01

    monkeypatch.setattr(helper, "_run_tesseract", fake_run)

    result = helper._process_sample(
        {
            "id": "vertical_region",
            "source_image": str(image_path),
            "bbox": [0, 0, 40, 40],
            "japanese_region_type": "vertical_modern_japanese",
            "expected_terms": ["忍者", "伊賀", "甲賀"],
        },
        output_dir=tmp_path / "out",
        crops_dir=tmp_path / "out" / "crops",
        languages=["eng"],
        psms=["6"],
        profile_override=None,
        use_routing_profiles=True,
        easyocr_reader=None,
    )

    assert result["routing_profile"]["name"] == "vertical_modern_japanese"
    assert result["selected_routes"][0]["language"] == "jpn_vert"
    assert result["selected_routes"][0]["psm"] == "5"
    assert result["best_tesseract"]["language"] == "jpn_vert"
    assert result["best_tesseract"]["preprocess_profile"] in {
        "none",
        "upscale_2x",
        "threshold",
    }
    assert result["best_tesseract"]["expected_terms_recovered"] == ["忍者", "伊賀", "甲賀"]
    assert result["best_tesseract"]["quality_judgment"] == "meaningful"


def test_legacy_matrix_mode_uses_requested_languages_psms_and_all_preprocess_profiles(tmp_path, monkeypatch):
    image_path = tmp_path / "region.png"
    Image.new("RGB", (50, 50), "white").save(image_path)

    calls: list[tuple[str, str]] = []

    def fake_run(_image, *, language: str, psm: str):
        calls.append((language, str(psm)))
        return f"{language}-{psm}", 0.01

    monkeypatch.setattr(helper, "_run_tesseract", fake_run)

    result = helper._process_sample(
        {
            "id": "legacy_region",
            "source_image": str(image_path),
            "bbox": [0, 0, 20, 20],
            "expected_terms": [],
        },
        output_dir=tmp_path / "out",
        crops_dir=tmp_path / "out" / "crops",
        languages=["jpn", "eng"],
        psms=["6", "11"],
        profile_override=None,
        use_routing_profiles=False,
        easyocr_reader=None,
    )

    routed_variant_count = (
        len(["jpn", "eng"])
        * len(["6", "11"])
        * len(helper.DEFAULT_PREPROCESS_PROFILES)
    )
    full_page_count = len(["jpn", "eng"]) * len(["6", "11"])

    assert result["routing_profile"]["name"] == "unknown_japanese_like"
    assert len(result["variants"]) == routed_variant_count
    assert len(result["full_page_results"]) == full_page_count
    assert len(calls) == routed_variant_count + full_page_count


def test_quality_judgment_distinguishes_meaningful_partial_noisy_and_fail():
    assert helper._quality_judgment("忍者 伊賀", ["忍者", "伊賀"]) == "meaningful"
    assert helper._quality_judgment("忍者", ["忍者", "伊賀"]) == "partial"
    assert helper._quality_judgment("かなカナ漢字", ["忍者"]) == "noisy"
    assert helper._quality_judgment("latin noise", ["忍者"]) == "fail"


def test_review_result_shape_includes_route_terms_quality_and_output(tmp_path):
    image_path = tmp_path / "region.png"
    image_path.write_text("placeholder", encoding="utf-8")
    profile = helper.REGION_PROFILES["vertical_modern_japanese"]

    result = helper._build_review_result(
        {
            "id": "vertical_region",
            "notes": "manual crop",
        },
        source_path=image_path,
        bbox=[1, 2, 30, 40],
        routing_profile=profile,
        best={
            "language": "jpn_vert",
            "psm": "5",
            "preprocess_profile": "none",
            "route_role": "primary",
            "text": "忍者 伊賀",
            "quality_judgment": "partial",
        },
        expected_terms=["忍者", "伊賀", "甲賀"],
        needs_review=False,
    )

    assert result["sample_id"] == "vertical_region"
    assert result["source_image"] == str(image_path)
    assert result["bbox"] == [1, 2, 30, 40]
    assert result["region_type"] == "vertical_modern_japanese"
    assert result["route"] == {
        "language": "jpn_vert",
        "psm": 5,
        "preprocess_profile": "none",
        "role": "primary",
    }
    assert result["ocr_output"] == "忍者 伊賀"
    assert result["expected_terms"] == ["忍者", "伊賀", "甲賀"]
    assert result["terms_recovered"] == ["忍者", "伊賀"]
    assert result["terms_missing"] == ["甲賀"]
    assert result["quality_judgment"] == "partial"
    assert result["needs_review"] is True


def test_review_exports_write_json_csv_and_markdown(tmp_path):
    review_results = [
        {
            "sample_id": "vertical_region",
            "source_image": "sample.jpg",
            "bbox": [1, 2, 30, 40],
            "region_type": "vertical_modern_japanese",
            "route": {
                "language": "jpn_vert",
                "psm": 5,
                "preprocess_profile": "none",
                "role": "primary",
            },
            "ocr_output": "忍者 伊賀",
            "ocr_output_preview": ["忍者 伊賀"],
            "expected_terms": ["忍者", "伊賀", "甲賀"],
            "terms_recovered": ["忍者", "伊賀"],
            "terms_missing": ["甲賀"],
            "quality_judgment": "partial",
            "needs_review": True,
            "notes": "manual crop",
        }
    ]

    helper._write_review_exports(
        review_results,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.local.json",
    )

    json_payload = json.loads((tmp_path / "region_ocr_results.json").read_text(encoding="utf-8"))
    assert json_payload["schema_version"] == "japanese_region_ocr_review.v1"
    assert json_payload["result_count"] == 1
    assert json_payload["results"][0]["terms_missing"] == ["甲賀"]

    with (tmp_path / "region_ocr_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sample_id": "vertical_region",
            "source_image": "sample.jpg",
            "bbox": "[1, 2, 30, 40]",
            "region_type": "vertical_modern_japanese",
            "language": "jpn_vert",
            "psm": "5",
            "preprocess_profile": "none",
            "quality_judgment": "partial",
            "expected_terms": "忍者 | 伊賀 | 甲賀",
            "terms_recovered": "忍者 | 伊賀",
            "terms_missing": "甲賀",
            "needs_review": "true",
        }
    ]

    markdown = (tmp_path / "region_ocr_review.md").read_text(encoding="utf-8")
    assert "# Japanese Region OCR Review" in markdown
    assert "| vertical_region | vertical_modern_japanese | jpn_vert / PSM 5 / none |" in markdown
    assert "### vertical_region" in markdown
    assert "忍者 伊賀" in markdown


def test_process_sample_populates_review_result_without_ocr_engine(tmp_path, monkeypatch):
    image_path = tmp_path / "japanese.png"
    Image.new("RGB", (80, 60), "white").save(image_path)

    def fake_run(_image, *, language: str, psm: str):
        if language == "jpn" and str(psm) == "6":
            return "日本語 漢文", 0.01
        return "", 0.01

    monkeypatch.setattr(helper, "_run_tesseract", fake_run)

    result = helper._process_sample(
        {
            "id": "horizontal_region",
            "source_image": str(image_path),
            "bbox": [0, 0, 40, 40],
            "japanese_region_type": "horizontal_modern_japanese",
            "expected_terms": ["日本語", "漢文", "横書き"],
        },
        output_dir=tmp_path / "out",
        crops_dir=tmp_path / "out" / "crops",
        languages=["eng"],
        psms=["11"],
        profile_override=None,
        use_routing_profiles=True,
        easyocr_reader=None,
    )

    review = result["review_result"]
    assert review["sample_id"] == "horizontal_region"
    assert review["route"]["language"] == "jpn"
    assert review["route"]["psm"] == 6
    assert review["terms_recovered"] == ["日本語", "漢文"]
    assert review["terms_missing"] == ["横書き"]
    assert review["quality_judgment"] == "partial"
