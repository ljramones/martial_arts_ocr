from __future__ import annotations

from PIL import Image

from experiments import review_japanese_region_ocr as helper


def test_parser_exposes_routing_profile_flags():
    help_text = helper.build_parser().format_help()

    assert "--profile" in help_text
    assert "--use-routing-profiles" in help_text
    assert "--no-use-routing-profiles" in help_text


def test_region_profiles_encode_expected_routes():
    vertical = helper.REGION_PROFILES["vertical_modern_japanese"]
    mixed = helper.REGION_PROFILES["mixed_english_japanese"]
    calligraphy = helper.REGION_PROFILES["stylized_calligraphy"]

    assert vertical.routes[0].language == "jpn_vert"
    assert vertical.routes[0].psm == "5"
    assert "upscale_2x" in vertical.routes[0].preprocess_profiles

    assert ("eng+jpn", "6") in {
        (route.language, route.psm) for route in mixed.routes
    }
    assert ("eng+jpn", "11") in {
        (route.language, route.psm) for route in mixed.routes
    }

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
