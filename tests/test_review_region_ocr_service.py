from __future__ import annotations

from martial_arts_ocr.review.region_ocr_service import RegionOCRService


def test_region_ocr_service_routes_review_types_without_changing_defaults():
    service = RegionOCRService()

    assert service.route_for_region_type("english_text").to_dict() == {
        "engine": "tesseract",
        "language": "eng",
        "psm": 6,
        "preprocess_profile": "none",
        "variant_id": "default",
        "status": "ready",
        "reason": "",
    }
    assert service.route_for_region_type("modern_japanese_horizontal").language == "jpn"
    assert service.route_for_region_type("modern_japanese_horizontal").psm == 6
    assert service.route_for_region_type("modern_japanese_horizontal").preprocess_profile == "upscale_2x"
    assert service.route_for_region_type("modern_japanese_vertical").language == "jpn_vert"
    assert service.route_for_region_type("modern_japanese_vertical").psm == 5
    assert service.route_for_region_type("mixed_english_japanese").language == "eng+jpn"


def test_region_ocr_service_exposes_deterministic_variants():
    service = RegionOCRService()

    variants = service.routes_for_region_type("english_text")

    assert [route.variant_id for route in variants] == [
        "default",
        "psm_4",
        "psm_11",
        "grayscale",
        "threshold",
        "upscale_2x",
        "contrast_sharpen",
    ]
    assert variants[0].language == "eng"
    assert variants[0].psm == 6


def test_region_ocr_service_variants_for_non_text_region_are_skipped():
    service = RegionOCRService()

    results = service.run_variants(
        image_path="missing.png",
        bbox=[0, 0, 10, 10],
        region_type="image",
        region_id="r_001",
    )

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].route.status == "skipped"


def test_region_ocr_service_skips_non_text_region_types():
    service = RegionOCRService()

    route = service.route_for_region_type("diagram")

    assert route.status == "skipped"
    assert route.language is None
    assert "not OCR-routable" in route.reason
