from __future__ import annotations

from martial_arts_ocr.review.region_ocr_service import RegionOCRService


def test_region_ocr_service_routes_review_types_without_changing_defaults():
    service = RegionOCRService()

    assert service.route_for_region_type("english_text").to_dict() == {
        "engine": "tesseract",
        "language": "eng",
        "psm": 6,
        "preprocess_profile": "none",
        "status": "ready",
        "reason": "",
    }
    assert service.route_for_region_type("modern_japanese_horizontal").language == "jpn"
    assert service.route_for_region_type("modern_japanese_horizontal").psm == 6
    assert service.route_for_region_type("modern_japanese_horizontal").preprocess_profile == "upscale_2x"
    assert service.route_for_region_type("modern_japanese_vertical").language == "jpn_vert"
    assert service.route_for_region_type("modern_japanese_vertical").psm == 5
    assert service.route_for_region_type("mixed_english_japanese").language == "eng+jpn"


def test_region_ocr_service_skips_non_text_region_types():
    service = RegionOCRService()

    route = service.route_for_region_type("diagram")

    assert route.status == "skipped"
    assert route.language is None
    assert "not OCR-routable" in route.reason
