from pathlib import Path

from martial_arts_ocr.pipeline.extraction_adapters import (
    document_result_from_extractions,
    image_region_from_extraction,
    page_result_from_extractions,
    text_region_from_extraction,
)
from martial_arts_ocr.pipeline.document_models import DocumentResult, ImageRegion, TextRegion
from utils.image.regions.core_types import ImageRegion as UtilityImageRegion


def test_text_extraction_output_maps_to_text_region():
    region = text_region_from_extraction(
        {
            "region_id": "text-a",
            "text": "Daitō-ryū Aiki-jūjutsu",
            "bbox": (10, 20, 210, 60),
            "confidence": 0.91,
            "language": "en",
        }
    )

    assert isinstance(region, TextRegion)
    assert region.region_id == "text-a"
    assert region.bbox.to_dict() == {"x": 10, "y": 20, "width": 200, "height": 40}
    assert region.text == "Daitō-ryū Aiki-jūjutsu"


def test_image_extraction_output_maps_to_image_region(tmp_path):
    utility_region = UtilityImageRegion(x=50, y=60, width=70, height=80, region_type="figure", confidence=0.88)
    image_path = tmp_path / "crop.png"
    image_path.write_bytes(b"fake")

    region = image_region_from_extraction(
        {
            "region_id": "image-a",
            "image_path": str(image_path),
            "region": utility_region,
            "confidence": 0.88,
        }
    )

    assert isinstance(region, ImageRegion)
    assert region.region_type == "diagram"
    assert region.image_path == image_path
    assert region.bbox.to_dict() == {"x": 50, "y": 60, "width": 70, "height": 80}


def test_extraction_outputs_build_document_result(tmp_path):
    page = page_result_from_extractions(
        page_number=1,
        source_width=300,
        source_height=400,
        raw_text="武道とは何か。\nDaitō-ryū",
        text_items=[{"text": "武道とは何か。", "bbox": (10, 10, 200, 40), "language": "ja"}],
        image_items=[{"image_path": str(tmp_path / "diagram.png"), "bbox": (20, 80, 220, 200), "image_type": "diagram"}],
        confidence=0.9,
    )
    document = document_result_from_extractions(
        source_path=Path("scan.png"),
        document_id=7,
        pages=[page],
        detected_languages=["ja", "en"],
        confidence=0.9,
    )

    assert isinstance(document, DocumentResult)
    assert document.combined_text() == "武道とは何か。\nDaitō-ryū"
    assert document.pages[0].text_regions[0].language == "ja"
    assert document.pages[0].image_regions[0].region_type == "diagram"
    assert document.to_dict()["pages"][0]["width"] == 300
