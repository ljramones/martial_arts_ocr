from pathlib import Path

from martial_arts_ocr.pipeline.document_models import (
    BoundingBox,
    DocumentResult,
    PageResult,
    TextRegion,
)


def test_bounding_box_to_dict():
    assert BoundingBox(x=1, y=2, width=3, height=4).to_dict() == {
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
    }


def test_text_region_to_dict():
    region = TextRegion(
        region_id="r1",
        text="hello",
        bbox=BoundingBox(x=1, y=2, width=3, height=4),
        confidence=0.9,
        language="en",
        reading_order=1,
        metadata={"source": "test"},
    )

    assert region.to_dict() == {
        "region_id": "r1",
        "type": "text",
        "text": "hello",
        "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
        "confidence": 0.9,
        "language": "en",
        "reading_order": 1,
        "metadata": {"source": "test"},
    }


def test_page_result_combined_text_uses_regions_when_raw_text_is_empty():
    page = PageResult(
        page_number=1,
        text_regions=[
            TextRegion(region_id="r1", text="first"),
            TextRegion(region_id="r2", text="second"),
        ],
    )

    assert page.combined_text() == "first\nsecond"


def test_document_result_combined_text_joins_pages():
    document = DocumentResult(
        document_id=5,
        source_path=Path("scan.png"),
        pages=[
            PageResult(page_number=1, raw_text="page one"),
            PageResult(page_number=2, raw_text="page two"),
        ],
    )

    assert document.combined_text() == "page one\n\npage two"
    assert document.to_dict()["source_path"] == "scan.png"


def test_page_reconstructor_accepts_page_result():
    from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor

    reconstructed = PageReconstructor().reconstruct_page(
        PageResult(page_number=1, raw_text="canonical text")
    )

    assert reconstructed.page_id == "page_1"
    assert reconstructed.elements[0].content == "canonical text"
    assert "canonical text" in reconstructed.html_content
