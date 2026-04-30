from pathlib import Path

from martial_arts_ocr.pipeline.document_models import (
    BoundingBox,
    DocumentResult,
    ImageRegion,
    PageResult,
    TextRegion,
)
from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor


def _line(region_id: str, text: str, *, uncertain: bool = False) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        text=text,
        bbox=BoundingBox(x=10, y=20, width=180, height=20),
        confidence=0.9,
        metadata={
            "source": "ocr_normalization",
            "ocr_level": "line",
            "line_grouping_method": "adaptive_center_overlap_v1",
            "reading_order_uncertain": uncertain,
        },
    )


def _word(region_id: str, text: str) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        text=text,
        bbox=BoundingBox(x=10, y=20, width=40, height=12),
        confidence=0.8,
        metadata={"source": "ocr_engine", "engine": "tesseract", "ocr_level": "word"},
    )


def test_canonical_reconstruction_uses_line_regions_as_visible_text():
    page = PageResult(
        page_number=1,
        text_regions=[
            _line("line_1", "Visible readable line"),
            _word("word_1", "WORD_ONLY_DEBUG"),
            _word("word_2", "ANOTHER_DEBUG_WORD"),
        ],
        metadata={"ocr_word_count": 2, "ocr_line_count": 1},
    )

    reconstructed = PageReconstructor().reconstruct_page(
        DocumentResult(document_id=7, source_path=Path("scan.png"), pages=[page])
    )

    text_elements = [element for element in reconstructed.elements if element.element_type == "text"]
    assert [element.content for element in text_elements] == ["Visible readable line"]
    assert "Visible readable line" in reconstructed.html_content
    assert "WORD_ONLY_DEBUG" not in reconstructed.html_content
    assert reconstructed.reconstruction_metadata["visible_text_source"] == "line_regions"
    assert reconstructed.reconstruction_metadata["ocr_word_count"] == 2
    assert reconstructed.reconstruction_metadata["ocr_line_count"] == 1


def test_canonical_reconstruction_uses_readable_text_when_lines_are_missing():
    page = PageResult(
        page_number=1,
        text_regions=[_word("word_1", "WORD_ONLY_DEBUG")],
        metadata={"readable_text": "Readable fallback text", "ocr_word_count": 1, "ocr_line_count": 0},
    )

    reconstructed = PageReconstructor().reconstruct_page(
        DocumentResult(document_id=8, source_path=Path("scan.png"), pages=[page])
    )

    assert [element.content for element in reconstructed.elements if element.element_type == "text"] == [
        "Readable fallback text"
    ]
    assert "WORD_ONLY_DEBUG" not in reconstructed.html_content
    assert reconstructed.reconstruction_metadata["visible_text_source"] == "readable_text"


def test_canonical_reconstruction_uses_raw_text_when_readable_text_is_missing():
    page = PageResult(page_number=1, raw_text="Raw OCR fallback")

    reconstructed = PageReconstructor().reconstruct_page(
        DocumentResult(document_id=9, source_path=Path("scan.png"), pages=[page])
    )

    assert [element.content for element in reconstructed.elements if element.element_type == "text"] == [
        "Raw OCR fallback"
    ]
    assert reconstructed.reconstruction_metadata["visible_text_source"] == "raw_text"


def test_canonical_reconstruction_surfaces_reading_order_uncertainty():
    page = PageResult(
        page_number=1,
        text_regions=[_line("line_1", "Uncertain line", uncertain=True), _word("word_1", "Uncertain")],
        metadata={"ocr_word_count": 1, "ocr_line_count": 1},
    )

    reconstructed = PageReconstructor().reconstruct_page(
        DocumentResult(document_id=10, source_path=Path("scan.png"), pages=[page])
    )

    assert reconstructed.reconstruction_metadata["reading_order_uncertain"] is True
    assert "Reading order may be uncertain for this page." in reconstructed.html_content
    text_element = next(element for element in reconstructed.elements if element.element_type == "text")
    assert text_element.metadata["reading_order_uncertain"] is True
    assert text_element.metadata["visible_text_source"] == "line_regions"


def test_canonical_reconstruction_preserves_image_regions():
    page = PageResult(
        page_number=1,
        text_regions=[_line("line_1", "Visible line")],
        image_regions=[
            ImageRegion(
                region_id="image_1",
                image_path=Path("image_regions/image_1.png"),
                bbox=BoundingBox(x=20, y=80, width=100, height=60),
                confidence=0.91,
            )
        ],
    )

    reconstructed = PageReconstructor().reconstruct_page(
        DocumentResult(document_id=11, source_path=Path("scan.png"), pages=[page])
    )

    image_elements = [element for element in reconstructed.elements if element.element_type == "image"]
    assert len(image_elements) == 1
    assert image_elements[0].content == "image_regions/image_1.png"
    assert reconstructed.reconstruction_metadata["image_elements"] == 1
