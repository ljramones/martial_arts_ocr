from pathlib import Path

from martial_arts_ocr.pipeline.adapters import document_result_from_ocr_output, ocr_text_regions_from_ocr_output
from martial_arts_ocr.pipeline.document_models import PageResult
from martial_arts_ocr.pipeline.text_normalization import (
    build_readable_page_text,
    compact_ocr_metadata_for_page,
    group_word_regions_into_lines,
)
from fixtures.ocr_text_samples import (
    EASYOCR_LIKE_POLYGON_ROWS,
    ENGLISH_PARAGRAPH_WORDS,
    MIXED_MARTIAL_ARTS_WORDS,
    TESSERACT_LIKE_ROWS,
)


def test_tesseract_like_word_boxes_group_into_readable_lines():
    word_regions = ocr_text_regions_from_ocr_output({"words": ENGLISH_PARAGRAPH_WORDS}, engine="fake_test")

    lines = group_word_regions_into_lines(word_regions)

    assert [line.text for line in lines] == [
        "Donn Draeger studied",
        "classical bujutsu.",
    ]
    assert lines[0].bbox.to_dict() == {"x": 10, "y": 12, "width": 138, "height": 12}
    assert lines[0].metadata["source"] == "ocr_normalization"
    assert lines[0].metadata["ocr_level"] == "line"
    assert lines[0].metadata["derived_from_count"] == 3


def test_mixed_english_japanese_macrons_and_punctuation_survive_line_grouping():
    word_regions = ocr_text_regions_from_ocr_output({"words": MIXED_MARTIAL_ARTS_WORDS}, engine="fake_test")

    readable = build_readable_page_text(group_word_regions_into_lines(word_regions))

    for token in ["武道", "柔術", "koryū", "budō", "Daitō-ryū", "ō", "ū", "—", "・", "「", "」"]:
        assert token in readable
    assert readable.splitlines() == [
        "武道 とは何か。",
        "Daitō-ryū Aiki-jūjutsu",
        "「柔術」・budō — koryū",
    ]


def test_line_text_orders_words_left_to_right_within_line():
    word_regions = ocr_text_regions_from_ocr_output(
        {
            "words": [
                {"text": "second", "x": 80, "y": 10, "width": 40, "height": 10, "confidence": 0.8},
                {"text": "first", "x": 10, "y": 11, "width": 35, "height": 10, "confidence": 0.8},
            ]
        },
        engine="fake_test",
    )

    lines = group_word_regions_into_lines(word_regions)

    assert lines[0].text == "first second"


def test_document_adapter_preserves_word_boxes_and_adds_readable_line_regions():
    result = document_result_from_ocr_output(
        {
            "text": "Donn Draeger studied\nclassical bujutsu.",
            "confidence": 0.9,
            "words": ENGLISH_PARAGRAPH_WORDS,
        },
        document_id=10,
        source_path=Path("scan.png"),
    )

    page = result.pages[0]
    word_regions = [region for region in page.text_regions if region.metadata.get("ocr_level") == "word"]
    line_regions = [region for region in page.text_regions if region.metadata.get("ocr_level") == "line"]

    assert len(word_regions) == 5
    assert len(line_regions) == 2
    assert page.metadata["readable_text"] == "Donn Draeger studied\nclassical bujutsu."
    assert page.metadata["ocr_word_count"] == 5
    assert page.metadata["ocr_line_count"] == 2
    assert "ocr_text_boxes" in page.metadata


def test_compact_page_metadata_summarizes_hierarchy_without_dropping_words():
    result = document_result_from_ocr_output(
        {"text": "Donn Draeger", "confidence": 0.9, "words": ENGLISH_PARAGRAPH_WORDS[:2]},
        document_id=11,
        source_path=Path("scan.png"),
    )

    summary = compact_ocr_metadata_for_page(result.pages[0])

    assert summary == {
        "readable_text": "Donn Draeger",
        "ocr_word_count": 2,
        "ocr_line_count": 1,
    }
    assert any(region.metadata.get("ocr_level") == "word" for region in result.pages[0].text_regions)


def test_easyocr_polygon_rows_still_normalize_into_rectangular_regions_and_lines():
    word_regions = ocr_text_regions_from_ocr_output({"lines": EASYOCR_LIKE_POLYGON_ROWS}, engine="easyocr")

    assert word_regions[0].bbox.to_dict() == {"x": 10, "y": 10, "width": 72, "height": 14}
    assert word_regions[0].metadata["polygon"] == EASYOCR_LIKE_POLYGON_ROWS[0][0]

    page = PageResult(page_number=1, text_regions=word_regions)
    summary = compact_ocr_metadata_for_page(page)
    assert summary["ocr_word_count"] == 0
