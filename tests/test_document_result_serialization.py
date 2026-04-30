from pathlib import Path

from martial_arts_ocr.pipeline.document_models import BoundingBox, DocumentResult, PageResult, TextRegion


def _region(region_id: str, text: str, level: str, x: int = 1) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        text=text,
        bbox=BoundingBox(x=x, y=2, width=10, height=8),
        metadata={
            "source": "ocr_normalization" if level == "line" else "ocr_engine",
            "ocr_level": level,
            **(
                {
                    "line_grouping_method": "adaptive_center_overlap_v1",
                    "reading_order_uncertain": False,
                }
                if level == "line"
                else {}
            ),
        },
    )


def test_page_serialization_exposes_text_summary_and_region_aliases():
    page = PageResult(
        page_number=1,
        raw_text="raw text",
        text_regions=[
            _region("line_1", "Daitō-ryū line", "line"),
            _region("word_1", "Daitō-ryū", "word"),
            _region("word_2", "line", "word", x=20),
        ],
        metadata={
            "readable_text": "Daitō-ryū line",
            "ocr_word_count": 2,
            "ocr_line_count": 1,
            "ocr_alternative_candidates": [
                {"psm": "3", "confidence": 0.9, "text_length": 20, "word_box_count": 2, "selected": True}
            ],
        },
    )

    serialized = page.to_dict()

    assert serialized["readable_text"] == "Daitō-ryū line"
    assert serialized["text_summary"] == {
        "raw_text": "raw text",
        "readable_text": "Daitō-ryū line",
        "word_count": 2,
        "line_count": 1,
        "line_grouping_method": "adaptive_center_overlap_v1",
        "reading_order_uncertain": False,
    }
    assert [region["text"] for region in serialized["line_regions"]] == ["Daitō-ryū line"]
    assert [region["text"] for region in serialized["word_regions"]] == ["Daitō-ryū", "line"]
    assert len(serialized["text_regions"]) == 3
    assert serialized["metadata"]["ocr_alternative_candidates"][0]["word_box_count"] == 2


def test_document_serialization_exposes_compact_text_summary_without_losing_pages():
    document = DocumentResult(
        document_id=5,
        source_path=Path("scan.png"),
        pages=[
            PageResult(
                page_number=1,
                raw_text="raw 武道",
                text_regions=[
                    _region("line_1", "武道 koryū budō — 「柔術」", "line"),
                    _region("word_1", "武道", "word"),
                ],
                metadata={"readable_text": "武道 koryū budō — 「柔術」", "ocr_word_count": 1, "ocr_line_count": 1},
            )
        ],
    )

    serialized = document.to_dict()

    assert serialized["text_summary"]["page_count"] == 1
    assert serialized["text_summary"]["word_count"] == 1
    assert serialized["text_summary"]["line_count"] == 1
    assert "武道 koryū budō — 「柔術」" in serialized["text_summary"]["readable_text"]
    assert serialized["pages"][0]["raw_text"] == "raw 武道"
    assert serialized["pages"][0]["line_regions"][0]["text"] == "武道 koryū budō — 「柔術」"
