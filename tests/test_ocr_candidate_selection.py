from pathlib import Path

from martial_arts_ocr.pipeline.adapters import document_result_from_ocr_output


def _box(text: str, x: int, y: int) -> dict:
    return {
        "text": text,
        "x": x,
        "y": y,
        "width": 20,
        "height": 10,
        "confidence": 0.9,
        "engine": "fake_test",
    }


def test_best_ocr_result_boxes_are_canonical_when_alternates_exist():
    best = {
        "text": "best line",
        "engine": "tesseract",
        "confidence": 0.9,
        "metadata": {"psm": "3"},
        "bounding_boxes": [_box("best", 10, 10), _box("line", 36, 10)],
    }
    alternate = {
        "text": "alternate line",
        "engine": "tesseract",
        "confidence": 0.8,
        "metadata": {"psm": "11"},
        "bounding_boxes": [_box("alternate", 10, 10), _box("line", 80, 10)],
    }

    result = document_result_from_ocr_output(
        {
            "cleaned_text": "best line",
            "best_ocr_result": best,
            "ocr_results": [alternate, best],
        },
        document_id=1,
        source_path=Path("scan.png"),
    )
    page = result.pages[0]
    word_regions = [
        region for region in page.text_regions
        if region.metadata.get("ocr_level") == "word"
    ]
    line_regions = [
        region for region in page.text_regions
        if region.metadata.get("ocr_level") == "line"
    ]

    assert [region.text for region in word_regions] == ["best", "line"]
    assert [region.text for region in line_regions] == ["best line"]
    assert page.metadata["readable_text"] == "best line"
    assert "alternate" not in page.metadata["readable_text"]
    assert page.metadata["ocr_word_count"] == 2
    assert page.metadata["ocr_line_count"] == 1


def test_alternate_psm_candidates_are_preserved_as_compact_metadata_only():
    best = {
        "text": "best line",
        "engine": "tesseract",
        "confidence": 0.9,
        "metadata": {"psm": "3"},
        "bounding_boxes": [_box("best", 10, 10), _box("line", 36, 10)],
    }
    alternate = {
        "text": "alternate line",
        "engine": "tesseract",
        "confidence": 0.8,
        "metadata": {"psm": "11"},
        "bounding_boxes": [_box("alternate", 10, 10), _box("line", 80, 10)],
    }

    result = document_result_from_ocr_output(
        {
            "text": "best line",
            "best_ocr_result": best,
            "ocr_results": [alternate, best],
        },
        document_id=2,
        source_path=Path("scan.png"),
    )
    metadata = result.pages[0].metadata

    assert metadata["ocr_text_boxes"]
    assert all("alternate" not in region["text"] for region in metadata["ocr_text_boxes"])
    assert metadata["ocr_alternative_candidates"] == [
        {
            "engine": "tesseract",
            "psm": "11",
            "confidence": 0.8,
            "text_length": len("alternate line"),
            "word_box_count": 2,
            "selected": False,
        },
        {
            "engine": "tesseract",
            "psm": "3",
            "confidence": 0.9,
            "text_length": len("best line"),
            "word_box_count": 2,
            "selected": True,
        },
    ]


def test_top_level_bounding_boxes_still_work_for_simple_legacy_output():
    result = document_result_from_ocr_output(
        {
            "text": "legacy line",
            "confidence": 0.85,
            "bounding_boxes": [_box("legacy", 10, 10), _box("line", 42, 10)],
        },
        document_id=3,
        source_path=Path("scan.png"),
    )
    page = result.pages[0]

    assert page.metadata["readable_text"] == "legacy line"
    assert page.metadata["ocr_word_count"] == 2
    assert [
        region.text for region in page.text_regions
        if region.metadata.get("ocr_level") == "word"
    ] == ["legacy", "line"]


def test_object_like_best_ocr_result_uses_selected_boxes_only():
    class OCRCandidate:
        def __init__(self, text, psm, boxes):
            self.text = text
            self.engine = "tesseract"
            self.confidence = 0.9 if psm == "6" else 0.7
            self.metadata = {"psm": psm}
            self.bounding_boxes = boxes

    class Output:
        cleaned_text = "chosen output"
        best_ocr_result = OCRCandidate("chosen output", "6", [_box("chosen", 10, 10), _box("output", 45, 10)])
        ocr_results = [
            OCRCandidate("other output", "11", [_box("other", 10, 10), _box("output", 45, 10)]),
            best_ocr_result,
        ]

    result = document_result_from_ocr_output(
        Output(),
        document_id=4,
        source_path=Path("scan.png"),
    )
    page = result.pages[0]

    assert page.metadata["readable_text"] == "chosen output"
    assert [
        region.text for region in page.text_regions
        if region.metadata.get("ocr_level") == "word"
    ] == ["chosen", "output"]
    assert page.metadata["ocr_alternative_candidates"][0]["selected"] is False
    assert page.metadata["ocr_alternative_candidates"][1]["selected"] is True
