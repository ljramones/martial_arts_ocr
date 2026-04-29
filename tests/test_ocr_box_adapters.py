from __future__ import annotations

from pathlib import Path

from martial_arts_ocr.pipeline.adapters import (
    document_result_from_ocr_output,
    ocr_text_boxes_from_ocr_output,
    ocr_text_regions_from_ocr_output,
)


def test_tesseract_like_tsv_rows_convert_to_text_regions():
    rows = [
        {"left": 10, "top": 20, "width": 30, "height": 12, "conf": "95", "text": "Donn", "level": 5},
        {"left": 45, "top": 20, "width": 45, "height": 12, "conf": "91", "text": "Draeger", "level": "word"},
    ]

    regions = ocr_text_regions_from_ocr_output({"words": rows}, engine="tesseract")

    assert len(regions) == 2
    assert regions[0].bbox.to_dict() == {"x": 10, "y": 20, "width": 30, "height": 12}
    assert regions[0].confidence == 0.95
    assert regions[0].metadata == {"source": "ocr_engine", "engine": "tesseract", "ocr_level": "word"}


def test_tesseract_output_dict_converts_to_boxes():
    output = {
        "bounding_boxes": {
            "left": [5],
            "top": [6],
            "width": [20],
            "height": [8],
            "conf": [88],
            "text": ["武道"],
            "level": [5],
            "engine": "tesseract",
        }
    }

    boxes = ocr_text_boxes_from_ocr_output(output)

    assert boxes == [
        {
            "text": "武道",
            "x": 5,
            "y": 6,
            "width": 20,
            "height": 8,
            "confidence": 0.88,
            "language": None,
            "source": "ocr_engine",
            "engine": "tesseract",
            "ocr_level": "word",
            "bbox_convention": "xywh",
        }
    ]


def test_easyocr_like_polygon_output_converts_to_rectangular_bbox():
    easyocr_output = {
        "lines": [
            ([(10, 10), (60, 12), (58, 35), (8, 33)], "Daitō-ryū", 0.84),
        ]
    }

    regions = ocr_text_regions_from_ocr_output(easyocr_output, engine="easyocr")

    assert len(regions) == 1
    assert regions[0].text == "Daitō-ryū"
    assert regions[0].bbox.to_dict() == {"x": 8, "y": 10, "width": 52, "height": 25}
    assert regions[0].metadata["engine"] == "easyocr"
    assert regions[0].metadata["ocr_level"] == "line"
    assert "polygon" in regions[0].metadata


def test_words_and_lines_shapes_are_preserved_with_provenance():
    output = {
        "words": [{"text": "koryū", "x": 1, "y": 2, "width": 30, "height": 10, "confidence": 0.9}],
        "lines": [{"text": "budō line", "x": 1, "y": 22, "width": 80, "height": 12, "confidence": 0.8, "level": "line"}],
    }

    regions = ocr_text_regions_from_ocr_output(output, engine="fake_test")

    assert [region.text for region in regions] == ["koryū", "budō line"]
    assert [region.metadata["ocr_level"] for region in regions] == ["word", "line"]
    assert all(region.metadata["source"] == "ocr_engine" for region in regions)
    assert all(region.metadata["engine"] == "fake_test" for region in regions)


def test_document_adapter_preserves_existing_ocr_text_boxes_metadata():
    result = document_result_from_ocr_output(
        {
            "text": "hello",
            "confidence": 0.9,
            "ocr_text_boxes": [
                {"text": "hello", "x": 5, "y": 6, "width": 40, "height": 11, "confidence": 0.87, "engine": "fake_test"}
            ],
        },
        document_id=1,
        source_path=Path("scan.png"),
    )

    word_regions = [
        region for region in result.pages[0].text_regions
        if region.metadata.get("ocr_level") == "word"
    ]
    line_regions = [
        region for region in result.pages[0].text_regions
        if region.metadata.get("ocr_level") == "line"
    ]
    assert len(word_regions) == 1
    assert len(line_regions) == 1
    assert word_regions[0].metadata["engine"] == "fake_test"
    assert result.pages[0].metadata["ocr_text_boxes"][0]["metadata"]["source"] == "ocr_engine"
