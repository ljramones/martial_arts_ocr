from pathlib import Path

from martial_arts_ocr.pipeline.adapters import document_result_from_ocr_output


class ObjectOCRResult:
    text = "object text"
    confidence = 0.8


class ObjectPage:
    page_number = 2
    text = "object page"
    confidence = 0.7


class ObjectPagedResult:
    pages = [ObjectPage()]
    detected_languages = ["en"]
    confidence = 0.75


class BestOCRResult:
    text = "best text"
    engine = "fake"
    bounding_boxes = []
    metadata = {"source": "test"}

    def to_dict(self):
        return {"text": self.text, "engine": self.engine, "metadata": self.metadata}


class CurrentProcessingShape:
    cleaned_text = "cleaned text"
    raw_text = "raw text"
    best_ocr_result = BestOCRResult()
    text_regions = []
    image_regions = []
    extracted_images = []
    language_segments = [{"language": "en", "text": "cleaned text"}]
    overall_confidence = 0.92
    processing_metadata = {"image_dimensions": {"width": 10, "height": 20}}

    def to_dict(self):
        return {
            "cleaned_text": self.cleaned_text,
            "overall_confidence": self.overall_confidence,
            "best_ocr_result": self.best_ocr_result.to_dict(),
        }


def test_adapter_handles_simple_dict_output():
    result = document_result_from_ocr_output(
        {"text": "hello", "confidence": 0.9},
        document_id=1,
        source_path=Path("scan.png"),
    )

    assert result.document_id == 1
    assert result.confidence == 0.9
    assert result.combined_text() == "hello"
    assert result.pages[0].text_regions[0].text == "hello"


def test_adapter_handles_paged_dict_output():
    result = document_result_from_ocr_output(
        {"pages": [{"text": "page one"}, {"text": "page two"}]},
        document_id=2,
        source_path=Path("scan.png"),
    )

    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.combined_text() == "page one\n\npage two"


def test_adapter_handles_object_with_text():
    result = document_result_from_ocr_output(
        ObjectOCRResult(),
        document_id=3,
        source_path=Path("scan.png"),
    )

    assert result.combined_text() == "object text"
    assert result.confidence == 0.8


def test_adapter_handles_object_with_pages():
    result = document_result_from_ocr_output(
        ObjectPagedResult(),
        document_id=4,
        source_path=Path("scan.png"),
    )

    assert result.detected_languages == ["en"]
    assert result.pages[0].page_number == 2
    assert result.combined_text() == "object page"


def test_adapter_handles_current_processing_result_shape():
    result = document_result_from_ocr_output(
        CurrentProcessingShape(),
        document_id=5,
        source_path=Path("scan.png"),
    )

    assert result.confidence == 0.92
    assert result.detected_languages == ["en"]
    assert result.pages[0].width == 10
    assert result.pages[0].height == 20
    assert result.combined_text() == "cleaned text"


def test_adapter_promotes_best_ocr_bounding_boxes_to_text_regions():
    class BoxedBestOCRResult:
        text = "boxed text"
        engine = "fake_test"
        bounding_boxes = [
            {"text": "boxed", "x": 10, "y": 12, "width": 34, "height": 11, "confidence": 0.8}
        ]
        metadata = {}

    class BoxedProcessingShape(CurrentProcessingShape):
        cleaned_text = "boxed text"
        best_ocr_result = BoxedBestOCRResult()
        ocr_results = []

    result = document_result_from_ocr_output(
        BoxedProcessingShape(),
        document_id=6,
        source_path=Path("scan.png"),
    )

    text_region = result.pages[0].text_regions[0]
    assert text_region.text == "boxed"
    assert text_region.bbox.to_dict() == {"x": 10, "y": 12, "width": 34, "height": 11}
    assert text_region.metadata["source"] == "ocr_engine"
    assert text_region.metadata["engine"] == "fake_test"
