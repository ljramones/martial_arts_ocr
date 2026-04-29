from pathlib import Path

from martial_arts_ocr.ocr.processor import OCRProcessor
from martial_arts_ocr.pipeline.document_models import DocumentResult


def test_ocr_processor_process_to_document_result_adapts_legacy_output(monkeypatch):
    processor = OCRProcessor.__new__(OCRProcessor)

    def fake_process_document(image_path: str, document_id: int | None = None):
        return {
            "text": "contract text",
            "confidence": 0.88,
            "language_segments": [{"language": "en", "text": "contract text"}],
            "ocr_text_boxes": [
                {
                    "text": "contract",
                    "x": 4,
                    "y": 5,
                    "width": 55,
                    "height": 12,
                    "confidence": 0.91,
                    "engine": "fake_test",
                }
            ],
        }

    monkeypatch.setattr(processor, "process_document", fake_process_document)

    result = processor.process_to_document_result(Path("scan.png"), document_id=12)

    assert isinstance(result, DocumentResult)
    assert result.document_id == 12
    assert result.combined_text() == "contract text"
    assert result.confidence == 0.88
    word_region = next(
        region for region in result.pages[0].text_regions
        if region.metadata.get("ocr_level") == "word"
    )
    assert word_region.text == "contract"
    assert word_region.metadata["source"] == "ocr_engine"
    assert word_region.metadata["engine"] == "fake_test"
    assert result.pages[0].metadata["readable_text"] == "contract"
