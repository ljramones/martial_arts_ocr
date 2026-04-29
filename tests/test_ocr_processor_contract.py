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
    assert result.pages[0].text_regions[0].text == "contract"
    assert result.pages[0].text_regions[0].metadata["source"] == "ocr_engine"
    assert result.pages[0].text_regions[0].metadata["engine"] == "fake_test"
