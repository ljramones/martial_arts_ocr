from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from martial_arts_ocr.pipeline.document_models import BoundingBox, DocumentResult, PageResult, TextRegion
from martial_arts_ocr.pipeline.extraction_service import ExtractionService, ExtractionServiceOptions
from utils.image.regions.core_types import ImageRegion as UtilityImageRegion


class FakeLayoutAnalyzer:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.seen_ocr_text_boxes = None

    def detect_image_regions_with_diagnostics(self, image, ocr_text_boxes=None):
        self.seen_ocr_text_boxes = ocr_text_boxes
        if self.fail:
            raise RuntimeError("detector failed")
        return {
            "accepted_regions": [
                UtilityImageRegion(x=20, y=24, width=80, height=70, region_type="diagram", confidence=0.91)
            ],
            "accepted": [],
            "rejected": [{"region": {"bbox": (1, 2, 3, 4)}, "rejection_reason": "text_like_components"}],
            "consolidation": [{"reason": "overlap_merge"}],
        }


def _document(image_path: Path) -> DocumentResult:
    return DocumentResult(
        document_id=7,
        source_path=image_path,
        pages=[PageResult(page_number=1, raw_text="sample text", confidence=0.9)],
        confidence=0.9,
    )


def _write_synthetic_image(path: Path) -> None:
    image = np.full((180, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 24), (100, 94), (0, 0, 0), 3)
    cv2.imwrite(str(path), image)


def test_disabled_extraction_returns_document_unchanged(tmp_path):
    image_path = tmp_path / "scan.png"
    _write_synthetic_image(image_path)
    document = _document(image_path)
    service = ExtractionService(ExtractionServiceOptions(enable_image_regions=False))

    result = service.enrich_document_result(document, output_dir=tmp_path / "processed")

    assert result is document
    assert result.pages[0].image_regions == []


def test_enabled_extraction_adds_image_regions_and_saves_crops(tmp_path):
    image_path = tmp_path / "scan.png"
    output_dir = tmp_path / "processed" / "doc_7"
    _write_synthetic_image(image_path)
    service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True),
        layout_analyzer_factory=lambda: FakeLayoutAnalyzer(),
    )

    result = service.enrich_document_result(_document(image_path), output_dir=output_dir)

    regions = result.pages[0].image_regions
    assert len(regions) == 1
    assert regions[0].bbox.to_dict() == {"x": 20, "y": 24, "width": 80, "height": 70}
    assert regions[0].image_path.exists()
    assert regions[0].image_path.parent == output_dir / "image_regions"
    assert result.metadata["image_extraction"]["status"] == "completed"
    assert result.metadata["image_extraction"]["rejected"][0]["rejection_reason"] == "text_like_components"
    assert result.pages[0].metadata["image_extraction"]["accepted_count"] == 1


def test_extraction_error_is_recorded_without_failing_by_default(tmp_path):
    image_path = tmp_path / "scan.png"
    _write_synthetic_image(image_path)
    service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True, fail_on_extraction_error=False),
        layout_analyzer_factory=lambda: FakeLayoutAnalyzer(fail=True),
    )

    result = service.enrich_document_result(_document(image_path), output_dir=tmp_path / "processed")

    assert result.pages[0].image_regions == []
    assert result.metadata["image_extraction"]["status"] == "failed"
    assert "detector failed" in result.metadata["image_extraction"]["error"]


def test_extraction_error_can_be_configured_to_raise(tmp_path):
    image_path = tmp_path / "scan.png"
    _write_synthetic_image(image_path)
    service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True, fail_on_extraction_error=True),
        layout_analyzer_factory=lambda: FakeLayoutAnalyzer(fail=True),
    )

    with pytest.raises(RuntimeError, match="detector failed"):
        service.enrich_document_result(_document(image_path), output_dir=tmp_path / "processed")


def test_extraction_service_passes_available_text_region_boxes(tmp_path):
    image_path = tmp_path / "scan.png"
    _write_synthetic_image(image_path)
    fake_analyzer = FakeLayoutAnalyzer()
    document = DocumentResult(
        document_id=7,
        source_path=image_path,
        pages=[
            PageResult(
                page_number=1,
                raw_text="sample text",
                text_regions=[
                    TextRegion(
                        region_id="text_1",
                        text="sample text",
                        bbox=BoundingBox(x=10, y=12, width=90, height=18),
                        confidence=0.93,
                    )
                ],
            )
        ],
    )
    service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True, save_crops=False),
        layout_analyzer_factory=lambda: fake_analyzer,
    )

    result = service.enrich_document_result(document, output_dir=tmp_path / "processed")

    assert fake_analyzer.seen_ocr_text_boxes
    assert fake_analyzer.seen_ocr_text_boxes[0]["bbox"] == {"x": 10, "y": 12, "width": 90, "height": 18}
    assert result.metadata["image_extraction"]["ocr_text_boxes_used"] is True


def test_extraction_service_prefers_engine_boxes_over_non_engine_text_regions(tmp_path):
    image_path = tmp_path / "scan.png"
    _write_synthetic_image(image_path)
    fake_analyzer = FakeLayoutAnalyzer()
    document = DocumentResult(
        document_id=7,
        source_path=image_path,
        pages=[
            PageResult(
                page_number=1,
                text_regions=[
                    TextRegion(
                        region_id="layout_text",
                        text="layout text",
                        bbox=BoundingBox(x=1, y=1, width=20, height=10),
                        metadata={"source": "layout_detector"},
                    )
                ],
                metadata={
                    "ocr_text_boxes": [
                        {
                            "text": "engine text",
                            "bbox": {"x": 30, "y": 40, "width": 50, "height": 12},
                            "metadata": {"source": "ocr_engine", "engine": "fake_test", "ocr_level": "line"},
                        }
                    ]
                },
            )
        ],
    )
    service = ExtractionService(
        ExtractionServiceOptions(enable_image_regions=True, save_crops=False),
        layout_analyzer_factory=lambda: fake_analyzer,
    )

    service.enrich_document_result(document, output_dir=tmp_path / "processed")

    assert len(fake_analyzer.seen_ocr_text_boxes) == 1
    assert fake_analyzer.seen_ocr_text_boxes[0]["text"] == "engine text"
