"""Optional image-region extraction for canonical document results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from martial_arts_ocr.pipeline.document_models import DocumentResult, PageResult
from martial_arts_ocr.pipeline.extraction_adapters import image_region_from_extraction


@dataclass(frozen=True)
class ExtractionServiceOptions:
    """Runtime options for optional extraction enrichment."""

    enable_image_regions: bool = False
    save_crops: bool = True
    fail_on_extraction_error: bool = False
    crop_subdir: str = "image_regions"
    enable_paddle_layout_fusion: bool = False
    paddle_layout_model_dir: str | None = None


class ExtractionService:
    """Enrich a DocumentResult with utility-detected image regions.

    This service is intentionally opt-in. The detector is useful for review
    workflows, but current real-page review does not support enabling it as a
    silent default for all OCR processing.
    """

    def __init__(
        self,
        options: ExtractionServiceOptions | None = None,
        layout_analyzer_factory=None,
        paddle_strategy_factory=None,
    ):
        self.options = options or ExtractionServiceOptions()
        self._layout_analyzer_factory = layout_analyzer_factory
        self._paddle_strategy_factory = paddle_strategy_factory

    def enrich_document_result(
        self,
        document_result: DocumentResult,
        *,
        output_dir: Path,
    ) -> DocumentResult:
        if not self.options.enable_image_regions:
            return document_result

        try:
            return self._enrich_document_result(document_result, output_dir=output_dir)
        except Exception as exc:
            if self.options.fail_on_extraction_error:
                raise
            metadata = dict(document_result.metadata)
            metadata["image_extraction"] = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
            }
            return replace(document_result, metadata=metadata)

    def _enrich_document_result(self, document_result: DocumentResult, *, output_dir: Path) -> DocumentResult:
        source_path = Path(document_result.source_path)
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read source image for extraction: {source_path}")

        analyzer = self._build_layout_analyzer()
        ocr_text_boxes = self._ocr_text_boxes_from_document(document_result)
        diagnostics = analyzer.detect_image_regions_with_diagnostics(
            image,
            ocr_text_boxes=ocr_text_boxes,
        )
        accepted_regions = diagnostics.get("accepted_regions", [])
        paddle_fusion = None
        if self.options.enable_paddle_layout_fusion:
            accepted_regions, paddle_fusion = self._fuse_paddle_layout(
                image,
                accepted_regions,
            )

        crop_records = []
        crop_dir = output_dir / self.options.crop_subdir
        if self.options.save_crops:
            from utils.image.ops.extract import save_region_crops

            crop_records = save_region_crops(image, accepted_regions, crop_dir, prefix="image_region")
        else:
            crop_records = [
                {
                    "region_id": getattr(region, "id", None) or f"image_region_{index:03d}",
                    "region": region.to_dict() if hasattr(region, "to_dict") else {"bbox": region.bbox},
                    "reading_order": index,
                }
                for index, region in enumerate(accepted_regions, start=1)
            ]

        image_regions = [
            image_region_from_extraction(record, index=index)
            for index, record in enumerate(crop_records, start=1)
        ]

        page = document_result.pages[0] if document_result.pages else PageResult(page_number=1)
        page_metadata = dict(page.metadata)
        page_metadata["image_extraction"] = {
            "enabled": True,
            "status": "completed",
            "accepted_count": len(image_regions),
            "rejected_count": len(diagnostics.get("rejected", [])),
            "consolidation_count": len(diagnostics.get("consolidation", [])),
            "refinement_count": len(diagnostics.get("refinement", [])),
            "raw_candidate_count": len(diagnostics.get("raw_candidates", [])),
            "detector_diagnostics": diagnostics.get("detector_diagnostics", []),
        }
        if paddle_fusion is not None:
            page_metadata["image_extraction"]["paddle_layout_fusion"] = paddle_fusion
        enriched_page = replace(
            page,
            image_regions=list(page.image_regions) + image_regions,
            metadata=page_metadata,
        )
        pages = [enriched_page, *document_result.pages[1:]] if document_result.pages else [enriched_page]

        metadata = dict(document_result.metadata)
        metadata["image_extraction"] = {
            "enabled": True,
            "status": "completed",
            "accepted_count": len(image_regions),
            "accepted": diagnostics.get("accepted", []),
            "rejected": diagnostics.get("rejected", []),
            "consolidation": diagnostics.get("consolidation", []),
            "refinement": diagnostics.get("refinement", []),
            "raw_candidates": diagnostics.get("raw_candidates", []),
            "detector_diagnostics": diagnostics.get("detector_diagnostics", []),
            "crop_dir": str(crop_dir) if self.options.save_crops else None,
            "ocr_text_boxes_used": bool(ocr_text_boxes),
        }
        if paddle_fusion is not None:
            metadata["image_extraction"]["paddle_layout_fusion"] = paddle_fusion
        return replace(document_result, pages=pages, metadata=metadata)

    def _fuse_paddle_layout(self, image, accepted_regions):
        from utils.image.layout.fusion import fuse_paddle_layout_regions

        try:
            strategy = self._build_paddle_strategy()
            layout_result = strategy.detect(image)
            if not layout_result.available:
                return accepted_regions, {
                    "enabled": True,
                    "status": "skipped",
                    "backend": layout_result.strategy_name,
                    "reason": layout_result.skipped_reason,
                }
            fusion_result = fuse_paddle_layout_regions(
                accepted_regions,
                layout_result.regions,
                backend_name=layout_result.strategy_name,
            )
            return fusion_result.regions, {
                "enabled": True,
                "status": "completed",
                "backend": layout_result.strategy_name,
                "layout_region_count": len(layout_result.regions),
                "fusion_event_count": len(fusion_result.events),
                "events": fusion_result.events,
                "layout_metadata": layout_result.metadata,
            }
        except Exception as exc:
            return accepted_regions, {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
            }

    def _build_layout_analyzer(self):
        if self._layout_analyzer_factory is not None:
            return self._layout_analyzer_factory()
        from utils.image.layout.analyzer import LayoutAnalyzer

        return LayoutAnalyzer(
            {
                "use_yolo_figure": False,
                "enabled_detectors": ["figure", "contours", "multi_figure_rows"],
                "contours_always": True,
                "filter_text_like": True,
            }
        )

    def _build_paddle_strategy(self):
        if self._paddle_strategy_factory is not None:
            return self._paddle_strategy_factory()
        from utils.image.layout.strategies import PaddleLayoutStrategy

        config = {}
        if self.options.paddle_layout_model_dir:
            config["model_dir"] = self.options.paddle_layout_model_dir
        return PaddleLayoutStrategy(config)

    @staticmethod
    def _ocr_text_boxes_from_document(document_result: DocumentResult) -> list[dict]:
        """Collect available OCR text geometry without requiring an OCR engine."""
        boxes: list[dict] = []
        metadata_boxes = document_result.metadata.get("ocr_text_boxes")
        if isinstance(metadata_boxes, list):
            boxes.extend(metadata_boxes)

        for page in document_result.pages:
            page_boxes = page.metadata.get("ocr_text_boxes")
            if isinstance(page_boxes, list):
                boxes.extend(page_boxes)
            for region in page.text_regions:
                if not region.bbox:
                    continue
                boxes.append(
                    {
                        "text": region.text,
                        "bbox": region.bbox.to_dict(),
                        "confidence": region.confidence,
                        "language": region.language,
                        "source": region.metadata.get("source", "canonical_text_region"),
                        "engine": region.metadata.get("engine", "unknown"),
                        "ocr_level": region.metadata.get("ocr_level", region.metadata.get("level", "block")),
                    }
                )
        engine_boxes = [
            box
            for box in boxes
            if _metadata_value(box, "source") == "ocr_engine" or _metadata_value(box, "engine") not in {None, "", "unknown"}
        ]
        return engine_boxes or boxes


def _metadata_value(value: dict, key: str):
    if key in value:
        return value.get(key)
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None
