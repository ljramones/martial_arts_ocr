from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from martial_arts_ocr.pipeline.document_models import DocumentResult, PageResult
from martial_arts_ocr.pipeline.extraction_service import ExtractionService, ExtractionServiceOptions
from martial_arts_ocr.pipeline.orchestrator import WorkflowOrchestrator
from martial_arts_ocr.pipeline.result_models import PipelineRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/runtime/review_mode"


@dataclass(frozen=True)
class ReviewProcessor:
    """No-OCR processor for manual extraction review."""

    def process_to_document_result(self, image_path: Path, document_id: int | None = None) -> DocumentResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read input image: {image_path}")
        height, width = image.shape[:2]
        return DocumentResult(
            document_id=document_id,
            source_path=image_path,
            pages=[
                PageResult(
                    page_number=1,
                    width=width,
                    height=height,
                    raw_text="Review-mode extraction run; OCR was not executed.",
                    metadata={"review_mode": True, "ocr_executed": False},
                )
            ],
            metadata={"ocr_engine": "review_mode_no_ocr", "review_mode": True},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manual review-mode image-region extraction on one page."
    )
    parser.add_argument("image_path", help="Input page image path.")
    parser.add_argument("--document-id", type=int, default=None, help="Document id to use in artifact names.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/runtime/review_mode/doc_<id>.",
    )
    parser.add_argument(
        "--enable-image-extraction",
        action="store_true",
        help="Run review-mode image-region extraction.",
    )
    parser.add_argument(
        "--enable-paddle-fusion",
        action="store_true",
        help="Enable optional Paddle layout fusion. Requires PaddleOCR to be installed.",
    )
    parser.add_argument(
        "--paddle-model-dir",
        default=None,
        help="Optional Paddle model/cache directory for Paddle fusion.",
    )
    parser.add_argument("--no-crops", action="store_true", help="Do not write image-region crop files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image_path).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    if not image_path.exists():
        raise SystemExit(f"Input image does not exist: {image_path}")

    document_id = args.document_id if args.document_id is not None else int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / f"doc_{document_id}"
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    extraction_service = ExtractionService(
        ExtractionServiceOptions(
            enable_image_regions=args.enable_image_extraction,
            save_crops=not args.no_crops,
            fail_on_extraction_error=False,
            enable_paddle_layout_fusion=args.enable_paddle_fusion,
            paddle_layout_model_dir=args.paddle_model_dir,
        )
    )
    orchestrator = WorkflowOrchestrator(
        processor=ReviewProcessor(),
        extraction_service=extraction_service,
        processed_path_factory=lambda _name: output_dir,
        persist=False,
    )
    result = orchestrator.process_document(
        PipelineRequest(document_id=document_id, image_path=image_path)
    )
    if not result.success:
        print(f"Review-mode extraction failed: {result.error}")
        return 1

    document_result = result.payload
    image_regions = []
    if document_result and document_result.pages:
        image_regions = document_result.pages[0].image_regions
    summary = summarize_regions(image_regions)

    print(f"Output directory: {result.output_dir}")
    print(f"data.json: {result.json_path}")
    print(f"page_1.html: {result.html_path}")
    print(f"text.txt: {result.text_path}")
    print(f"crop output: {output_dir / 'image_regions'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.enable_paddle_fusion:
        fusion = (document_result.metadata.get("image_extraction", {}) if document_result else {}).get(
            "paddle_layout_fusion"
        )
        if fusion and fusion.get("status") != "completed":
            print(f"Paddle fusion status: {fusion.get('status')} ({fusion.get('reason') or fusion.get('error')})")
    return 0


def summarize_regions(image_regions: list[Any]) -> dict[str, Any]:
    mixed_count = 0
    needs_review_count = 0
    paddle_fused_count = 0
    for region in image_regions:
        metadata = getattr(region, "metadata", {}) or {}
        if metadata.get("mixed_region"):
            mixed_count += 1
        if metadata.get("needs_review"):
            needs_review_count += 1
        if metadata.get("layout_fusion_applied"):
            paddle_fused_count += 1
    return {
        "image_region_count": len(image_regions),
        "mixed_region_count": mixed_count,
        "needs_review_count": needs_review_count,
        "paddle_fused_count": paddle_fused_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
