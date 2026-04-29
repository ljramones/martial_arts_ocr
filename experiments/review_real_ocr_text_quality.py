from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from martial_arts_ocr.ocr.processor import OCRProcessor
from martial_arts_ocr.ocr.models import OCRResult
from martial_arts_ocr.pipeline.orchestrator import WorkflowOrchestrator
from martial_arts_ocr.pipeline.result_models import PipelineRequest


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/notebook_outputs/ocr_text_quality_review"
DEFAULT_SAMPLES = [
    ("dfd", "original_img_3337", "text-heavy"),
    ("dfd", "original_img_3288", "mixed English/Japanese"),
    ("dfd", "original_img_3344", "diagram/labeled"),
    ("dfd", "original_img_3330", "noisy/odd layout"),
    ("corpus2", "corpus2_new_doc_2026_04_28_16_56_38", "text-heavy"),
    ("corpus2", "corpus2_new_doc_2026_04_28_18_29_28", "photo/visual"),
    ("corpus2", "corpus2_new_doc_2026_04_28_16_55_48", "broad/mixed"),
    ("corpus2", "corpus2_new_doc_2026_04_28_20_26_02", "noisy/odd layout"),
]
MANIFESTS = {
    "dfd": REPO_ROOT / "data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json",
    "corpus2": REPO_ROOT / "data/corpora/ad_hoc/corpus2/manifests/manifest.local.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected real pages through OCR and summarize text hierarchy output."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored output directory for review artifacts.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Specific sample id to run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Tesseract language override for this review run, e.g. eng or eng+jpn.",
    )
    parser.add_argument(
        "--psm",
        action="append",
        default=None,
        help="Limit full-page Tesseract OCR to one or more PSM values. Can be passed multiple times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples_by_id = _load_samples()
    selected = _select_samples(samples_by_id, args.sample_id)
    processor = OCRProcessor()
    _configure_processor_for_review(processor, language=args.language, psms=args.psm)
    orchestrator = WorkflowOrchestrator(
        processor=processor,
        processed_path_factory=lambda name: output_root / name,
        persist=False,
    )

    results = []
    for index, sample in enumerate(selected, start=1):
        document_id = 920000 + index
        image_path = (REPO_ROOT / sample["path"]).resolve()
        pipeline_result = orchestrator.process_document(
            PipelineRequest(document_id=document_id, image_path=image_path)
        )
        summary = _summarize_pipeline_result(
            sample,
            pipeline_result,
            requested_language=args.language,
            requested_psms=args.psm,
        )
        results.append(summary)
        print(
            f"{sample['id']}: status={summary['status']} "
            f"words={summary.get('word_region_count')} "
            f"lines={summary.get('line_region_count')} "
            f"readable_len={summary.get('readable_text_length')}"
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


def _load_samples() -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for corpus, manifest_path in MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sample in manifest.get("samples", []):
            record = dict(sample)
            record["corpus"] = corpus
            samples[record["id"]] = record
    return samples


def _select_samples(
    samples_by_id: dict[str, dict[str, Any]],
    sample_ids: list[str] | None,
) -> list[dict[str, Any]]:
    wanted = []
    if sample_ids:
        for sample_id in sample_ids:
            sample = dict(samples_by_id[sample_id])
            sample["page_type"] = "manual selection"
            wanted.append(sample)
        return wanted

    for corpus, sample_id, page_type in DEFAULT_SAMPLES:
        sample = dict(samples_by_id[sample_id])
        sample["corpus"] = corpus
        sample["page_type"] = page_type
        wanted.append(sample)
    return wanted


def _configure_processor_for_review(
    processor: OCRProcessor,
    *,
    language: str | None,
    psms: list[str] | None,
) -> None:
    if language and getattr(processor.tesseract_engine, "available", False):
        processor.tesseract_engine.config = dict(processor.tesseract_engine.config)
        processor.tesseract_engine.config["languages"] = language.split("+")

    if not psms:
        return

    def _run_review_psms(self: OCRProcessor, image: Any) -> list[OCRResult]:
        ocr_results: list[OCRResult] = []
        if self.tesseract_engine.available:
            for psm in psms:
                try:
                    result = self.tesseract_engine.process_image(image, psm=psm)
                    result.metadata = {
                        **(result.metadata or {}),
                        "full_page": True,
                        "psm": str(psm),
                        "review_psm_override": True,
                    }
                    ocr_results.append(result)
                except Exception as exc:
                    print(f"Tesseract PSM {psm} failed: {exc}")
        if not ocr_results:
            ocr_results.append(
                OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    engine="none",
                    bounding_boxes=[],
                    metadata={"review_psm_override": list(psms)},
                )
            )
        return ocr_results

    processor._run_full_page_ocr = MethodType(_run_review_psms, processor)


def _summarize_pipeline_result(
    sample: dict[str, Any],
    pipeline_result: Any,
    *,
    requested_language: str | None,
    requested_psms: list[str] | None,
) -> dict[str, Any]:
    summary = {
        "corpus": sample["corpus"],
        "sample_id": sample["id"],
        "path": sample["path"],
        "page_type": sample.get("page_type"),
        "requested_language": requested_language,
        "requested_psms": requested_psms,
        "status": pipeline_result.status,
        "output_dir": str(pipeline_result.output_dir),
        "data_json": str(pipeline_result.json_path) if pipeline_result.json_path else None,
        "page_html": str(pipeline_result.html_path) if pipeline_result.html_path else None,
        "text_txt": str(pipeline_result.text_path) if pipeline_result.text_path else None,
        "error": pipeline_result.error,
    }
    document_result = pipeline_result.payload
    if not document_result or not document_result.pages:
        return summary

    page = document_result.pages[0]
    word_regions = [
        region for region in page.text_regions
        if (region.metadata or {}).get("ocr_level") == "word"
    ]
    line_regions = [
        region for region in page.text_regions
        if (region.metadata or {}).get("ocr_level") == "line"
    ]
    legacy = document_result.metadata.get("legacy", {})
    best_result = legacy.get("best_ocr_result", {}) if isinstance(legacy, dict) else {}
    best_metadata = best_result.get("metadata", {}) if isinstance(best_result, dict) else {}
    raw_text = str(legacy.get("raw_text", page.raw_text or ""))
    cleaned_text = document_result.combined_text()
    readable_text = str(page.metadata.get("readable_text", ""))

    summary.update(
        {
            "ocr_engine": document_result.metadata.get("ocr_engine"),
            "selected_psm": best_metadata.get("psm", best_metadata.get("psm_used")),
            "selected_language": best_metadata.get("languages"),
            "confidence": document_result.confidence,
            "raw_text_length": len(raw_text),
            "cleaned_text_length": len(cleaned_text),
            "readable_text_length": len(readable_text),
            "word_region_count": len(word_regions),
            "line_region_count": len(line_regions),
            "canonical_word_region_count": len(word_regions),
            "canonical_line_region_count": len(line_regions),
            "metadata_ocr_word_count": page.metadata.get("ocr_word_count"),
            "metadata_ocr_line_count": page.metadata.get("ocr_line_count"),
            "ocr_text_box_count": len(page.metadata.get("ocr_text_boxes", [])),
            "alternate_candidate_count": len(page.metadata.get("ocr_alternative_candidates", [])),
            "alternate_candidate_word_box_count": sum(
                int(candidate.get("word_box_count") or 0)
                for candidate in page.metadata.get("ocr_alternative_candidates", [])
                if not candidate.get("selected")
            ),
            "alternate_candidates": page.metadata.get("ocr_alternative_candidates", []),
            "raw_text_preview": _first_lines(raw_text),
            "cleaned_text_preview": _first_lines(cleaned_text),
            "readable_text_preview": _first_lines(readable_text),
            "word_samples": [
                {"text": region.text, "bbox": region.bbox.to_dict() if region.bbox else None}
                for region in word_regions[:8]
            ],
            "line_samples": [
                {"text": region.text, "bbox": region.bbox.to_dict() if region.bbox else None}
                for region in line_regions[:6]
            ],
            "has_japanese": any(_contains_japanese(text) for text in [raw_text, cleaned_text, readable_text]),
            "has_macron": any(_contains_macron(text) for text in [raw_text, cleaned_text, readable_text]),
        }
    )
    return summary


def _first_lines(text: str, *, limit: int = 6) -> list[str]:
    return [line for line in text.splitlines()[:limit]]


def _contains_japanese(text: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
        for char in text
    )


def _contains_macron(text: str) -> bool:
    return any(char in text for char in "āēīōūĀĒĪŌŪ")


if __name__ == "__main__":
    raise SystemExit(main())
