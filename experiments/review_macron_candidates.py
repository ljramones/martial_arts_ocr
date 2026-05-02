from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.text.macron_candidates import find_macron_normalization_candidates


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/notebook_outputs/macron_candidate_review"
DEFAULT_SUMMARY_PATHS = [
    REPO_ROOT / "data/notebook_outputs/macron_ocr_eval/summary.json",
    REPO_ROOT / "data/notebook_outputs/macron_ocr_engine_comparison/summary.json",
    REPO_ROOT / "data/notebook_outputs/ocr_text_quality_review/summary.json",
    REPO_ROOT / "data/notebook_outputs/ocr_text_readability_sampling/eng_auto/summary.json",
    REPO_ROOT / "data/notebook_outputs/ocr_text_reading_order_after_line_grouping/eng_auto/summary.json",
    REPO_ROOT / "data/notebook_outputs/document_result_serialization_review/summary.json",
]

TEXT_KEYS = {
    "cleaned_output",
    "cleaned_text",
    "cleaned_text_preview",
    "output",
    "raw_output",
    "raw_text",
    "raw_text_preview",
    "readable_text",
    "readable_text_preview",
    "serialized_text",
    "text",
    "text_txt",
}

FIXTURE_TEXTS = [
    {
        "source_id": "fixture_ascii_variants",
        "text": "Daito-ryu aikijujutsu appears beside koryu budo and old jujutsu notes.",
        "notes": "ASCII variants that should produce review-required macron candidates.",
    },
    {
        "source_id": "fixture_ocr_confusions",
        "text": "OCR saw koryG, bud6, Dait6-rya, d6j6, and aikijGjutsu on one page.",
        "notes": "Known OCR-confusion variants that are glossary-listed only.",
    },
    {
        "source_id": "fixture_canonical_control",
        "text": "koryū budō Daitō-ryū jūjutsu dōjō ryūha sōke iaidō aikijūjutsu",
        "notes": "Already-canonical macron text; should not produce replacement candidates.",
    },
]


@dataclass(frozen=True)
class TextSource:
    source_id: str
    source_type: str
    text: str
    source_path: str | None = None
    field_path: str | None = None
    notes: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan OCR review summaries and fixture strings for review-only macron normalization candidates."
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        default=None,
        help="OCR review summary JSON to scan. Can be repeated. Defaults to known ignored review summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored output directory for the macron candidate review summary.",
    )
    parser.add_argument(
        "--no-fixtures",
        action="store_true",
        help="Do not include controlled fixture strings in the scan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = [_resolve(path) for path in args.summary_json] if args.summary_json else DEFAULT_SUMMARY_PATHS
    sources: list[TextSource] = []
    if not args.no_fixtures:
        sources.extend(_fixture_sources())

    existing_paths = []
    missing_paths = []
    for path in summary_paths:
        if not path.exists():
            missing_paths.append(str(path))
            continue
        existing_paths.append(str(path))
        sources.extend(load_text_sources_from_json_file(path))

    reviewed_sources = review_text_sources(sources)
    candidate_sources = [source for source in reviewed_sources if source["candidate_count"] > 0]
    candidates = [
        {**candidate, "source_id": source["source_id"], "source_type": source["source_type"]}
        for source in candidate_sources
        for candidate in source["candidates"]
    ]

    payload = {
        "input_summary_paths": [str(path) for path in summary_paths],
        "existing_summary_paths": existing_paths,
        "missing_summary_paths": missing_paths,
        "sources_scanned": len(reviewed_sources),
        "sources_with_candidates": len(candidate_sources),
        "candidate_count": len(candidates),
        "candidate_summary": summarize_candidates(candidates),
        "candidate_sources": candidate_sources,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Scanned {payload['sources_scanned']} text sources; "
        f"found {payload['candidate_count']} candidates in {payload['sources_with_candidates']} sources."
    )
    print(f"Summary: {summary_path}")
    return 0


def load_text_sources_from_json_file(path: Path) -> list[TextSource]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sources: list[TextSource] = []
    for field_path, text in _extract_text_values(data):
        sources.append(
            TextSource(
                source_id=f"{path.parent.name}:{field_path}",
                source_type="summary_json",
                source_path=str(path),
                field_path=field_path,
                text=text,
            )
        )
    return _deduplicate_sources(sources)


def review_text_sources(sources: list[TextSource]) -> list[dict[str, Any]]:
    reviewed = []
    for source in sources:
        candidates = find_macron_normalization_candidates(source.text)
        reviewed.append(
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_path": source.source_path,
                "field_path": source.field_path,
                "notes": source.notes,
                "text_preview": _preview(source.text),
                "candidate_count": len(candidates),
                "candidates": [candidate.to_dict() for candidate in candidates],
            }
        )
    return reviewed


def summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate = Counter(candidate["candidate"] for candidate in candidates)
    by_match_type = Counter(candidate["match_type"] for candidate in candidates)
    by_source_type = Counter(candidate["source_type"] for candidate in candidates)
    return {
        "by_candidate": dict(sorted(by_candidate.items())),
        "by_match_type": dict(sorted(by_match_type.items())),
        "by_source_type": dict(sorted(by_source_type.items())),
    }


def _fixture_sources() -> list[TextSource]:
    return [
        TextSource(
            source_id=fixture["source_id"],
            source_type="fixture",
            text=fixture["text"],
            notes=fixture["notes"],
        )
        for fixture in FIXTURE_TEXTS
    ]


def _extract_text_values(data: Any, path: str = "$") -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}"
            if key in TEXT_KEYS:
                values.extend(_coerce_text_values(value, child_path))
            values.extend(_extract_text_values(value, child_path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            values.extend(_extract_text_values(value, f"{path}[{index}]"))
    return values


def _coerce_text_values(value: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(value, str) and value.strip():
        return [(path, value)]
    if isinstance(value, list):
        values = []
        for index, item in enumerate(value):
            if isinstance(item, str) and item.strip():
                values.append((f"{path}[{index}]", item))
        return values
    return []


def _deduplicate_sources(sources: list[TextSource]) -> list[TextSource]:
    seen: set[tuple[str | None, str]] = set()
    deduplicated: list[TextSource] = []
    for source in sources:
        key = (source.source_path, source.text)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(source)
    return deduplicated


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
