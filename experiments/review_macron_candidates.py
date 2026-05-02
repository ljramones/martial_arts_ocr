from __future__ import annotations

import argparse
import csv
import hashlib
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

FILTER_CHOICES = ("all", "pending", "accepted", "rejected", "deferred", "edited", "reviewed", "stale")
SORT_CHOICES = ("source", "candidate", "observed", "decision", "match_type")
SOURCE_FILTER_CHOICES = ("all", "fixture", "summary_json", "real_ocr", "synthetic", "macron_eval")

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
    parser.add_argument(
        "--decisions-file",
        default=None,
        help=(
            "Local ignored decisions file. Defaults to decisions.local.json in the output directory. "
            "The helper creates a template when the file is missing."
        ),
    )
    parser.add_argument(
        "--reviewed-export",
        default=None,
        help="Ignored reviewed-decision export path. Defaults to reviewed_decisions.json in the output directory.",
    )
    parser.add_argument(
        "--overwrite-decisions-template",
        action="store_true",
        help="Overwrite an existing local decisions template. Existing files are preserved by default.",
    )
    parser.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default="all",
        help="Candidate queue filter for Markdown/CSV review exports.",
    )
    parser.add_argument(
        "--source-filter",
        choices=SOURCE_FILTER_CHOICES,
        default="all",
        help="Source filter for Markdown/CSV review queue exports.",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_CHOICES,
        default="source",
        help="Sort key for Markdown/CSV review queue exports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum candidates to include in the review queue exports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = _resolve(args.decisions_file) if args.decisions_file else output_dir / "decisions.local.json"
    reviewed_export_path = (
        _resolve(args.reviewed_export) if args.reviewed_export else output_dir / "reviewed_decisions.json"
    )

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
        "decisions_file": str(decisions_path),
        "reviewed_export": str(reviewed_export_path),
        "sources_scanned": len(reviewed_sources),
        "sources_with_candidates": len(candidate_sources),
        "candidate_count": len(candidates),
        "candidate_summary": summarize_candidates(candidates),
        "candidate_sources": candidate_sources,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    decisions_payload = build_decisions_template(candidate_sources)
    decisions_template_written = write_decisions_template(
        decisions_path,
        decisions_payload,
        overwrite=args.overwrite_decisions_template,
    )
    decisions_payload = load_decisions_file(decisions_path)
    reviewed_export = build_reviewed_export(candidate_sources, decisions_payload)
    reviewed_export_path.write_text(json.dumps(reviewed_export, ensure_ascii=False, indent=2), encoding="utf-8")
    queue = build_review_queue(
        candidate_sources,
        decisions_payload,
        decision_filter=args.filter,
        source_filter=args.source_filter,
        sort_key=args.sort,
        limit=args.limit,
    )
    queue_markdown_path = output_dir / f"review_queue_{args.filter}_{args.source_filter}.md"
    queue_csv_path = output_dir / f"review_queue_{args.filter}_{args.source_filter}.csv"
    write_review_queue_markdown(queue_markdown_path, queue)
    write_review_queue_csv(queue_csv_path, queue)

    print(
        f"Scanned {payload['sources_scanned']} text sources; "
        f"found {payload['candidate_count']} candidates in {payload['sources_with_candidates']} sources."
    )
    print(f"Summary: {summary_path}")
    print(
        f"Decisions template: {decisions_path}"
        f"{' (created)' if decisions_template_written else ' (preserved existing)'}"
    )
    print(f"Reviewed export: {reviewed_export_path}")
    print(f"Review queue: {queue_markdown_path}")
    print(f"Review queue CSV: {queue_csv_path}")
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
        candidate_dicts = []
        for candidate in candidates:
            candidate_dict = candidate.to_dict()
            candidate_dict.update(
                {
                    "candidate_id": candidate_id_for(
                        source_id=source.source_id,
                        source_path=source.source_path,
                        field_path=source.field_path,
                        candidate=candidate_dict,
                    ),
                    "decision": None,
                    "reviewed_value": None,
                    "reviewer_notes": [],
                }
            )
            candidate_dicts.append(candidate_dict)
        reviewed.append(
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_path": source.source_path,
                "field_path": source.field_path,
                "notes": source.notes,
                "text_preview": _preview(source.text),
                "candidate_count": len(candidates),
                "candidates": candidate_dicts,
            }
        )
    return reviewed


def candidate_id_for(
    *,
    source_id: str,
    source_path: str | None,
    field_path: str | None,
    candidate: dict[str, Any],
) -> str:
    identity = {
        "source_id": source_id,
        "source_path": source_path,
        "field_path": field_path,
        "span": candidate["span"],
        "observed": candidate["observed"],
        "candidate": candidate["candidate"],
        "match_type": candidate["match_type"],
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_decisions_template(candidate_sources: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    for source in candidate_sources:
        for candidate in source["candidates"]:
            decisions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_id": source["source_id"],
                    "source_path": source["source_path"],
                    "field_path": source["field_path"],
                    "observed": candidate["observed"],
                    "candidate": candidate["candidate"],
                    "span": candidate["span"],
                    "context": candidate["context"],
                    "match_type": candidate["match_type"],
                    "decision": None,
                    "reviewed_value": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "notes": [],
                }
            )
    return {
        "schema_version": "macron_candidate_decisions.v1",
        "instructions": [
            "Fill decision with one of: accept, reject, defer, edit.",
            "For edit, set reviewed_value to the reviewer-supplied correction.",
            "Do not edit source OCR artifacts; this file records review decisions only.",
            "Keep local decisions private unless source text/provenance is safe to share.",
        ],
        "decisions": decisions,
    }


def write_decisions_template(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def load_decisions_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "macron_candidate_decisions.v1", "decisions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def build_reviewed_export(
    candidate_sources: list[dict[str, Any]],
    decisions_payload: dict[str, Any],
) -> dict[str, Any]:
    candidates_by_id = {
        candidate["candidate_id"]: {**candidate, "source_id": source["source_id"], "source_path": source["source_path"]}
        for source in candidate_sources
        for candidate in source["candidates"]
    }
    decisions = decisions_payload.get("decisions", [])
    reviewed = []
    pending = 0
    stale = []
    counts = Counter({"accept": 0, "reject": 0, "defer": 0, "edit": 0, "pending": 0, "stale": 0})
    for decision in decisions:
        candidate_id = decision.get("candidate_id")
        if candidate_id not in candidates_by_id:
            stale.append(decision)
            counts["stale"] += 1
            continue
        decision_value = decision.get("decision")
        if decision_value in {None, ""}:
            pending += 1
            counts["pending"] += 1
            continue
        if decision_value not in {"accept", "reject", "defer", "edit"}:
            pending += 1
            counts["pending"] += 1
            continue
        counts[decision_value] += 1
        reviewed.append({**candidates_by_id[candidate_id], "review_decision": decision})

    return {
        "schema_version": "macron_candidate_review_export.v1",
        "source_text_mutated": False,
        "counts": dict(counts),
        "pending_decision_count": pending,
        "reviewed_decisions": reviewed,
        "stale_decisions": stale,
    }


def build_review_queue(
    candidate_sources: list[dict[str, Any]],
    decisions_payload: dict[str, Any],
    *,
    decision_filter: str = "all",
    source_filter: str = "all",
    sort_key: str = "source",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    decisions_by_id = {
        decision["candidate_id"]: decision
        for decision in decisions_payload.get("decisions", [])
        if decision.get("candidate_id")
    }
    rows = []
    for source in candidate_sources:
        for candidate in source["candidates"]:
            decision = decisions_by_id.get(candidate["candidate_id"], {})
            row = {
                "candidate_id": candidate["candidate_id"],
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "source_kind": source_kind_for(source),
                "source_path": source["source_path"],
                "field_path": source["field_path"],
                "observed": candidate["observed"],
                    "candidate": candidate["candidate"],
                    "case_pattern": candidate.get("case_pattern"),
                    "reviewed_value_suggestion": candidate.get("reviewed_value_suggestion"),
                    "match_type": candidate["match_type"],
                    "decision": _normalized_decision(decision.get("decision")),
                    "reviewed_value": decision.get("reviewed_value"),
                "context": candidate["context"],
                "notes": decision.get("notes", []),
            }
            if _queue_row_matches(row, decision_filter=decision_filter, source_filter=source_filter):
                rows.append(row)

    rows = sorted(rows, key=_sort_key(sort_key))
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def source_kind_for(source: dict[str, Any]) -> str:
    if source["source_type"] == "fixture":
        return "fixture"
    source_path = source.get("source_path") or ""
    if "macron_ocr_eval" in source_path or "macron_ocr_engine_comparison" in source_path:
        return "synthetic"
    return "real_ocr"


def write_review_queue_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Macron Candidate Review Queue",
        "",
        f"Candidate count: {len(rows)}",
        "",
        (
            "| Decision | Source Kind | Observed | Candidate | Reviewed Value Suggestion | "
            "Case Pattern | Match Type | Source ID | Context | Notes |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(row["decision"]),
                    _markdown_cell(row["source_kind"]),
                    _markdown_cell(row["observed"]),
                    _markdown_cell(row["candidate"]),
                    _markdown_cell(row["reviewed_value_suggestion"]),
                    _markdown_cell(row["case_pattern"]),
                    _markdown_cell(row["match_type"]),
                    _markdown_cell(row["source_id"]),
                    _markdown_cell(row["context"]),
                    _markdown_cell("; ".join(row.get("notes") or [])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_queue_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "candidate_id",
        "decision",
        "source_kind",
        "source_id",
        "source_path",
        "field_path",
        "observed",
        "candidate",
        "reviewed_value_suggestion",
        "case_pattern",
        "match_type",
        "context",
        "reviewed_value",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _queue_row_matches(row: dict[str, Any], *, decision_filter: str, source_filter: str) -> bool:
    if decision_filter != "all":
        decision = row["decision"]
        if decision_filter == "pending" and decision != "pending":
            return False
        if decision_filter == "accepted" and decision != "accept":
            return False
        if decision_filter == "rejected" and decision != "reject":
            return False
        if decision_filter == "deferred" and decision != "defer":
            return False
        if decision_filter == "edited" and decision != "edit":
            return False
        if decision_filter == "reviewed" and decision == "pending":
            return False
        if decision_filter == "stale":
            return False

    if source_filter == "all":
        return True
    if source_filter == "macron_eval":
        source_path = row.get("source_path") or ""
        return "macron_ocr_eval" in source_path or "macron_ocr_engine_comparison" in source_path
    return row["source_kind"] == source_filter


def _sort_key(sort_key: str):
    def key(row: dict[str, Any]) -> tuple[str, ...]:
        if sort_key == "candidate":
            return (row["candidate"].casefold(), row["observed"].casefold(), row["source_id"])
        if sort_key == "observed":
            return (row["observed"].casefold(), row["candidate"].casefold(), row["source_id"])
        if sort_key == "decision":
            return (row["decision"], row["candidate"].casefold(), row["source_id"])
        if sort_key == "match_type":
            return (row["match_type"], row["candidate"].casefold(), row["source_id"])
        return (row["source_kind"], row["source_id"], row["candidate"].casefold(), row["observed"].casefold())

    return key


def _normalized_decision(decision: Any) -> str:
    if decision in {None, ""}:
        return "pending"
    return str(decision)


def _markdown_cell(value: Any) -> str:
    text = _csv_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


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
