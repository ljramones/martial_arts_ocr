from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - local OCR availability varies
    pytesseract = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.review_macron_ocr import (  # noqa: E402
    MACRON_TERMS,
    SYNTHETIC_FIXTURES,
    _render_fixture,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/notebook_outputs/macron_ocr_engine_comparison"
DEFAULT_TESSERACT_CONFIGS = [
    ("eng", "3"),
    ("eng+jpn", "3"),
    ("jpn", "3"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare available OCR engines on synthetic macron romanization fixtures."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored output directory for synthetic images and OCR comparison JSON.",
    )
    parser.add_argument(
        "--skip-easyocr",
        action="store_true",
        help="Skip EasyOCR even if it is importable with local models.",
    )
    parser.add_argument(
        "--skip-paddle",
        action="store_true",
        help="Skip the .venv-eval PaddleOCR availability check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    fixtures = []
    all_results = []
    for fixture in SYNTHETIC_FIXTURES:
        image_path = synthetic_dir / f"{fixture.fixture_id}.png"
        image = _render_fixture(fixture)
        image.save(image_path)
        expected_text = "\n".join(fixture.lines)
        expected_terms = [term for term in MACRON_TERMS if term in expected_text]
        fixtures.append(
            {
                "fixture": fixture.fixture_id,
                "path": str(image_path),
                "terms": expected_terms,
                "source": "synthetic_generated",
            }
        )

        all_results.extend(_run_tesseract_configs(image, fixture.fixture_id, expected_terms))

        if not args.skip_easyocr:
            all_results.append(_run_easyocr(image, fixture.fixture_id, expected_terms))
        else:
            all_results.append(_unavailable("easyocr", "skipped", fixture.fixture_id, expected_terms))

        if not args.skip_paddle:
            all_results.append(_run_paddle_availability_check(image_path, fixture.fixture_id, expected_terms))
        else:
            all_results.append(_unavailable("paddleocr", "skipped", fixture.fixture_id, expected_terms))

    payload = {
        "fixtures": fixtures,
        "terms": MACRON_TERMS,
        "results": all_results,
        "summary": _summarize_results(all_results),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in payload["summary"]:
        print(
            f"{row['engine']} {row['config']}: macron_preserved={row['macron_preserved']} "
            f"preserved={row['preserved']} "
            f"stripped={row['stripped']} misread={row['misread']} "
            f"missing={row['missing']} unavailable={row['unavailable']}"
        )
    print(f"Summary: {summary_path}")
    return 0


def _run_tesseract_configs(
    image: Image.Image,
    fixture_id: str,
    expected_terms: list[str],
) -> list[dict[str, Any]]:
    results = []
    for language, psm in DEFAULT_TESSERACT_CONFIGS:
        started = time.time()
        config = f"--psm {psm} --oem 1 --dpi 300 -c preserve_interword_spaces=1"
        if pytesseract is None:
            output = ""
            notes = "pytesseract unavailable"
            classification = "engine_unavailable"
        else:
            try:
                output = pytesseract.image_to_string(image, lang=language, config=config)
                notes = ""
                classification = None
            except Exception as exc:
                output = ""
                notes = str(exc)
                classification = "engine_unavailable"
        results.append(
            _result_payload(
                engine="tesseract",
                config=f"{language} psm {psm}",
                fixture=fixture_id,
                expected_terms=expected_terms,
                output=output,
                elapsed=time.time() - started,
                forced_classification=classification,
                notes=notes,
            )
        )
    return results


def _run_easyocr(
    image: Image.Image,
    fixture_id: str,
    expected_terms: list[str],
) -> dict[str, Any]:
    started = time.time()
    try:
        import easyocr  # type: ignore
        import numpy as np

        reader = easyocr.Reader(["ja"], gpu=False, verbose=False, download_enabled=False)
        rows = reader.readtext(np.array(image), detail=1, paragraph=False)
        output = "\n".join(str(row[1]) for row in rows)
        return _result_payload(
            engine="easyocr",
            config="ja download_enabled=false",
            fixture=fixture_id,
            expected_terms=expected_terms,
            output=output,
            elapsed=time.time() - started,
            notes=f"detections={len(rows)}",
        )
    except Exception as exc:
        return _result_payload(
            engine="easyocr",
            config="ja download_enabled=false",
            fixture=fixture_id,
            expected_terms=expected_terms,
            output="",
            elapsed=time.time() - started,
            forced_classification="engine_unavailable",
            notes=str(exc),
        )


def _run_paddle_availability_check(
    image_path: Path,
    fixture_id: str,
    expected_terms: list[str],
) -> dict[str, Any]:
    started = time.time()
    python_path = REPO_ROOT / ".venv-eval/bin/python"
    if not python_path.exists():
        return _result_payload(
            engine="paddleocr",
            config=".venv-eval",
            fixture=fixture_id,
            expected_terms=expected_terms,
            output="",
            elapsed=0.0,
            forced_classification="engine_unavailable",
            notes=".venv-eval missing",
        )

    script = (
        "import paddleocr; "
        "print(getattr(paddleocr, '__version__', 'unknown')); "
        "print('available')"
    )
    completed = subprocess.run(
        [str(python_path), "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return _result_payload(
            engine="paddleocr",
            config=".venv-eval availability",
            fixture=fixture_id,
            expected_terms=expected_terms,
            output="",
            elapsed=time.time() - started,
            forced_classification="engine_unavailable",
            notes=(completed.stderr or completed.stdout).strip(),
        )

    return _result_payload(
        engine="paddleocr",
        config=".venv-eval availability",
        fixture=fixture_id,
        expected_terms=expected_terms,
        output="",
        elapsed=time.time() - started,
        forced_classification="engine_unavailable",
        notes="PaddleOCR import available, but OCR execution is not attempted in this helper to avoid model setup/download side effects.",
    )


def _result_payload(
    *,
    engine: str,
    config: str,
    fixture: str,
    expected_terms: list[str],
    output: str,
    elapsed: float,
    forced_classification: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    term_results = {
        term: forced_classification or _classify_term(term, output)
        for term in expected_terms
    }
    counts = _count_classifications(term_results)
    return {
        "engine": engine,
        "config": config,
        "fixture": fixture,
        "expected_terms": expected_terms,
        "output": output,
        "term_results": term_results,
        "classification": _overall_classification(counts),
        "counts": counts,
        "processing_time": elapsed,
        "notes": notes,
    }


def _classify_term(term: str, output: str) -> str:
    if not output:
        return "missing"
    if term in output:
        return "preserved"
    stripped = _strip_macrons(term)
    if stripped != term and stripped in output:
        return "stripped"
    if any(piece and piece in output for piece in _term_pieces(term)):
        return "misread"
    return "missing"


def _strip_macrons(text: str) -> str:
    return (
        text.replace("ā", "a")
        .replace("ē", "e")
        .replace("ī", "i")
        .replace("ō", "o")
        .replace("ū", "u")
        .replace("Ā", "A")
        .replace("Ē", "E")
        .replace("Ī", "I")
        .replace("Ō", "O")
        .replace("Ū", "U")
    )


def _term_pieces(term: str) -> list[str]:
    clean = _strip_macrons(term)
    pieces = [part for part in clean.replace("-", " ").split() if len(part) >= 4]
    return pieces


def _count_classifications(term_results: dict[str, str]) -> dict[str, int]:
    labels = ["preserved", "stripped", "misread", "missing", "engine_unavailable"]
    return {label: sum(1 for value in term_results.values() if value == label) for label in labels}


def _overall_classification(counts: dict[str, int]) -> str:
    if counts["engine_unavailable"]:
        return "engine_unavailable"
    if counts["preserved"]:
        return "preserved"
    if counts["stripped"]:
        return "stripped"
    if counts["misread"]:
        return "misread"
    return "missing"


def _summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        key = (result["engine"], result["config"])
        row = grouped.setdefault(
            key,
            {
                "engine": result["engine"],
                "config": result["config"],
                "preserved": 0,
                "stripped": 0,
                "misread": 0,
                "missing": 0,
                "unavailable": 0,
                "macron_preserved": 0,
                "macron_stripped": 0,
                "macron_misread": 0,
                "macron_missing": 0,
                "notes": set(),
            },
        )
        row["preserved"] += result["counts"]["preserved"]
        row["stripped"] += result["counts"]["stripped"]
        row["misread"] += result["counts"]["misread"]
        row["missing"] += result["counts"]["missing"]
        row["unavailable"] += result["counts"]["engine_unavailable"]
        for term, classification in result["term_results"].items():
            if not _has_macron(term):
                continue
            if classification == "preserved":
                row["macron_preserved"] += 1
            elif classification == "stripped":
                row["macron_stripped"] += 1
            elif classification == "misread":
                row["macron_misread"] += 1
            elif classification == "missing":
                row["macron_missing"] += 1
        if result.get("notes"):
            row["notes"].add(str(result["notes"]).splitlines()[0][:180])

    summary = []
    for row in grouped.values():
        summary.append({**row, "notes": "; ".join(sorted(row["notes"]))})
    return sorted(summary, key=lambda item: (item["engine"], item["config"]))


def _has_macron(text: str) -> bool:
    return any(char in text for char in "āēīōūĀĒĪŌŪ")


def _unavailable(engine: str, notes: str, fixture_id: str, expected_terms: list[str]) -> dict[str, Any]:
    return _result_payload(
        engine=engine,
        config="skipped",
        fixture=fixture_id,
        expected_terms=expected_terms,
        output="",
        elapsed=0.0,
        forced_classification="engine_unavailable",
        notes=notes,
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
