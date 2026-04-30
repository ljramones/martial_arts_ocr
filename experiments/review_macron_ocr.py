from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

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

from martial_arts_ocr.ocr.postprocessor import OCRPostProcessor
from martial_arts_ocr.pipeline.document_models import DocumentResult, PageResult
from utils.text.text_utils import TextCleaner


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/notebook_outputs/macron_ocr_eval"
DEFAULT_LANGUAGES = ["eng", "eng+jpn", "jpn"]
DEFAULT_PSMS = ["3", "6", "11"]
MACRON_TERMS = [
    "koryū",
    "budō",
    "Daitō-ryū",
    "jūjutsu",
    "dōjō",
    "ryūha",
    "sōke",
    "iaidō",
    "kenjutsu",
    "aikijūjutsu",
    "ō",
    "ū",
]


@dataclass(frozen=True)
class SyntheticFixture:
    fixture_id: str
    font_name: str
    font_size: int
    lines: tuple[str, ...]


SYNTHETIC_FIXTURES = [
    SyntheticFixture(
        fixture_id="arial_large_terms",
        font_name="Arial.ttf",
        font_size=38,
        lines=(
            "koryū budō Daitō-ryū",
            "jūjutsu dōjō ryūha sōke",
            "iaidō kenjutsu aikijūjutsu",
        ),
    ),
    SyntheticFixture(
        fixture_id="times_large_terms",
        font_name="Times New Roman.ttf",
        font_size=40,
        lines=(
            "koryū budō Daitō-ryū",
            "jūjutsu dōjō ryūha sōke",
            "iaidō kenjutsu aikijūjutsu",
        ),
    ),
    SyntheticFixture(
        fixture_id="arial_small_sentence",
        font_name="Arial.ttf",
        font_size=24,
        lines=(
            "Draeger studied koryū budō and Daitō-ryū.",
            "The dōjō taught aikijūjutsu and iaidō.",
        ),
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic macron romanization samples and run OCR preservation checks."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored output directory for synthetic images and OCR summaries.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help="Tesseract language config. Can be repeated. Defaults to eng, eng+jpn, jpn.",
    )
    parser.add_argument(
        "--psm",
        action="append",
        default=None,
        help="Tesseract PSM value. Can be repeated. Defaults to 3, 6, 11.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve(args.output_dir)
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    languages = args.language or DEFAULT_LANGUAGES
    psms = args.psm or DEFAULT_PSMS
    postprocessor = OCRPostProcessor(domain="martial_arts")
    cleaner = TextCleaner()

    results = []
    for fixture in SYNTHETIC_FIXTURES:
        image_path = synthetic_dir / f"{fixture.fixture_id}.png"
        image = _render_fixture(fixture)
        image.save(image_path)
        expected_text = "\n".join(fixture.lines)
        expected_control = _cleanup_serialization_control(
            fixture.fixture_id,
            image_path,
            expected_text,
            postprocessor=postprocessor,
            cleaner=cleaner,
        )

        fixture_results = []
        for language in languages:
            for psm in psms:
                raw_text, elapsed = _run_tesseract(image, language=language, psm=psm)
                cleaned_text = postprocessor.clean_text(raw_text)
                readable_text, _stats = cleaner.clean_text(cleaned_text)
                serialized = _serialize_readable_text(fixture.fixture_id, image_path, readable_text)
                fixture_results.append(
                    {
                        "language": language,
                        "psm": str(psm),
                        "raw_output": raw_text,
                        "cleaned_output": cleaned_text,
                        "readable_text": readable_text,
                        "serialized_text": serialized["text_summary"]["readable_text"],
                        "text_txt": readable_text,
                        "raw_terms": _term_hits(raw_text, MACRON_TERMS),
                        "cleaned_terms": _term_hits(cleaned_text, MACRON_TERMS),
                        "readable_terms": _term_hits(readable_text, MACRON_TERMS),
                        "serialized_terms": _term_hits(serialized["text_summary"]["readable_text"], MACRON_TERMS),
                        "macron_chars_raw": _macron_chars(raw_text),
                        "macron_chars_readable": _macron_chars(readable_text),
                        "result": _preservation_result(readable_text, fixture.lines),
                        "processing_time": elapsed,
                    }
                )

        best = _best_result(fixture_results)
        results.append(
            {
                "fixture_id": fixture.fixture_id,
                "image_path": str(image_path),
                "font_name": fixture.font_name,
                "font_size": fixture.font_size,
                "expected": expected_text,
                "expected_terms": [term for term in MACRON_TERMS if term in expected_text],
                "expected_text_cleanup_control": expected_control,
                "best_result": _compact_result(best),
                "results": [_compact_result(result) for result in fixture_results],
            }
        )
        print(
            f"{fixture.fixture_id}: best={best['language']} psm={best['psm']} "
            f"terms={best['readable_terms']} result={best['result']}"
        )

    payload = {
        "languages": languages,
        "psms": psms,
        "terms": MACRON_TERMS,
        "synthetic_fixtures": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


def _render_fixture(fixture: SyntheticFixture) -> Image.Image:
    font = _load_font(fixture.font_name, fixture.font_size)
    padding = 36
    line_spacing = int(fixture.font_size * 1.45)
    width = max(_text_width(line, font) for line in fixture.lines) + padding * 2
    height = padding * 2 + line_spacing * len(fixture.lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(fixture.lines):
        draw.text((padding, padding + index * line_spacing), line, fill="black", font=font)
    return image


def _run_tesseract(image: Image.Image, *, language: str, psm: str) -> tuple[str, float]:
    if pytesseract is None:
        return "", 0.0
    started = time.time()
    config = f"--psm {psm} --oem 1 --dpi 300 -c preserve_interword_spaces=1"
    try:
        text = pytesseract.image_to_string(image, lang=language, config=config)
    except Exception as exc:
        text = f"[tesseract_error] {exc}"
    return text, time.time() - started


def _serialize_readable_text(fixture_id: str, image_path: Path, readable_text: str) -> dict[str, Any]:
    page = PageResult(
        page_number=1,
        raw_text=readable_text,
        metadata={"readable_text": readable_text, "ocr_word_count": 0, "ocr_line_count": 0},
    )
    document = DocumentResult(document_id=None, source_path=image_path, pages=[page])
    return document.to_dict()


def _cleanup_serialization_control(
    fixture_id: str,
    image_path: Path,
    expected_text: str,
    *,
    postprocessor: OCRPostProcessor,
    cleaner: TextCleaner,
) -> dict[str, Any]:
    cleaned_text = postprocessor.clean_text(expected_text)
    readable_text, _stats = cleaner.clean_text(cleaned_text)
    serialized = _serialize_readable_text(fixture_id, image_path, readable_text)
    return {
        "cleaned_output": cleaned_text,
        "readable_text": readable_text,
        "serialized_text": serialized["text_summary"]["readable_text"],
        "cleaned_terms": _term_hits(cleaned_text, MACRON_TERMS),
        "readable_terms": _term_hits(readable_text, MACRON_TERMS),
        "serialized_terms": _term_hits(serialized["text_summary"]["readable_text"], MACRON_TERMS),
        "macron_chars_readable": _macron_chars(readable_text),
    }


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda result: (
            len(result["readable_terms"]),
            len(result["macron_chars_readable"]),
            -_edit_distance_score(result["readable_text"]),
        ),
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": result["language"],
        "psm": result["psm"],
        "raw_output": result["raw_output"],
        "cleaned_output": result["cleaned_output"],
        "readable_text": result["readable_text"],
        "serialized_text": result["serialized_text"],
        "text_txt": result["text_txt"],
        "raw_terms": result["raw_terms"],
        "cleaned_terms": result["cleaned_terms"],
        "readable_terms": result["readable_terms"],
        "serialized_terms": result["serialized_terms"],
        "macron_chars_raw": result["macron_chars_raw"],
        "macron_chars_readable": result["macron_chars_readable"],
        "result": result["result"],
        "processing_time": result["processing_time"],
    }


def _preservation_result(readable_text: str, expected_lines: tuple[str, ...]) -> str:
    expected = "\n".join(expected_lines)
    expected_terms = [term for term in MACRON_TERMS if term in expected]
    hits = _term_hits(readable_text, expected_terms)
    if len(hits) == len(expected_terms):
        return "preserved"
    if hits:
        return "partial"
    stripped_terms = [
        term.replace("ō", "o").replace("ū", "u")
        for term in expected_terms
        if len(term) > 1
    ]
    if any(term and term in readable_text for term in stripped_terms):
        return "macrons_stripped"
    return "missing_or_misread"


def _term_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def _macron_chars(text: str) -> list[str]:
    return [char for char in text if char in "āēīōūĀĒĪŌŪ"]


def _edit_distance_score(_text: str) -> int:
    # Placeholder tie-breaker: prefer shorter noisy output after term/macron matches.
    return len(_text)


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental") / font_name,
        Path("/Library/Fonts") / font_name,
        Path(font_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype(font_name, size=size)


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
