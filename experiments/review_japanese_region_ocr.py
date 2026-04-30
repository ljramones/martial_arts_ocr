from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover - availability depends on local env
    pytesseract = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/notebook_outputs/japanese_region_ocr_eval"
DEFAULT_LANGUAGES = ["jpn", "eng+jpn", "eng"]
DEFAULT_PSMS = ["6"]
DEFAULT_PREPROCESS_PROFILES = [
    "none",
    "grayscale",
    "threshold",
    "upscale_2x",
    "upscale_3x",
    "contrast_sharpen",
]
DEFAULT_VARIANTS = DEFAULT_PREPROCESS_PROFILES


@dataclass(frozen=True)
class OcrRoute:
    language: str
    psm: str
    preprocess_profiles: tuple[str, ...]
    role: str = "primary"


@dataclass(frozen=True)
class RegionProfile:
    name: str
    description: str
    routes: tuple[OcrRoute, ...]
    needs_review_default: bool = False
    notes: str = ""


REGION_PROFILES: dict[str, RegionProfile] = {
    "horizontal_modern_japanese": RegionProfile(
        name="horizontal_modern_japanese",
        description="Horizontal modern Japanese body or line text.",
        routes=(
            OcrRoute("jpn", "6", ("none", "upscale_2x", "threshold")),
            OcrRoute("jpn", "7", ("none", "upscale_2x"), role="single_line_diagnostic"),
        ),
        notes="Use jpn first; PSM 7 remains diagnostic until more single-line samples exist.",
    ),
    "vertical_modern_japanese": RegionProfile(
        name="vertical_modern_japanese",
        description="Vertical modern Japanese columns, sidebars, or headers.",
        routes=(
            OcrRoute("jpn_vert", "5", ("none", "upscale_2x", "threshold")),
            OcrRoute("jpn", "6", ("none", "upscale_2x"), role="horizontal_comparison"),
        ),
        notes="jpn_vert + PSM 5 was decisive on the vertical sidebar sample.",
    ),
    "mixed_english_japanese_page_text": RegionProfile(
        name="mixed_english_japanese_page_text",
        description="English-heavy regions where the goal is blended readable text.",
        routes=(
            OcrRoute("eng+jpn", "6", ("none", "upscale_2x")),
            OcrRoute("eng+jpn", "11", ("none", "upscale_2x"), role="sparse_text_diagnostic"),
            OcrRoute("jpn", "6", ("none", "upscale_2x"), role="japanese_term_diagnostic"),
        ),
        notes="Optimize for blended readability; keep jpn as a diagnostic for embedded terms.",
    ),
    "mixed_japanese_parentheticals": RegionProfile(
        name="mixed_japanese_parentheticals",
        description="Japanese terms or parentheticals embedded inside mostly English context.",
        routes=(
            OcrRoute("jpn", "6", ("none", "upscale_2x"), role="primary_term_recovery"),
            OcrRoute("eng+jpn", "6", ("none", "upscale_2x"), role="blended_readability_comparison"),
            OcrRoute("eng+jpn", "11", ("none", "upscale_2x"), role="sparse_text_diagnostic"),
        ),
        notes="Optimize for Japanese term recovery in parenthetical/list regions.",
    ),
    "romanized_japanese_macrons": RegionProfile(
        name="romanized_japanese_macrons",
        description="Latin romanized Japanese text with possible macrons.",
        routes=(
            OcrRoute("eng", "6", ("none", "upscale_2x")),
            OcrRoute("eng+jpn", "6", ("none", "upscale_2x"), role="mixed_language_diagnostic"),
        ),
        notes="Real macron OCR remains unproven; avoid threshold by default.",
    ),
    "stylized_calligraphy": RegionProfile(
        name="stylized_calligraphy",
        description="Brush-like, decorative, or calligraphic Japanese.",
        routes=(
            OcrRoute("jpn_vert", "5", ("none", "contrast_sharpen"), role="diagnostic"),
            OcrRoute("jpn", "6", ("none", "contrast_sharpen"), role="diagnostic"),
        ),
        needs_review_default=True,
        notes="Tesseract remains unreliable; treat as review/manual transcription candidate.",
    ),
    "unknown_japanese_like": RegionProfile(
        name="unknown_japanese_like",
        description="Japanese-like region with unknown orientation or text type.",
        routes=(
            OcrRoute("jpn", "6", ("none", "upscale_2x"), role="diagnostic"),
            OcrRoute("jpn_vert", "5", ("none", "upscale_2x"), role="diagnostic"),
            OcrRoute("eng+jpn", "6", ("none", "upscale_2x"), role="diagnostic"),
        ),
        needs_review_default=True,
        notes="Exploratory route; use compact diagnostics only.",
    ),
}
REGION_PROFILES["mixed_english_japanese"] = REGION_PROFILES["mixed_english_japanese_page_text"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OCR on selected Japanese regions and preprocessing variants."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Region manifest JSON with source_image and bbox entries.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored output directory for crops and JSON results.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help="Tesseract language config. Can be repeated. Defaults to jpn, eng+jpn, eng.",
    )
    parser.add_argument(
        "--psm",
        action="append",
        default=None,
        help="Tesseract PSM value. Can be repeated. Defaults to 6.",
    )
    parser.add_argument(
        "--include-easyocr",
        action="store_true",
        help="Attempt EasyOCR if available locally. May require existing local model files.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(REGION_PROFILES),
        default=None,
        help="Override all manifest region types with a named region routing profile.",
    )
    parser.add_argument(
        "--use-routing-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use manifest region_type/japanese_region_type routing profiles. Defaults to enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    languages = args.language or DEFAULT_LANGUAGES
    psms = args.psm or DEFAULT_PSMS
    easyocr_reader = _init_easyocr_reader(args.include_easyocr)

    results: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        result = _process_sample(
            sample,
            output_dir=output_dir,
            crops_dir=crops_dir,
            languages=languages,
            psms=psms,
            profile_override=args.profile,
            use_routing_profiles=args.use_routing_profiles,
            easyocr_reader=easyocr_reader,
        )
        results.append(result)
        best = result.get("best_tesseract") or {}
        print(
            f"{sample['id']}: variants={len(result.get('variants', []))} "
            f"route_profile={result.get('routing_profile', {}).get('name')} "
            f"best={best.get('language')} psm={best.get('psm')} "
            f"preprocess={best.get('preprocess_profile')} hits={best.get('expected_terms_recovered')}"
        )

    payload = {
        "manifest": str(manifest_path),
        "languages": languages,
        "psms": psms,
        "preprocessing_profiles": DEFAULT_PREPROCESS_PROFILES,
        "available_region_profiles": list(REGION_PROFILES),
        "easyocr_tested": easyocr_reader is not None,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


def _process_sample(
    sample: dict[str, Any],
    *,
    output_dir: Path,
    crops_dir: Path,
    languages: list[str],
    psms: list[str],
    profile_override: str | None,
    use_routing_profiles: bool,
    easyocr_reader: Any,
) -> dict[str, Any]:
    source_path = _resolve(sample["source_image"])
    image = Image.open(source_path).convert("RGB")
    bbox = _coerce_bbox(sample["bbox"], image.size)
    crop = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    expected_terms = [str(term) for term in sample.get("expected_terms", [])]

    sample_dir = crops_dir / sample["id"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    routing_profile = _select_region_profile(
        sample,
        profile_override=profile_override,
        use_routing_profiles=use_routing_profiles,
    )
    route_plan = _route_plan_for_sample(
        routing_profile,
        languages=languages,
        psms=psms,
        use_routing_profiles=use_routing_profiles,
    )
    variants = _build_preprocess_variants(crop)
    variant_results = []
    for route in route_plan:
        for profile_name in route.preprocess_profiles:
            variant_image = variants[profile_name]
            image_path = sample_dir / f"{profile_name}.png"
            if not image_path.exists():
                variant_image.save(image_path)
            text, elapsed = _run_tesseract(variant_image, language=route.language, psm=route.psm)
            variant_results.append(
                {
                    "engine": "tesseract",
                    "language": route.language,
                    "psm": str(route.psm),
                    "preprocess_profile": profile_name,
                    "variant": profile_name,
                    "route_role": route.role,
                    "text": text,
                    "text_length": len(text),
                    "japanese_char_count": _count_japanese(text),
                    "expected_terms_recovered": _term_hits(text, expected_terms),
                    "quality_judgment": _quality_judgment(text, expected_terms),
                    "processing_time": elapsed,
                }
            )

    full_page_results = []
    for language in languages:
        for psm in psms:
            text, elapsed = _run_tesseract(image, language=language, psm=psm)
            full_page_results.append(
                {
                    "engine": "tesseract",
                    "language": language,
                    "psm": str(psm),
                    "text_preview": _preview(text),
                    "text_length": len(text),
                    "japanese_char_count": _count_japanese(text),
                    "expected_terms_recovered": _term_hits(text, expected_terms),
                    "processing_time": elapsed,
                }
            )

    easyocr_result = None
    if easyocr_reader is not None:
        easyocr_result = _run_easyocr(easyocr_reader, crop, expected_terms)

    best = _best_tesseract_result(variant_results)
    return {
        "sample_id": sample["id"],
        "source_image": str(source_path),
        "bbox": bbox,
        "routing_profile": _profile_to_dict(routing_profile),
        "selected_routes": [_route_to_dict(route) for route in route_plan],
        "description": sample.get("description"),
        "visible_text": sample.get("visible_text"),
        "expected_terms": expected_terms,
        "notes": sample.get("notes"),
        "needs_human_review": bool(
            sample.get("needs_human_review", routing_profile.needs_review_default)
        ),
        "full_page_results": full_page_results,
        "variants": [
            {k: v for k, v in result.items() if k != "text"} | {"text_preview": _preview(result["text"])}
            for result in variant_results
        ],
        "best_tesseract": (
            {k: v for k, v in best.items() if k != "text"} | {"text_preview": _preview(best["text"])}
            if best
            else None
        ),
        "easyocr": easyocr_result,
    }


def _select_region_profile(
    sample: dict[str, Any],
    *,
    profile_override: str | None,
    use_routing_profiles: bool,
) -> RegionProfile:
    if not use_routing_profiles:
        return REGION_PROFILES["unknown_japanese_like"]
    profile_name = (
        profile_override
        or sample.get("japanese_region_type")
        or sample.get("region_type")
        or sample.get("profile")
        or "unknown_japanese_like"
    )
    return REGION_PROFILES.get(str(profile_name), REGION_PROFILES["unknown_japanese_like"])


def _route_plan_for_sample(
    profile: RegionProfile,
    *,
    languages: list[str],
    psms: list[str],
    use_routing_profiles: bool,
) -> tuple[OcrRoute, ...]:
    if use_routing_profiles:
        return profile.routes
    return tuple(
        OcrRoute(language, str(psm), tuple(DEFAULT_PREPROCESS_PROFILES))
        for language in languages
        for psm in psms
    )


def _profile_to_dict(profile: RegionProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "description": profile.description,
        "needs_review_default": profile.needs_review_default,
        "notes": profile.notes,
    }


def _route_to_dict(route: OcrRoute) -> dict[str, Any]:
    return {
        "language": route.language,
        "psm": route.psm,
        "preprocess_profiles": list(route.preprocess_profiles),
        "role": route.role,
    }


def _build_preprocess_variants(crop: Image.Image) -> dict[str, Image.Image]:
    gray = ImageOps.grayscale(crop)
    threshold = gray.point(lambda px: 255 if px > 165 else 0, mode="1").convert("RGB")
    sharp = ImageEnhance.Contrast(crop).enhance(1.8).filter(ImageFilter.SHARPEN)
    return {
        "none": crop,
        "grayscale": gray.convert("RGB"),
        "threshold": threshold,
        "upscale_2x": crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS),
        "upscale_3x": crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS),
        "contrast_sharpen": sharp,
    }


def _build_variants(crop: Image.Image) -> dict[str, Image.Image]:
    """Backward-compatible alias for older review snippets/tests."""
    variants = _build_preprocess_variants(crop)
    return variants | {
        "original_crop": variants["none"],
        "contrast_or_sharpen": variants["contrast_sharpen"],
    }


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


def _init_easyocr_reader(include_easyocr: bool) -> Any:
    if not include_easyocr:
        return None
    try:
        import easyocr  # type: ignore

        return easyocr.Reader(["ja", "en"], gpu=False, verbose=False)
    except Exception as exc:
        print(f"EasyOCR unavailable for region review: {exc}")
        return None


def _run_easyocr(reader: Any, image: Image.Image, expected_terms: list[str]) -> dict[str, Any]:
    started = time.time()
    try:
        import numpy as np

        rows = reader.readtext(np.array(image), detail=1, paragraph=False)
        text = "\n".join(str(row[1]) for row in rows)
        return {
            "engine": "easyocr",
            "text_preview": _preview(text),
            "text_length": len(text),
            "japanese_char_count": _count_japanese(text),
            "expected_terms_recovered": _term_hits(text, expected_terms),
            "processing_time": time.time() - started,
            "detections": len(rows),
        }
    except Exception as exc:
        return {
            "engine": "easyocr",
            "error": str(exc),
            "processing_time": time.time() - started,
        }


def _best_tesseract_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda result: (
            len(result.get("expected_terms_recovered", [])),
            result.get("japanese_char_count", 0),
            result.get("text_length", 0),
        ),
    )


def _coerce_bbox(raw_bbox: list[Any], image_size: tuple[int, int]) -> list[int]:
    x, y, width, height = [int(round(float(value))) for value in raw_bbox]
    image_width, image_height = image_size
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))
    return [x, y, width, height]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _count_japanese(text: str) -> int:
    return sum(1 for char in text if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff")


def _term_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def _quality_judgment(text: str, expected_terms: list[str]) -> str:
    if text.startswith("[tesseract_error]"):
        return "fail"
    hits = _term_hits(text, expected_terms)
    if expected_terms and len(hits) == len(expected_terms):
        return "meaningful"
    if hits:
        return "partial"
    if _count_japanese(text) >= 5:
        return "noisy"
    return "fail"


def _preview(text: str, *, limit: int = 6) -> list[str]:
    return [line for line in text.splitlines()[:limit]]


if __name__ == "__main__":
    raise SystemExit(main())
