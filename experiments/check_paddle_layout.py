from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/notebook_outputs/paddle_layout_eval"
DFD_MANIFEST = REPO_ROOT / "data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json"
CORPUS2_MANIFEST = REPO_ROOT / "data/corpora/ad_hoc/corpus2/manifests/manifest.local.json"

DEFAULT_SAMPLE_IDS = [
    # Corpus 2 broad/mixed cases from mixed-region refinement review.
    "corpus2_new_doc_2026_04_28_16_55_48",
    "corpus2_new_doc_2026_04_28_17_10_58",
    "corpus2_new_doc_2026_04_28_17_19_36",
    "corpus2_new_doc_2026_04_28_18_29_28",
    "corpus2_new_doc_2026_04_28_18_54_00",
    # DFD hard pages.
    "original_img_3335",
    "original_img_3344",
    "original_img_3397",
    "original_img_3352",
    "original_img_3330",
    # Known-good regression checks.
    "original_img_3292",
    "original_img_3340",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    corpus: str
    description: str = ""


def main() -> int:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)

    samples = select_samples(args)
    backend_info = detect_backend()
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PaddleOCR: {backend_info.get('paddleocr_version', 'unavailable')}")
    print(f"PaddlePaddle: {backend_info.get('paddle_version', 'unavailable')}")
    print(f"API variant: {backend_info.get('api_variant', 'unavailable')}")

    comparison: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            **backend_info,
        },
        "samples": [],
    }

    if not backend_info["available"]:
        comparison["status"] = "skipped"
        comparison["skip_reason"] = backend_info["skip_reason"]
        write_json(output_dir / "comparison.json", comparison)
        print(f"Skipped: {backend_info['skip_reason']}")
        return 0

    engine = build_engine(backend_info)
    for sample in samples:
        result = evaluate_sample(sample, engine, backend_info, output_dir)
        comparison["samples"].append(result)
        print(
            f"{sample.sample_id}: {len(result.get('regions', []))} regions, "
            f"{result.get('elapsed_seconds', 0):.2f}s"
        )

    comparison["status"] = "completed"
    write_json(output_dir / "comparison.json", comparison)
    print(f"Wrote {output_dir / 'comparison.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Throwaway PaddleOCR layout evaluation.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSON path. Can be provided more than once.")
    parser.add_argument("--sample-id", action="append", default=[], help="Sample ID to evaluate. Can be provided more than once.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected samples after filtering.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)), help="Output directory for JSON and overlays.")
    return parser.parse_args()


def select_samples(args: argparse.Namespace) -> list[Sample]:
    manifests = [Path(path) for path in args.manifest] if args.manifest else [DFD_MANIFEST, CORPUS2_MANIFEST]
    all_samples: dict[str, Sample] = {}
    for manifest in manifests:
        all_samples.update(load_manifest(manifest))

    wanted = args.sample_id or DEFAULT_SAMPLE_IDS
    selected: list[Sample] = []
    missing: list[str] = []
    for sample_id in wanted:
        sample = all_samples.get(sample_id)
        if sample is None:
            missing.append(sample_id)
            continue
        selected.append(sample)

    if missing:
        print(f"Missing sample IDs: {', '.join(missing)}", file=sys.stderr)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No samples selected.")
    return selected


def load_manifest(path: Path) -> dict[str, Sample]:
    manifest_path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path
    if not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}", file=sys.stderr)
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Sample] = {}
    for item in data.get("samples", []):
        sample_id = item.get("id")
        image_path = item.get("path")
        if not sample_id or not image_path:
            continue
        source = item.get("source") or {}
        corpus = str(source.get("collection") or infer_corpus(sample_id))
        result[sample_id] = Sample(
            sample_id=sample_id,
            path=(REPO_ROOT / image_path).resolve(),
            corpus=corpus,
            description=str(item.get("description") or ""),
        )
    return result


def infer_corpus(sample_id: str) -> str:
    if sample_id.startswith("corpus2_"):
        return "corpus2"
    if sample_id.startswith("original_"):
        return "dfd_notes_master"
    return "unknown"


def detect_backend() -> dict[str, Any]:
    try:
        import paddleocr  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "available": False,
            "skip_reason": f"paddleocr import failed: {exc}",
            "paddleocr_version": None,
            "paddle_version": None,
            "api_variant": None,
        }

    try:
        import paddle  # type: ignore[import-not-found]

        paddle_version = getattr(paddle, "__version__", "unknown")
    except Exception as exc:
        paddle_version = f"unavailable: {exc}"

    api_variant = None
    api_class = None
    if hasattr(paddleocr, "PPStructureV3"):
        api_variant = "PPStructureV3"
        api_class = paddleocr.PPStructureV3
    elif hasattr(paddleocr, "PPStructure"):
        api_variant = "PPStructure"
        api_class = paddleocr.PPStructure

    if api_class is None:
        return {
            "available": False,
            "skip_reason": "paddleocr is installed, but PPStructureV3/PPStructure is not exposed",
            "paddleocr_version": getattr(paddleocr, "__version__", "unknown"),
            "paddle_version": paddle_version,
            "api_variant": None,
        }

    return {
        "available": True,
        "skip_reason": None,
        "paddleocr_version": getattr(paddleocr, "__version__", "unknown"),
        "paddle_version": paddle_version,
        "api_variant": api_variant,
        "api_class": api_class,
    }


def build_engine(backend_info: dict[str, Any]) -> Any:
    api_class = backend_info["api_class"]
    variant = backend_info["api_variant"]
    attempts: list[tuple[str, dict[str, Any]]] = []
    if variant == "PPStructureV3":
        attempts = [
            ("default", {}),
        ]
    else:
        attempts = [
            ("old_ppstructure_no_ocr", {"show_log": False, "layout": True, "ocr": False}),
            ("old_ppstructure_default", {}),
        ]
    last_error: Exception | None = None
    for name, kwargs in attempts:
        try:
            engine = api_class(**kwargs)
            backend_info["constructor"] = name
            return engine
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not construct {variant}: {last_error}")


def evaluate_sample(sample: Sample, engine: Any, backend_info: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    raw: Any = None
    error: str | None = None
    try:
        raw = run_engine(engine, sample.path)
        regions = normalize_regions(raw)
    except Exception as exc:
        regions = []
        error = repr(exc)

    overlay_path = None
    if sample.path.exists() and regions:
        try:
            overlay_path = draw_overlay(sample, regions, output_dir / "overlays")
        except Exception as exc:
            error = f"{error}; overlay failed: {exc}" if error else f"overlay failed: {exc}"

    elapsed = time.perf_counter() - started
    return {
        "sample_id": sample.sample_id,
        "corpus": sample.corpus,
        "path": str(sample.path.relative_to(REPO_ROOT) if sample.path.is_relative_to(REPO_ROOT) else sample.path),
        "backend": "paddle_pp_structure",
        "api_variant": backend_info.get("api_variant"),
        "constructor": backend_info.get("constructor"),
        "regions": regions,
        "raw": make_jsonable(raw),
        "overlay_path": str(overlay_path.relative_to(REPO_ROOT)) if overlay_path else None,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def run_engine(engine: Any, image_path: Path) -> Any:
    if hasattr(engine, "predict"):
        return engine.predict(str(image_path))
    try:
        return engine(str(image_path))
    except TypeError:
        pass

    # Older PPStructure examples often pass ndarray images.
    try:
        from PIL import Image
        import numpy as np

        with Image.open(image_path) as image:
            return engine(np.array(image.convert("RGB")))
    except Exception:
        # Re-raise the path-call error shape when ndarray support is unavailable.
        return engine(str(image_path))


def normalize_regions(raw: Any) -> list[dict[str, Any]]:
    items = flatten_result(raw)
    regions: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = find_bbox(item)
        if bbox is None:
            continue
        label = normalize_label(str(item.get("type") or item.get("label") or item.get("category") or item.get("class") or "unknown"))
        regions.append(
            {
                "label": label,
                "bbox": bbox,
                "score": find_score(item),
                "raw_label": str(item.get("type") or item.get("label") or item.get("category") or item.get("class") or "unknown"),
                "raw": make_jsonable(item),
            }
        )
    return regions


def flatten_result(raw: Any) -> list[Any]:
    raw = unwrap_result(raw)
    if isinstance(raw, list):
        items: list[Any] = []
        for value in raw:
            unwrapped = unwrap_result(value)
            if isinstance(unwrapped, dict):
                nested = unwrapped.get("layout") or unwrapped.get("res") or unwrapped.get("boxes") or unwrapped.get("regions")
                if isinstance(nested, list):
                    items.extend(flatten_result(nested))
                items.append(unwrapped)
            else:
                items.extend(flatten_result(unwrapped))
        return items
    if isinstance(raw, dict):
        for key in ("layout", "res", "boxes", "regions", "results"):
            if isinstance(raw.get(key), list):
                return flatten_result(raw[key])
        return [raw]
    return []


def unwrap_result(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, type(None))):
        return value
    for attr in ("json", "to_dict", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                continue
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


def find_bbox(item: dict[str, Any]) -> list[int] | None:
    for key in ("bbox", "box", "coordinate", "coordinates"):
        bbox = item.get(key)
        normalized = normalize_bbox(bbox)
        if normalized is not None:
            return normalized
    layout = item.get("layout")
    if isinstance(layout, dict):
        return find_bbox(layout)
    return None


def normalize_bbox(bbox: Any) -> list[int] | None:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if {"x", "y", "width", "height"}.issubset(bbox):
            return [int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])]
        if {"left", "top", "width", "height"}.issubset(bbox):
            return [int(bbox["left"]), int(bbox["top"]), int(bbox["width"]), int(bbox["height"])]
        if {"x1", "y1", "x2", "y2"}.issubset(bbox):
            return xyxy_to_xywh([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        if all(isinstance(point, (list, tuple)) for point in bbox[:4]):
            xs = [float(point[0]) for point in bbox[:4]]
            ys = [float(point[1]) for point in bbox[:4]]
            return [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
        return xyxy_to_xywh(bbox[:4])
    return None


def xyxy_to_xywh(values: Any) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in values[:4]]
    if x2 > x1 and y2 > y1:
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    return [int(x1), int(y1), int(x2), int(y2)]


def find_score(item: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "prob"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def normalize_label(label: str) -> str:
    normalized = label.lower()
    if any(token in normalized for token in ("figure", "image", "pic", "photo")):
        return "figure"
    if "table" in normalized:
        return "table"
    if "title" in normalized:
        return "title"
    if "caption" in normalized:
        return "caption"
    if "text" in normalized or "paragraph" in normalized:
        return "text"
    return "unknown"


def draw_overlay(sample: Sample, regions: list[dict[str, Any]], overlay_dir: Path) -> Path:
    from PIL import Image, ImageDraw

    overlay_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    palette = {
        "figure": "red",
        "table": "blue",
        "text": "green",
        "title": "orange",
        "caption": "purple",
        "unknown": "yellow",
    }
    for region in regions:
        x, y, width, height = region["bbox"]
        color = palette.get(region["label"], "yellow")
        draw.rectangle([x, y, x + width, y + height], outline=color, width=4)
        draw.text((x, max(0, y - 14)), f"{region['label']} {region.get('score') or ''}", fill=color)
    output_path = overlay_dir / f"{sample.sample_id}.png"
    canvas.save(output_path)
    return output_path


def make_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return make_jsonable(vars(value))
    return repr(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(make_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
