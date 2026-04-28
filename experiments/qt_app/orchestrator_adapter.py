from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Set, List, Any

# ------------------------------------------------------------
# Public constants / types
# ------------------------------------------------------------

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LangKey = str  # one of: "english", "jp_modern", "jp_classical"
ProgressFn = Optional[Callable[[int, str], None]]  # progress(0..100, "message")


# ------------------------------------------------------------
# Config flags (safe defaults for stability on macOS)
# ------------------------------------------------------------

USE_TESSERACT_SNIFF: bool = False  # set True to enable quick OCR-based detection
TESSERACT_SNiff_LANGS: str = "eng+jpn"  # fallback to "eng" if jpn pack missing
JP_RATIO_THRESHOLD: float = 0.12        # raise to 0.18-0.20 to be stricter


# ------------------------------------------------------------
# Utilities (kept light and pure-Python)
# ------------------------------------------------------------

_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_HIRA  = re.compile(r"[\u3040-\u309F]")
_RE_KATA  = re.compile(r"[\u30A0-\u30FF]")
_RE_KANJI = re.compile(r"[\u4E00-\u9FFF]")


def _bucket_scripts(s: str) -> Dict[str, int]:
    return {
        "latin": len(_RE_LATIN.findall(s)),
        "hira":  len(_RE_HIRA.findall(s)),
        "kata":  len(_RE_KATA.findall(s)),
        "kanji": len(_RE_KANJI.findall(s)),
    }


def _sniff_text_fast(img_path: str) -> str:
    """
    Very quick OCR sniff to assist language detection.
    Runs only if USE_TESSERACT_SNIFF is True. Otherwise returns "".
    """
    if not USE_TESSERACT_SNIFF:
        return ""
    try:
        from PIL import Image  # pillow is lightweight
        import pytesseract
        im = Image.open(img_path)
        cfg = "--psm 6 -c preserve_interword_spaces=1"
        try:
            txt = pytesseract.image_to_string(im, lang=TESSERACT_SNiff_LANGS, config=cfg)
        except Exception:
            txt = pytesseract.image_to_string(im, lang="eng", config=cfg)
        return (txt or "")[:4000]
    except Exception:
        return ""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_doc_dir(root: Path) -> Path:
    return root / f"doc_{int(time.time())}"


# ------------------------------------------------------------
# Public Orchestrator
# ------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """
    A compact, UI-friendly result. The full processor-native objects remain internal.
    """
    doc_dir: Path
    data: Dict[str, Any]  # what we saved to data.json


class Orchestrator:
    """
    High-level façade around your processors with:
      - Optional multi-label language detection
      - Override-aware processing
      - Single JSON artifact for the UI to consume
      - Lazy imports to keep the UI stable
    """

    def __init__(self, processed_root: Path | str = "processed") -> None:
        self.processed_root = Path(processed_root)
        _ensure_dir(self.processed_root)

    # ------------------------
    # Detection (multi-label)
    # ------------------------
    def analyze_types(self, image_path: str) -> Set[LangKey]:
        """
        Content-based multi-label detection. Returns a subset of:
        {'english', 'jp_modern', 'jp_classical'}.

        Safe default: returns {'english'} unless USE_TESSERACT_SNIFF is enabled
        and detects visible kana/kanji.
        """
        img = Path(image_path)
        if not img.exists():
            return {"english"}

        sample = _sniff_text_fast(str(img)).strip()
        if not sample:
            # No sniff: default to English; user can tick JP boxes manually
            return {"english"}

        b = _bucket_scripts(sample)
        latin = b["latin"]
        jp    = b["hira"] + b["kata"] + b["kanji"]
        total = max(latin + jp, 1)
        latin_ratio = latin / total
        jp_ratio    = jp / total

        langs: Set[LangKey] = set()
        if latin_ratio >= 0.70:
            langs.add("english")
        if jp_ratio >= JP_RATIO_THRESHOLD:
            langs.add("jp_modern")
        # classical heuristic: many kanji, almost no kana
        if b["kanji"] >= 50 and (b["hira"] + b["kata"]) <= 5:
            langs.add("jp_classical")
        if not langs:
            langs.add("english" if latin_ratio >= jp_ratio else "jp_modern")
        return langs

    # ------------------------
    # Processing
    # ------------------------
    def process(
        self,
        image_path: str,
        selected_langs: Set[LangKey],
        progress: ProgressFn = None,
    ) -> OrchestratorResult:
        """
        Run the pipeline for the given image and selected language labels.
        Writes processed/doc_<ts>/data.json and returns an OrchestratorResult.
        """
        def emit(pct: int, msg: str) -> None:
            if progress:
                try:
                    progress(int(max(0, min(100, pct))), msg)
                except Exception:
                    pass

        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(image_path)
        if img.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Unsupported image extension: {img.suffix}")

        emit(2, "Initializing pipeline…")

        # Prefer your OCRProcessor.process_document if available
        pr = None
        try:
            from processors.ocr_processor import OCRProcessor  # lazy import
            proc = OCRProcessor()
            emit(8, "Running OCR pipeline…")
            pr = proc.process_document(str(img))
            emit(75, "Post-processing/analysis…")
            result_dict = self._pack_result_from_processing_result(
                pr,
                image_path=str(img),
                selected_langs=selected_langs,
            )
            emit(90, "Saving results…")
            doc_dir = self._save_json(result_dict)
            emit(100, "Done.")
            return OrchestratorResult(doc_dir=doc_dir, data=result_dict)
        except Exception as e:
            # If OCRProcessor isn't available or failed early, try the manual path
            last_err = e

        # Manual fallback if the above fails
        try:
            emit(5, "Extracting layout…")
            from processors.content_extractor import ContentExtractor
            ce = ContentExtractor()
            layout = ce.extract(str(img))  # expected to return regions/lines

            emit(25, "Running OCR (combined)…")
            from processors.ocr_processor import OCRProcessor  # still try to reuse OCR core
            ocr = OCRProcessor()

            # Map selection to OCR codes; supports multi-lang
            lang_codes: List[str] = []
            if "english" in selected_langs:
                lang_codes.append("eng")
            if "jp_modern" in selected_langs:
                lang_codes.append("jpn")
            if "jp_classical" in selected_langs:
                lang_codes.append("jpn_classical")  # adjust to your model label

            try:
                # If your OCR supports multi-language API:
                #   run_ocr(layout, langs=[...])
                ocr_result = ocr.run_ocr(layout, langs=lang_codes)  # type: ignore[arg-type]
            except TypeError:
                # Sequential fallback per language; naive merge by append
                ocr_result = {"regions": []}
                for code in lang_codes or ["eng"]:
                    part = ocr.run_ocr(layout, lang=code)  # type: ignore[call-arg]
                    _merge_regions_inplace(ocr_result, part, code)

            emit(55, "Japanese analysis…")
            has_jp = any(k in selected_langs for k in ("jp_modern", "jp_classical"))
            jp_result = dict(ocr_result)
            if has_jp:
                from processors.japanese_processor import JapaneseProcessor
                jp = JapaneseProcessor()
                jp_result = jp.analyze(
                    ocr_result,
                    classical=("jp_classical" in selected_langs),
                    modern=("jp_modern" in selected_langs),
                )

            emit(72, "Reconstructing outputs…")
            from processors.page_reconstructor import PageReconstructor
            recon = PageReconstructor()
            out = recon.build_components(
                jp_result if has_jp else ocr_result,
                original_image=str(img),
                metadata={"selected_langs": sorted(list(selected_langs))}
            )

            doc_dir = None
            if isinstance(out, dict) and "doc_dir" in out:
                doc_dir = Path(out["doc_dir"])
            else:
                doc_dir = _now_doc_dir(self.processed_root)

            result_dict = {
                "image_path": str(img),
                "selected_langs": sorted(list(selected_langs)),
                "regions": (jp_result if has_jp else ocr_result).get("regions", []),
                "japanese_result": jp_result if has_jp else None,
            }

            # Respect explicit English-only override
            if not has_jp:
                result_dict["japanese_result"] = None

            emit(90, "Saving results…")
            if doc_dir and not (doc_dir / "data.json").exists():
                _ensure_dir(doc_dir)
                (doc_dir / "data.json").write_text(
                    json.dumps(result_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            else:
                # Fallback path if recon didn't return a doc_dir
                doc_dir = self._save_json(result_dict)

            emit(100, "Done.")
            return OrchestratorResult(doc_dir=doc_dir, data=result_dict)
        except Exception as e2:
            # surface the earlier error if manual also fails
            raise RuntimeError(f"Pipeline failed. Primary: {last_err}\nFallback: {e2}") from e2

    # ------------------------
    # Internal helpers
    # ------------------------
    def _pack_result_from_processing_result(
        self,
        pr: Any,
        image_path: str,
        selected_langs: Set[LangKey],
    ) -> Dict[str, Any]:
        """
        Convert your OCRProcessor's ProcessingResult into a compact dict for the UI.
        """
        # The following fields are based on your earlier structure; adjust if names differ.
        data: Dict[str, Any] = {
            "image_path": image_path,
            "text": getattr(pr, "cleaned_text", None),
            "overall_confidence": getattr(pr, "overall_confidence", None),
            "quality_score": getattr(pr, "quality_score", None),
            "processing_time": getattr(pr, "processing_time", None),
            "engines_used": [getattr(r, "engine", None) for r in getattr(pr, "ocr_results", [])] if getattr(pr, "ocr_results", None) else [],
            "has_japanese": bool(getattr(pr, "processing_metadata", {}).get("has_japanese")) if getattr(pr, "processing_metadata", None) else False,
            "language_ratio": getattr(pr, "text_statistics", {}).get("language_ratio", {}) if getattr(pr, "text_statistics", None) else {},
            "extracted_images": getattr(pr, "extracted_images", None),
            "boxes": getattr(getattr(pr, "best_ocr_result", None), "bounding_boxes", []) or [],
            "japanese_result": getattr(getattr(pr, "japanese_result", None), "to_dict", lambda: None)(),
            "selected_langs": sorted(list(selected_langs)),
        }

        # Respect UI override: if the user didn't check any JP box, drop JP
        if not (("jp_modern" in selected_langs) or ("jp_classical" in selected_langs)):
            data["japanese_result"] = None
            data["has_japanese"] = False

        return data

    def _save_json(self, result_dict: Dict[str, Any]) -> Path:
        doc_dir = _now_doc_dir(self.processed_root)
        _ensure_dir(doc_dir)
        (doc_dir / "data.json").write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return doc_dir


# ------------------------------------------------------------
# Small merge helper (manual OCR fallback)
# ------------------------------------------------------------

def _merge_regions_inplace(base: Dict[str, Any], add: Dict[str, Any], source_lang_code: str) -> None:
    regs = list(base.get("regions", []))
    for r in add.get("regions", []):
        if isinstance(r, dict) and "lang" not in r:
            r = dict(r)
            r["lang"] = source_lang_code
        regs.append(r)
    base["regions"] = regs
