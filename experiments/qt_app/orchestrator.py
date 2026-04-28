# qt_app/orchestrator.py
from __future__ import annotations
import json, re, time, os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Set, List, Any

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LangKey = str
ProgressFn = Optional[Callable[[int, str], None]]

# -------- Recognition knobs --------
USE_TESSERACT_SNIFF: bool = True
TESSERACT_BIN: str = "/opt/homebrew/bin/tesseract"  # adjust if different
JP_RATIO_THRESHOLD: float = 0.18                     # ratio of JP vs (JP+Latin)
JP_MIN_CHARS: int = 30                               # absolute JP char floor
IMAGE_DOM_SNIFLEN: int = 200                         # very little text overall
EN_DOM_RATIO: float = 0.80                           # English clearly dominates
JP_MIN_FRACTION_EN: float = 0.60                     # JP must be >=60% of Latin if EN dominates
JP_CONF_MIN: int = 15                                # high-confidence JP chars floor
JP_MIN_RUNS: int = 2                                 # at least two contiguous JP runs
JP_MIN_MAX_RUN: int = 4                              # longest contiguous JP run length
# -----------------------------------

# Unicode buckets
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

def _jp_conf_chars(image, cfg_base: str) -> int:
    """Count JP characters (kana/kanji) with good confidence to ignore garbage."""
    import pytesseract, re
    from pytesseract import Output
    data = pytesseract.image_to_data(
        image, lang="jpn", config=f"{cfg_base} --psm 6", output_type=Output.DICT
    )
    jp_re = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
    total = 0
    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    for text, conf in zip(texts, confs):
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if c >= 70 and text:
            total += sum(1 for ch in text if jp_re.match(ch))
    return total

def _jp_run_stats(s: str) -> dict:
    """Return number of JP runs and the longest JP run length in a string."""
    jp_re = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]+")
    runs = jp_re.findall(s or "")
    return {"run_count": len(runs), "max_run": max((len(r) for r in runs), default=0)}

def _sniff_text_fast(img_path: str) -> dict:
    """
    Return oriented, split-language sniffs:
      { "latin_text": "...", "jp_text": "...",
        "sniff_len": int, "rotation_applied": int, "jp_conf_chars": int }
    Tries 0/90/180/270 and picks the rotation that maximizes signal.
    """
    if not USE_TESSERACT_SNIFF:
        return {"latin_text": "", "jp_text": "", "sniff_len": 0, "rotation_applied": 0, "jp_conf_chars": 0}

    try:
        from PIL import Image, ImageOps
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_BIN
        REPO = Path(__file__).resolve().parents[1]
        TESSDATA = REPO / "tessdata"
        os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA))
        base_cfg = f'--oem 1 -c preserve_interword_spaces=1 --tessdata-dir "{TESSDATA}"'

        # Load & light preproc
        im0 = Image.open(img_path)
        im0 = ImageOps.exif_transpose(im0)
        im0 = ImageOps.autocontrast(im0.convert("L"))

        # Try all 4 rotations, score each by character buckets
        candidates = []
        for rot in (0, 90, 180, 270):
            im = im0 if rot == 0 else im0.rotate(360 - rot, expand=True)  # CCW rotate

            try:
                latin_text = pytesseract.image_to_string(im, lang="eng", config=f"{base_cfg} --psm 6").strip()
            except Exception:
                latin_text = ""

            jp_text = ""
            try:
                jp_text = pytesseract.image_to_string(im, lang="jpn", config=f"{base_cfg} --psm 6").strip()
                if not jp_text:
                    jp_text = pytesseract.image_to_string(im, lang="jpn_vert", config=f"{base_cfg} --psm 5").strip()
            except Exception:
                jp_text = ""

            bl = _bucket_scripts(latin_text)
            bj = _bucket_scripts(jp_text)
            latin = bl["latin"]
            jp    = bj["hira"] + bj["kata"] + bj["kanji"]
            candidates.append({
                "rot": rot,
                "latin_text": latin_text,
                "jp_text": jp_text,
                "latin": latin, "jp": jp,
                "image": im,  # keep for conf calc if chosen
            })

        # Choose best rotation
        if any(c["latin"] >= 200 for c in candidates):
            best = max(candidates, key=lambda c: c["latin"])
        else:
            best = max(candidates, key=lambda c: c["jp"])

        latin_text = best["latin_text"]
        jp_text    = best["jp_text"]
        sniff_len  = len(latin_text) + len(jp_text)
        rotation   = best["rot"]

        # High-confidence JP char count (filters noise)
        try:
            jp_conf = _jp_conf_chars(best["image"], base_cfg)
        except Exception:
            jp_conf = 0

        return {
            "latin_text": latin_text,
            "jp_text": jp_text,
            "sniff_len": sniff_len,
            "rotation_applied": rotation,
            "jp_conf_chars": int(jp_conf),
        }
    except Exception:
        return {"latin_text": "", "jp_text": "", "sniff_len": 0, "rotation_applied": 0, "jp_conf_chars": 0}


@dataclass
class OrchestratorResult:
    doc_dir: Path
    data: Dict[str, Any]

class Orchestrator:
    def __init__(self, processed_root: Path | str = "data/processed") -> None:
        self.processed_root = Path(processed_root)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self._last_debug: Dict[str, Any] = {}

    # -------------- Recognition --------------
    def analyze_types(self, image_path: str) -> Set[LangKey]:
        img = Path(image_path)
        if not img.exists():
            return {"english"}

        sniff = _sniff_text_fast(str(img))
        latin_text = sniff.get("latin_text", "")
        jp_text    = sniff.get("jp_text", "")
        sample     = (latin_text + "\n" + jp_text).strip()

        if not sample:
            self._last_debug = {"sniff_empty": True}
            return {"english"}

        b_lat = _bucket_scripts(latin_text)
        b_jp  = _bucket_scripts(jp_text)

        latin = b_lat["latin"]
        jp    = b_jp["hira"] + b_jp["kata"] + b_jp["kanji"]
        total = max(latin + jp, 1)
        latin_ratio = latin / total
        jp_ratio    = jp / total

        # Run-based JP strength
        runs = _jp_run_stats(jp_text)
        jp_run_count = runs["run_count"]
        max_jp_run   = runs["max_run"]

        langs: Set[LangKey] = set()

        # Strong English signal
        if latin_ratio >= 0.70:
            langs.add("english")

        # Image-dominant pages: never tick JP
        allow_jp = sniff.get("sniff_len", 0) >= IMAGE_DOM_SNIFLEN

        # Also require a minimum of high-confidence JP chars
        if allow_jp and sniff.get("jp_conf_chars", 0) < JP_CONF_MIN:
            allow_jp = False

        if allow_jp:
            en_dominant = (latin_ratio >= EN_DOM_RATIO)
            base_jp_ok = (
                jp_ratio >= JP_RATIO_THRESHOLD and
                jp >= JP_MIN_CHARS and
                jp_run_count >= JP_MIN_RUNS and
                max_jp_run >= JP_MIN_MAX_RUN
            )

            jp_ok = False
            if base_jp_ok and latin_ratio < 0.90:
                if en_dominant:
                    # when English dominates, JP must be close in magnitude
                    if sniff.get("jp_conf_chars", 0) >= JP_MIN_FRACTION_EN * max(latin, 1):
                        jp_ok = True
                else:
                    jp_ok = True

            if jp_ok:
                langs.add("jp_modern")

            # Classical heuristic (kanji-heavy, kana-light)
            if b_jp["kanji"] >= 50 and (b_jp["hira"] + b_jp["kata"]) <= 5:
                langs.add("jp_classical")

        # If nothing triggered, choose stronger signal
        if not langs:
            langs.add("english" if latin_ratio >= jp_ratio else "jp_modern")

        # Final post-filter: drop spurious JP if below absolute floor
        if "jp_modern" in langs and jp < JP_MIN_CHARS:
            langs.remove("jp_modern")

        # Debug snapshot for UI
        self._last_debug = {
            "sniff_len": sniff.get("sniff_len", 0),
            "rotation": sniff.get("rotation_applied", 0),
            "jp_conf_chars": sniff.get("jp_conf_chars", 0),
            "jp_run_count": jp_run_count,
            "max_jp_run": max_jp_run,
            "latin": latin,
            "hira": b_jp["hira"],
            "kata": b_jp["kata"],
            "kanji": b_jp["kanji"],
            "latin_ratio": round(latin_ratio, 3),
            "jp_ratio": round(jp_ratio, 3),
            "JP_RATIO_THRESHOLD": JP_RATIO_THRESHOLD,
            "JP_MIN_CHARS": JP_MIN_CHARS,
            "JP_CONF_MIN": JP_CONF_MIN,
            "JP_MIN_RUNS": JP_MIN_RUNS,
            "JP_MIN_MAX_RUN": JP_MIN_MAX_RUN,
            "IMAGE_DOM_SNIFLEN": IMAGE_DOM_SNIFLEN,
            "EN_DOM_RATIO": EN_DOM_RATIO,
            "JP_MIN_FRACTION_EN": JP_MIN_FRACTION_EN,
            "USE_TESSERACT_SNIFF": USE_TESSERACT_SNIFF,
        }
        return langs

    def last_analyze_debug(self) -> dict:
        return self._last_debug

    # -------------- Processing --------------
    def process(self, image_path: str, selected_langs: Set[LangKey], progress: ProgressFn = None) -> OrchestratorResult:
        def emit(pct: int, msg: str):
            if progress:
                try: progress(int(max(0, min(100, pct))), msg)
                except Exception: pass

        img = Path(image_path)
        if not img.exists(): raise FileNotFoundError(image_path)
        if img.suffix.lower() not in IMG_EXTS: raise ValueError(f"Unsupported extension: {img.suffix}")

        emit(5, "Initializing OCR pipeline…")
        from processors.ocr_processor import OCRProcessor
        proc = OCRProcessor()
        pr = proc.process_document(str(img))

        emit(75, "Packing results…")
        result_dict = self._pack_from_processing_result(pr, str(img), selected_langs)

        emit(90, "Saving JSON…")
        doc_dir = self._save_json(result_dict)

        emit(100, "Done.")
        return OrchestratorResult(doc_dir=doc_dir, data=result_dict)

    # -------------- Helpers --------------
    def _pack_from_processing_result(self, pr: Any, image_path: str, selected_langs: Set[LangKey]) -> Dict[str, Any]:
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
        if not (("jp_modern" in selected_langs) or ("jp_classical" in selected_langs)):
            data["japanese_result"] = None
            data["has_japanese"] = False
        return data

    def _save_json(self, result_dict: Dict[str, Any]) -> Path:
        doc_dir = self.processed_root / f"doc_{int(time.time())}"
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "data.json").write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc_dir
