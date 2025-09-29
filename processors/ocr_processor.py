"""
OCR Processor for Martial Arts OCR
Coordinates OCR engines, image processing, and Japanese text analysis.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata

# OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available - OCR capabilities limited")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available - OCR capabilities limited")

import numpy as np
import cv2

from config import get_config
from utils.image_utils import (
    ImageProcessor, LayoutAnalyzer, ImageRegion,
    save_image, extract_image_region, merge_regions_into_lines, validate_image_file
)
from utils.text_utils import TextCleaner, LanguageDetector, TextStatistics
from processors.japanese_processor import JapaneseProcessor, JapaneseProcessingResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Results from OCR processing."""
    text: str
    confidence: float
    processing_time: float
    engine: str
    bounding_boxes: List[Dict] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'engine': self.engine,
            'bounding_boxes': self.bounding_boxes or [],
            'metadata': self.metadata or {}
        }


@dataclass
class ProcessingResult:
    """Complete processing result for a document."""
    document_id: Optional[int]
    page_id: Optional[int]

    # OCR results
    ocr_results: List[OCRResult]
    best_ocr_result: OCRResult
    raw_text: str
    cleaned_text: str

    # Layout analysis
    text_regions: List[ImageRegion]
    image_regions: List[ImageRegion]
    extracted_images: List[Dict[str, Any]]

    # Language processing
    japanese_result: Optional[JapaneseProcessingResult]
    language_segments: List[Dict]
    text_statistics: Dict[str, Any]

    # Quality metrics
    overall_confidence: float
    quality_score: float
    processing_time: float

    # Output formats
    html_content: str
    markdown_content: str

    # Metadata
    processing_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'page_id': self.page_id,
            'ocr_results': [result.to_dict() for result in self.ocr_results],
            'best_ocr_result': self.best_ocr_result.to_dict(),
            'raw_text': self.raw_text,
            'cleaned_text': self.cleaned_text,
            'text_regions': [region.to_dict() for region in self.text_regions],
            'image_regions': [region.to_dict() for region in self.image_regions],
            'extracted_images': self.extracted_images,
            'japanese_result': self.japanese_result.to_dict() if self.japanese_result else None,
            'language_segments': self.language_segments,
            'text_statistics': self.text_statistics,
            'overall_confidence': self.overall_confidence,
            'quality_score': self.quality_score,
            'processing_time': self.processing_time,
            'html_content': self.html_content,
            'markdown_content': self.markdown_content,
            'processing_metadata': self.processing_metadata
        }


# --------------------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------------------

class TesseractEngine:
    """Tesseract OCR engine wrapper."""

    def __init__(self):
        self.config = get_config().OCR_ENGINES.get('tesseract', {})
        self.available = TESSERACT_AVAILABLE and self.config.get('enabled', True)
        if self.available:
            self._verify_installation()

    def _verify_installation(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            available_langs = pytesseract.get_languages()
            required_langs = self.config.get('languages', ['eng'])
            missing = set(required_langs) - set(available_langs)
            if missing:
                logger.warning(f"Missing Tesseract languages: {missing}")
        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
            self.available = False

    def _compose_config(self, base_cfg: str, psm: Optional[str]) -> str:
        """
        Merge defaults with optional --psm.
        Preserve '-c preserve_interword_spaces=1', use LSTM (--oem 1) and DPI=300.
        """
        base = base_cfg or "--psm 6"
        parts = base.split()

        # strip any existing --psm value
        if "--psm" in parts:
            try:
                i = parts.index("--psm")
                parts.pop(i)
                if i < len(parts):
                    parts.pop(i)
            except Exception:
                pass

        if psm:
            parts += ["--psm", str(psm)]

        cfg_str = " ".join(parts)
        if "preserve_interword_spaces=1" not in cfg_str:
            parts += ["-c", "preserve_interword_spaces=1"]
        cfg_str = " ".join(parts)
        if "--oem" not in cfg_str:
            parts += ["--oem", "1"]
        if "--dpi" not in cfg_str:
            parts += ["--dpi", "300"]
        return " ".join(parts)

    def _run_once(self, image: np.ndarray, cfg: str, langs: str) -> Tuple[str, float, List[Dict]]:
        """Run tesseract once, return (text, avg_conf, boxes)."""
        ocr_data = pytesseract.image_to_data(
            image, lang=langs, config=cfg, output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(image, lang=langs, config=cfg)

        confidences = []
        for c in ocr_data.get('conf', []):
            try:
                ci = int(c)
                if ci > 0:
                    confidences.append(ci)
            except Exception:
                continue
        avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        boxes = []
        n = len(ocr_data.get('text', []))
        for i in range(n):
            try:
                ci = int(ocr_data['conf'][i])
            except Exception:
                ci = -1
            if ci > 0 and (ocr_data['text'][i] or "").strip():
                boxes.append({
                    'text': ocr_data['text'][i],
                    'confidence': ci / 100.0,
                    'x': int(ocr_data['left'][i]),
                    'y': int(ocr_data['top'][i]),
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                })
        return text, float(avg_conf), boxes

    def process_image(self, image: np.ndarray, psm: Optional[str] = None, region_hint: str = "auto") -> OCRResult:
        """
        Process image with Tesseract OCR.
        For speed: try at most TWO PSMs per region chosen by hint; global/full uses just PSM 6.
        """
        if not self.available:
            raise RuntimeError("Tesseract is not available")

        start_time = time.time()
        langs = '+'.join(self.config.get('languages', ['eng']))

        if psm:
            psms = [psm]
        else:
            if region_hint == "word":
                psms = ["8", "6"]        # word then default
            elif region_hint == "line":
                psms = ["7", "6"]        # line then default
            else:
                psms = ["6"]             # full/auto: only one pass for speed

        base_cfg = self.config.get('config', '--psm 6')
        best: Optional[OCRResult] = None

        for p in psms:
            try:
                cfg = self._compose_config(base_cfg, p)
                text, avg_conf, boxes = self._run_once(image, cfg, langs)
                cand = OCRResult(
                    text=text,
                    confidence=avg_conf,
                    processing_time=0.0,
                    engine='tesseract',
                    bounding_boxes=boxes,
                    metadata={'languages': langs, 'config': cfg, 'psm_used': p}
                )
                if (best is None) or (cand.confidence > best.confidence) or \
                   (cand.confidence == best.confidence and len(cand.text) > len(best.text)):
                    best = cand
                # short-circuit if already strong
                if cand.confidence >= 0.90 and len((cand.text or "").strip()) >= 200:
                    best = cand
                    break
            except Exception as e:
                logger.debug(f"Tesseract PSM {p} failed: {e}")
                continue

        if best is None:
            return OCRResult(text="", confidence=0.0, processing_time=time.time() - start_time,
                             engine='tesseract', bounding_boxes=[], metadata={'languages': langs})

        best.processing_time = time.time() - start_time
        return best


class EasyOCREngine:
    """EasyOCR engine wrapper."""

    def __init__(self):
        self.config = get_config().OCR_ENGINES.get('easyocr', {})
        self.available = EASYOCR_AVAILABLE and self.config.get('enabled', True)
        self.reader = None
        if self.available:
            self._initialize_reader()

    def _initialize_reader(self):
        """Initialize EasyOCR reader."""
        try:
            languages = self.config.get('languages', ['en'])
            gpu = self.config.get('gpu', False)
            self.reader = easyocr.Reader(languages, gpu=gpu)  # paragraph control in call
            logger.info(f"EasyOCR initialized with languages: {languages}")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            self.available = False

    def process_image(self, image: np.ndarray, paragraph: bool = True) -> OCRResult:
        """Process image with EasyOCR."""
        if not self.available or not self.reader:
            raise RuntimeError("EasyOCR is not available")

        start_time = time.time()
        try:
            results = self.reader.readtext(image, detail=1, paragraph=paragraph)

            text_lines: List[str] = []
            confidences: List[float] = []
            bounding_boxes: List[Dict[str, Any]] = []

            for (bbox, text, conf) in results:
                confidence_threshold = self.config.get('confidence_threshold', 0.5)
                if conf >= confidence_threshold and (text or "").strip():
                    text_lines.append(text)
                    confidences.append(float(conf))

                    x_coords = [pt[0] for pt in bbox]
                    y_coords = [pt[1] for pt in bbox]
                    x = int(min(x_coords)); y = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))

                    bounding_boxes.append({
                        'text': text,
                        'confidence': float(conf),
                        'x': x, 'y': y, 'width': width, 'height': height
                    })

            full_text = '\n'.join(text_lines)
            avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                processing_time=processing_time,
                engine='easyocr',
                bounding_boxes=bounding_boxes,
                metadata={
                    'total_detections': len(results),
                    'accepted_detections': len(text_lines),
                    'confidence_threshold': self.config.get('confidence_threshold', 0.5),
                    'paragraph': paragraph
                }
            )

        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
            raise


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

class OCRProcessor:
    """Main OCR processing coordinator."""

    def __init__(self):
        self.config = get_config()
        self.image_processor = ImageProcessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()
        self.japanese_processor = JapaneseProcessor()

        # Initialize OCR engines
        self.tesseract_engine = TesseractEngine()
        self.easyocr_engine = EasyOCREngine()

        self._verify_engines()

    def _verify_engines(self):
        """Verify at least one OCR engine is available."""
        available_engines = []
        if self.tesseract_engine.available:
            available_engines.append('tesseract')
        if self.easyocr_engine.available:
            available_engines.append('easyocr')

        if not available_engines:
            raise RuntimeError("No OCR engines are available")

        logger.info(f"Available OCR engines: {available_engines}")

    # -------------------- text-level JP detection helpers --------------------

    @staticmethod
    def _strip_cjk_punct(s: str) -> str:
        """Remove characters from CJK Symbols & Punctuation block (U+3000–U+303F)."""
        return re.sub(r"[\u3000-\u303F]", "", s or "")

    @staticmethod
    def _has_kana_or_kanji(s: str) -> bool:
        """
        True only if text contains Hiragana/Katakana/Kanji (not punctuation).
        Hiragana U+3040–U+309F, Katakana U+30A0–U+30FF, CJK Unified U+3400–U+9FFF,
        Halfwidth Katakana U+FF66–U+FF9D.
        """
        return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u9FFF\uFF66-\uFF9D]", s or ""))

    # -------------------- MAIN --------------------

    def process_document(self, image_path: str, document_id: Optional[int] = None) -> ProcessingResult:
        """
        Process a complete document with OCR and analysis.
        """
        start_time = time.time()

        try:
            logger.info(f"Starting OCR processing for: {image_path}")

            # Load (EXIF-aware) and preprocess image
            original_image = self.image_processor.load_image(image_path)
            processed_image = self.image_processor.preprocess_for_ocr(original_image)

            # ---------- PRE-FLIGHT: full page, fast ----------
            preflight = self._run_ocr_engines(processed_image, "preflight_full", psm_override=None, region_hint="auto")
            pre_best = self._select_best_ocr_result(preflight)
            pre_text_fixed = self._post_ocr_fixups(pre_best.text)
            pre_clean, _ = self.text_cleaner.clean_text(pre_text_fixed)

            # If already strong & long, skip regions entirely
            if pre_best.confidence >= 0.88 and len(pre_clean) >= 1400:
                logger.debug("Preflight strong enough, skipping region OCR")
                final_text = pre_clean
                text_regions: List[ImageRegion] = []
                image_regions: List[ImageRegion] = []
                extracted_images: List[Dict[str, Any]] = []
                ocr_results = [pre_best]
                best_result = pre_best
            else:
                # Analyze layout
                text_regions_raw = self.layout_analyzer.detect_text_regions(processed_image)
                image_regions = self.layout_analyzer.detect_image_regions(processed_image)

                # Merge overlapping regions then group into lines
                merged_regions = self.layout_analyzer.merge_overlapping_regions(text_regions_raw + image_regions)
                text_regions = [r for r in merged_regions if r.region_type == 'text']
                image_regions = [r for r in merged_regions if r.region_type == 'image']

                if text_regions:
                    text_regions = merge_regions_into_lines(
                        text_regions,
                        max_gap_px=int(0.015 * processed_image.shape[1]) or 28,
                        min_y_overlap=0.45
                    )

                # Extract images for preservation
                extracted_images = self._extract_images(original_image, image_regions, image_path)

                # OCR text: regions if we have them, else full image
                if text_regions:
                    ocr_results = self._process_text_regions(processed_image, text_regions)
                else:
                    ocr_results = self._run_ocr_engines(processed_image, "full_image")

                # Select best + fixups
                best_result = self._select_best_ocr_result(ocr_results)
                fixed = self._post_ocr_fixups(best_result.text)
                final_text, _ = self.text_cleaner.clean_text(fixed)

            # ---------------- Language & JP gating (ROMAJI-SAFE) ----------------
            # Remove CJK punctuation first so punctuation alone can't flag 'ja'
            text_for_detection = self._strip_cjk_punct(final_text)
            has_japanese = self._has_kana_or_kanji(text_for_detection)

            if not has_japanese:
                # Force a single English segment and skip JP pipeline
                language_segments = [{'text': final_text, 'language': 'en'}]
                japanese_result = None
            else:
                language_segments = self.language_detector.segment_by_language(final_text)
                japanese_result = self.japanese_processor.process_text(final_text)

            # Stats and language ratio normalization
            text_stats = TextStatistics.get_stats(final_text)
            if not has_japanese:
                lr = text_stats.get('language_ratio', {}) or {}
                lr['english'] = 1.0
                lr['japanese'] = 0.0
                text_stats['language_ratio'] = lr

            # Quality metrics
            overall_confidence = self._calculate_overall_confidence(ocr_results, japanese_result)
            quality_score = self._calculate_quality_score(best_result, text_stats, overall_confidence)

            # Output formats
            html_content = self._generate_html_content(final_text, japanese_result if has_japanese else None)
            markdown_content = self._generate_markdown_content(final_text, Path(image_path).stem)

            # Metadata
            processing_time = time.time() - start_time
            processing_metadata = {
                'processing_date': datetime.now().isoformat(),
                'image_path': image_path,
                'engines_used': [result.engine for result in ocr_results],
                'layout_regions': {
                    'text_regions': len(text_regions) if 'text_regions' in locals() else 0,
                    'image_regions': len(image_regions) if 'image_regions' in locals() else 0
                },
                'processing_config': {
                    'tesseract_enabled': self.tesseract_engine.available,
                    'easyocr_enabled': self.easyocr_engine.available,
                    'japanese_processing': has_japanese
                },
                # single source of truth for UI “Japanese Text:”
                'has_japanese': has_japanese,
            }

            result = ProcessingResult(
                document_id=document_id,
                page_id=None,
                ocr_results=ocr_results,
                best_ocr_result=best_result,
                raw_text=best_result.text,
                cleaned_text=final_text,
                text_regions=text_regions if 'text_regions' in locals() else [],
                image_regions=image_regions if 'image_regions' in locals() else [],
                extracted_images=extracted_images if 'extracted_images' in locals() else [],
                japanese_result=japanese_result if has_japanese else None,
                language_segments=[
                    seg if isinstance(seg, dict) else seg.to_dict()
                    for seg in language_segments
                ],
                text_statistics=text_stats,
                overall_confidence=overall_confidence,
                quality_score=quality_score,
                processing_time=processing_time,
                html_content=html_content,
                markdown_content=markdown_content,
                processing_metadata=processing_metadata
            )

            logger.info(f"OCR processing completed in {processing_time:.2f}s")
            logger.info(f"Confidence: {overall_confidence:.2f}, Quality: {quality_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise

    # -------------------- REGION OCR --------------------

    def _process_text_regions(self, image: np.ndarray, text_regions: List[ImageRegion]) -> List[OCRResult]:
        """
        Process individual text regions and return a single aggregated OCRResult.
        Designed for speed: dynamic throttling, limited PSM tries, conditional EasyOCR.
        """
        if not text_regions:
            return self._run_ocr_engines(image, "full_image")

        # Adaptive size filter — drop tiny boxes by area threshold tied to page size
        H, W = image.shape[:2]
        page_area = H * W
        min_area = max(int(0.0008 * page_area), 80 * 25)  # ~0.08% of page or ~2k px
        regions = [r for r in text_regions if r.area >= min_area]

        # Sort top->bottom then left->right
        regions.sort(key=lambda r: (r.y, r.x))

        # Cap regions for runtime (40); keep top-to-bottom evenly
        MAX_REGIONS = 40
        if len(regions) > MAX_REGIONS:
            idx = np.linspace(0, len(regions) - 1, MAX_REGIONS).astype(int).tolist()
            regions = [regions[i] for i in idx]

        if not regions:
            return self._run_ocr_engines(image, "full_image")

        per_region_results: List[Tuple[ImageRegion, List[OCRResult]]] = []
        total_chars_so_far = 0

        for i, region in enumerate(regions):
            logger.debug(
                f"Processing text region {i + 1}/{len(regions)} "
                f"(x={region.x}, y={region.y}, w={region.width}, h={region.height}, area={region.area})"
            )
            region_img = extract_image_region(image, region)

            # Decide hint
            aspect = region.width / max(1, region.height)
            if region.height < 55 and aspect > 3.0:
                psm = "7"; hint = "line"
            elif region.width < 80 or aspect < 0.8:
                psm = "8"; hint = "word"
            else:
                psm = None; hint = "auto"

            # First: Tesseract (fast)
            tess = self.tesseract_engine.process_image(region_img, psm=psm, region_hint=hint)
            tess.metadata = (tess.metadata or {}) | {
                "region_bbox": {"x": region.x, "y": region.y, "w": region.width, "h": region.height},
                "region_hint": hint
            }
            region_results: List[OCRResult] = [tess]

            # Only if looks hard/weak, try EasyOCR (we don't rely on image-level jp hints)
            weak_tess = (tess.confidence < 0.82 and len((tess.text or "").strip()) < 40)
            if weak_tess and self.easyocr_engine.available:
                try:
                    ez = self.easyocr_engine.process_image(region_img, paragraph=(hint != "word"))
                    ez.metadata = (ez.metadata or {}) | {
                        "region_bbox": {"x": region.x, "y": region.y, "w": region.width, "h": region.height},
                        "region_hint": hint
                    }
                    region_results.append(ez)
                except Exception as e:
                    logger.debug(f"EasyOCR fail on region {i}: {e}")

            per_region_results.append((region, region_results))

            # Cheap early aggregation preview to exit sooner
            if (i + 1) % 10 == 0 or (i + 1) == len(regions):
                preview = self._combine_region_results(per_region_results)
                total_chars_so_far = len(preview.text)
                if preview.confidence >= 0.87 and total_chars_so_far >= 1800:
                    logger.debug("Early exit region OCR (chars=%d, conf=%.2f)", total_chars_so_far, preview.confidence)
                    return [preview]

        combined = self._combine_region_results(per_region_results)
        if not combined.text.strip():
            logger.debug("Region OCR produced no text; running full-page fallback.")
            return self._run_ocr_engines(image, "full_fallback")

        return [combined]

    # -------------------- COMBINE / SCORE --------------------

    def _combine_region_results(
        self,
        per_region_results: List[Tuple[ImageRegion, List[OCRResult]]]
    ) -> OCRResult:
        ordered = sorted(per_region_results, key=lambda item: (item[0].y, item[0].x))

        parts: List[str] = []
        total_time = 0.0
        tot_chars = 0
        conf_acc = 0.0
        boxes: List[Dict] = []
        engines_used = set()

        for region, results in ordered:
            best = None
            best_score = -1.0
            for r in results:
                score = self._score_ocr_result(r)
                if score > best_score:
                    best, best_score = r, score

            if not best or not (best.text or "").strip():
                continue

            text = best.text.strip()
            parts.append(text)
            engines_used.add(best.engine)
            total_time += (best.processing_time or 0.0)

            n = len(text)
            tot_chars += n
            conf_acc += (best.confidence or 0.0) * n

            if best.bounding_boxes:
                for b in best.bounding_boxes:
                    b2 = dict(b)
                    b2["region"] = best.metadata.get("region_bbox") if best.metadata else None
                    boxes.append(b2)

        full_text = "\n".join(parts).strip()
        if not full_text:
            return OCRResult(
                text="", confidence=0.0, processing_time=total_time,
                engine="+".join(sorted(engines_used)) if engines_used else "regions",
                bounding_boxes=[], metadata={"aggregated": True, "regions": len(per_region_results)}
            )

        avg_conf = (conf_acc / max(1, tot_chars))
        return OCRResult(
            text=full_text,
            confidence=float(avg_conf),
            processing_time=total_time,
            engine="+".join(sorted(engines_used)) if engines_used else "regions",
            bounding_boxes=boxes,
            metadata={"aggregated": True, "regions": len(per_region_results)}
        )

    def _run_ocr_engines(
            self,
            image: np.ndarray,
            region_id: str,
            psm_override: Optional[str] = None,
            region_hint: str = "auto",
    ) -> List[OCRResult]:
        """Run OCR engines with content-aware selection (optional PSM override for Tesseract)."""
        results: List[OCRResult] = []

        # quick content analysis (used for engine order & tesseract retry heuristic)
        content_hints = self._analyze_image_content(image)
        engine_priority = self._get_engine_priority(content_hints)

        for engine_name in engine_priority:
            try:
                if engine_name == "tesseract" and self.tesseract_engine.available:
                    # first pass (psm from caller or engine default)
                    tesseract_result = self.tesseract_engine.process_image(image, psm=psm_override)
                    tesseract_result.metadata = (tesseract_result.metadata or {})
                    tesseract_result.metadata["region_id"] = region_id
                    tesseract_result.metadata["content_hints"] = content_hints
                    tesseract_result.metadata["region_hint"] = region_hint
                    results.append(tesseract_result)

                    logger.debug(
                        f"Tesseract {region_id}: {len(tesseract_result.text)} chars, "
                        f"{tesseract_result.confidence:.2f} conf"
                    )

                    # On dense pages, if no explicit PSM override and the first pass is mediocre,
                    # try a single quick retry as a text block (PSM 4). Keep the better one.
                    dense = content_hints.get("edge_density", 0.0) > 0.18
                    if (
                            psm_override is None
                            and region_id in ("full_image", "preflight_full", "full_fallback")
                            and dense
                            and (tesseract_result.confidence < 0.88 or len(tesseract_result.text) < 600)
                    ):
                        retry = self.tesseract_engine.process_image(image, psm="4")
                        retry.metadata = (retry.metadata or {})
                        retry.metadata.update({
                            "region_id": region_id,
                            "content_hints": content_hints,
                            "region_hint": region_hint,
                            "psm_used_retry": "4",
                        })
                        # keep the stronger of the two (confidence, then length as tiebreaker)
                        better = retry if (retry.confidence > tesseract_result.confidence or
                                           (abs(retry.confidence - tesseract_result.confidence) < 1e-3
                                            and len(retry.text) > len(tesseract_result.text))) else tesseract_result
                        if better is not tesseract_result:
                            results[-1] = better  # replace in-place
                            logger.debug(
                                f"Tesseract retry (PSM 4) improved result: "
                                f"{better.confidence:.2f} conf, {len(better.text)} chars"
                            )

                    # early stop: strong & long tesseract
                    if results[-1].confidence >= 0.90 and len(results[-1].text) >= 1200:
                        break

                elif engine_name == "easyocr" and self.easyocr_engine.available:
                    # skip if we already have a solid tesseract result
                    if results and results[-1].engine == "tesseract" and results[-1].confidence >= 0.85:
                        continue

                    # run EasyOCR when Tesseract hasn't produced (or as a secondary view)
                    eo_result = self.easyocr_engine.process_image(image)
                    eo_result.metadata = (eo_result.metadata or {})
                    eo_result.metadata["region_id"] = region_id
                    eo_result.metadata["content_hints"] = content_hints
                    eo_result.metadata["region_hint"] = region_hint
                    results.append(eo_result)

                    logger.debug(
                        f"EasyOCR {region_id}: {len(eo_result.text)} chars, {eo_result.confidence:.2f} conf"
                    )

                    # early stop: strong & long easyocr
                    if eo_result.confidence >= 0.90 and len(eo_result.text) >= 1200:
                        break

            except Exception as e:
                logger.warning(f"{engine_name} failed for {region_id}: {e}")
                continue

        if not results:
            logger.warning(f"All OCR engines failed for {region_id}")
            return [OCRResult(text="", confidence=0.0, processing_time=0.0, engine="failed")]

        # short-circuit if the first result is already very confident
        if results and results[0].confidence > 0.80 and len(results[0].text) > 400:
            return [results[0]]

        return results

    # -------------------- ANALYSIS / SCORING --------------------

    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image content to guide engine selection."""
        try:
            if len(image.shape) == 3:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except cv2.error:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            height, width = gray.shape
            edge_density = self._calculate_edge_density(gray)
            aspect_ratio = width / max(height, 1)
            char_patterns = self._estimate_character_patterns(gray)

            return {
                'edge_density': edge_density,
                'aspect_ratio': aspect_ratio,
                'character_patterns': char_patterns,
                'image_size': (width, height),
                # kept only as an image hint; not used for text/Japanese decisions
                'likely_japanese': char_patterns.get('square_chars', 0) > char_patterns.get('tall_chars', 0)
            }

        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")
            return {
                'edge_density': 0.1,
                'aspect_ratio': 1.0,
                'character_patterns': {'square_chars': 0, 'tall_chars': 0},
                'image_size': (100, 100),
                'likely_japanese': False
            }

    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0)) / float(gray.shape[0] * gray.shape[1])

    def _estimate_character_patterns(self, gray: np.ndarray) -> Dict[str, int]:
        try:
            contours, _ = cv2.findContours(255 - gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            square_chars = 0
            tall_chars = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 5:
                    ar = w / h
                    if 0.7 <= ar <= 1.3:
                        square_chars += 1
                    elif ar < 0.7:
                        tall_chars += 1
            return {'square_chars': square_chars, 'tall_chars': tall_chars, 'total_chars': square_chars + tall_chars}
        except Exception:
            return {'square_chars': 0, 'tall_chars': 0, 'total_chars': 0}

    def _get_engine_priority(self, content_hints: Dict[str, Any]) -> List[str]:
        """
        Prefer Tesseract for clean, typeset English. Use EasyOCR as a fallback
        on visually busy pages.
        """
        edge = content_hints.get('edge_density', 0.0)
        if edge < 0.08:
            return ['tesseract']
        return ['tesseract', 'easyocr']

    def _select_best_ocr_result(self, ocr_results: List[OCRResult]) -> OCRResult:
        if not ocr_results:
            raise ValueError("No OCR results to select from")
        if len(ocr_results) == 1:
            return ocr_results[0]

        scored_results: List[Tuple[float, OCRResult]] = []
        for result in ocr_results:
            score = self._score_ocr_result(result)
            scored_results.append((score, result))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        _, best_result = scored_results[0]

        # Length-aware override
        for _, candidate in scored_results[1:]:
            if len(candidate.text.strip()) >= 3 * len(best_result.text.strip()) and \
               (candidate.confidence + 0.15) >= best_result.confidence:
                best_result = candidate
                break

        return best_result

    def _score_ocr_result(self, result: OCRResult) -> float:
        """
        Score OCR results; detect Japanese from TEXT (kana/kanji), not image hints,
        so romaji won't trigger JP logic.
        """
        score = 0.0
        text = (result.text or "").strip()
        text_length = len(text)
        if text_length == 0:
            return 0.0

        # Base confidence
        score += (result.confidence or 0.0) * 0.5

        # Content quality
        words = text.split()
        word_count = len(words)
        if word_count > 3:
            avg_word_length = text_length / word_count
            if 2 <= avg_word_length <= 12:
                score += 0.2
            elif 1 <= avg_word_length <= 20:
                score += 0.1

        # True JP detection from text
        has_jp = self._has_kana_or_kanji(text)

        # Penalize symbol noise only on non-JP pages
        if not has_jp and text_length > 0:
            special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / text_length
            if special_char_ratio > 0.5:
                score -= 0.1

        # Soft engine preference by page type
        if has_jp and result.engine == 'easyocr':
            score += 0.10
        elif not has_jp and result.engine == 'tesseract':
            score += 0.05

        # Confidence + structure
        if (result.confidence or 0.0) > 0.8 and word_count > 5:
            score += 0.1

        return max(0.0, min(1.0, score))

    # -------------------- IMAGE EXTRACT / QUALITY / OUTPUT --------------------

    def _extract_images(self, image: np.ndarray, image_regions: List[ImageRegion],
                        base_path: str) -> List[Dict[str, Any]]:
        extracted_images: List[Dict[str, Any]] = []
        base_name = Path(base_path).stem

        for i, region in enumerate(image_regions):
            try:
                extracted_img = extract_image_region(image, region)
                img_filename = f"{base_name}_image_{i+1}.png"
                img_path = get_config().EXTRACTED_CONTENT_DIR / img_filename

                if save_image(extracted_img, str(img_path)):
                    image_info = {
                        'filename': img_filename,
                        'path': str(img_path),
                        'region': region.to_dict(),
                        'width': region.width,
                        'height': region.height,
                        'area': region.area,
                        'description': f"Extracted image {i+1} from document"
                    }
                    extracted_images.append(image_info)

            except Exception as e:
                logger.warning(f"Failed to extract image region {i}: {e}")

        return extracted_images

    def _calculate_overall_confidence(self, ocr_results: List[OCRResult],
                                      japanese_result: Optional[JapaneseProcessingResult]) -> float:
        if not ocr_results:
            return 0.0

        avg_ocr_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
        japanese_confidence = japanese_result.confidence_score if japanese_result else 1.0

        if japanese_result:
            overall_confidence = (avg_ocr_confidence * 0.7) + (japanese_confidence * 0.3)
        else:
            overall_confidence = avg_ocr_confidence

        return min(1.0, overall_confidence)

    def _calculate_quality_score(self, best_result: OCRResult, text_stats: Dict,
                                 overall_confidence: float) -> float:
        score = 0.0
        score += overall_confidence * 0.4

        char_count = text_stats.get('characters', 0)
        if char_count > 200:
            score += 0.2
        elif char_count > 50:
            score += 0.15
        elif char_count > 10:
            score += 0.1

        word_count = text_stats.get('words', 0)
        if word_count > 50:
            score += 0.15
        elif word_count > 20:
            score += 0.1
        elif word_count > 5:
            score += 0.05

        sentences = text_stats.get('sentences', 0)
        if sentences > 0 and word_count > 0:
            avg_sentence_length = word_count / sentences
            if 8 <= avg_sentence_length <= 25:
                score += 0.15
            elif 5 <= avg_sentence_length <= 35:
                score += 0.1

        japanese_ratio = text_stats.get('language_ratio', {}).get('japanese', 0)
        if japanese_ratio > 0:
            score += 0.05
        if text_stats.get('language_ratio', {}).get('english', 0) > 0.5:
            score += 0.05

        return min(1.0, score)

    @staticmethod
    def _normalize_ascii_punct(s: str) -> str:
        s = unicodedata.normalize("NFKC", s or "")
        s = s.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
        s = s.replace('—', '-').replace('–', '-')
        return s

    @staticmethod
    def _line_signature(line: str) -> set:
        toks = re.findall(r"[A-Za-z0-9]+", (line or "").lower())
        return set(toks)

    @classmethod
    def _near_dup(cls, a: str, b: str, thresh: float = 0.90) -> bool:
        sa = cls._line_signature(a)
        sb = cls._line_signature(b)
        if not sa or not sb:
            return False
        inter = len(sa & sb)
        union = len(sa | sb)
        j = inter / union if union else 0.0
        return j >= thresh

    @classmethod
    def _post_ocr_fixups(self, text: str) -> str:
        """
        Light, deterministic cleanups:
        - join hyphenated line breaks: 'thrust-\ning' -> 'thrusting'
        - merge soft line wraps when the next line continues the sentence
        - drop consecutive duplicate lines (from overlapping regions)
        - collapse multiple blank lines
        """
        if not text:
            return text

        lines = [ln.rstrip() for ln in text.splitlines()]

        # 1) join hyphenated breaks
        out = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.endswith('-') and i + 1 < len(lines):
                nxt = lines[i + 1].lstrip()
                if nxt and (nxt[0].islower() or nxt[0].isdigit()):
                    out.append(ln[:-1] + nxt)
                    i += 2
                    continue
            out.append(ln)
            i += 1

        # 2) merge soft wraps (short continuation that doesn't end with terminal punctuation)
        merged = []
        for ln in out:
            if merged and ln and ln[0].islower() and not merged[-1].endswith(('.', '!', '?', ':', ';')):
                merged[-1] = (merged[-1] + ' ' + ln).strip()
            else:
                merged.append(ln)

        # 3) drop immediate duplicate lines
        dedup = []
        for ln in merged:
            if not dedup or ln != dedup[-1]:
                dedup.append(ln)

        # 4) collapse blank lines
        final_lines = []
        for ln in dedup:
            if ln.strip() == "":
                if final_lines and final_lines[-1].strip() == "":
                    continue
            final_lines.append(ln)

        return "\n".join(final_lines).strip()

    def _generate_html_content(self, text: str, japanese_result: Optional[JapaneseProcessingResult]) -> str:
        html = text
        if japanese_result:
            html = self.japanese_processor.get_japanese_markup(text, japanese_result)
        html = html.replace('\n\n', '</p><p>').replace('\n', '<br>')
        return f"""<div class="ocr-content"><div class="text-content"><p>{html}</p></div></div>"""

    def _generate_markdown_content(self, text: str, title: str) -> str:
        from utils.text_utils import TextFormatter
        formatter = TextFormatter()
        return formatter.to_markdown(text, title)

    def get_engine_status(self) -> Dict[str, Any]:
        return {
            'tesseract': {
                'available': self.tesseract_engine.available,
                'config': self.tesseract_engine.config if self.tesseract_engine.available else None
            },
            'easyocr': {
                'available': self.easyocr_engine.available,
                'config': self.easyocr_engine.config if self.easyocr_engine.available else None
            },
            'japanese_processor': {
                'available': True,
                'stats': self.japanese_processor.get_processing_stats()
            }
        }


# --------------------------------------------------------------------------------------
# Convenience API (module-level)
# --------------------------------------------------------------------------------------

def process_document(image_path: Union[str, Path], document_id: Optional[int] = None) -> ProcessingResult:
    """
    Convenient function to process a document with OCR.

    - Accepts str or Path
    - Resolves/normalizes the path
    - Validates extension against configured/known raster formats
    - Verifies the file is a readable image (via PIL.verify())
    """
    p = Path(image_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")

    cfg = get_config()
    allowed_exts = set(getattr(cfg, "ALLOWED_IMAGE_EXTS", {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}))
    ext = p.suffix.lower()
    if ext not in allowed_exts:
        allowed_list = ", ".join(sorted(allowed_exts))
        raise ValueError(f"Unsupported image format '{ext}' for {p.name}. Allowed: {allowed_list}")

    if not validate_image_file(str(p)):
        raise ValueError(f"File exists but is not a valid/decodable image: {p}")

    processor = OCRProcessor()
    try:
        return processor.process_document(str(p), document_id)
    except Exception as e:
        logger.error(f"OCR processing failed for '{p}': {e}")
        raise


def process_text(text: str) -> ProcessingResult:
    processor = OCRProcessor()
    return processor.process_text_only(text)


def get_ocr_engines_status() -> Dict[str, Any]:
    processor = OCRProcessor()
    return processor.get_engine_status()


# --------------------------------------------------------------------------------------
# Script Entry
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        print("OCR Processor Test")
        print("=" * 50)

        status = get_ocr_engines_status()
        print("Engine Status:")
        for engine, info in status.items():
            print(f"  {engine}: {'Available' if info['available'] else 'Not Available'}")

        print(f"\nProcessing: {image_path}")
        result = process_document(image_path)

        print(f"\nProcessing Results:")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Overall confidence: {result.overall_confidence:.2f}")
        print(f"  Quality score: {result.quality_score:.2f}")
        print(f"  Text length: {len(result.cleaned_text)} characters")
        print(f"  OCR engines used: {[r.engine for r in result.ocr_results]}")

        if result.japanese_result:
            print(f"  Japanese segments: {len(result.japanese_result.segments)}")
            print(f"  Martial arts terms: {len(result.japanese_result.martial_arts_terms)}")

        print(f"\nExtracted text (first 200 chars):")
        print(result.cleaned_text[:200] + ("..." if len(result.cleaned_text) > 200 else ""))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
