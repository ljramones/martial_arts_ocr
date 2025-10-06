"""
OCR engine wrappers for Tesseract and EasyOCR.
"""
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

from config import get_config
from .ocr_models import OCRResult

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

logger = logging.getLogger(__name__)


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
        """Merge defaults with optional --psm."""
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
        """Process image with Tesseract OCR."""
        if not self.available:
            raise RuntimeError("Tesseract is not available")

        start_time = time.time()
        langs = '+'.join(self.config.get('languages', ['eng']))

        if psm:
            psms = [psm]
        else:
            if region_hint == "word":
                psms = ["8", "6"]
            elif region_hint == "line":
                psms = ["7", "6"]
            else:
                psms = ["6"]

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
            self.reader = easyocr.Reader(languages, gpu=gpu)
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