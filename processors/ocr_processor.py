"""
Main OCR Processor - coordinates engines, layout analysis, and post-processing.
"""
import logging
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

from config import get_config
from utils import (
    ImageProcessor, LayoutAnalyzer, ImageRegion,
    load_image,  # Add load_image here
    save_image, extract_image_region
)
from utils import TextCleaner, LanguageDetector, TextStatistics
from processors.japanese_processor import JapaneseProcessor
from processors.image_preprocessor import AdvancedImagePreprocessor

from .ocr_models import OCRResult, ProcessingResult
from .ocr_engines import TesseractEngine, EasyOCREngine
from .ocr_postprocessor import OCRPostProcessor

logger = logging.getLogger(__name__)

# Set tessdata path
tessdata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tessdata')
os.environ['TESSDATA_PREFIX'] = tessdata_path


class OCRProcessor:
    """Main OCR processing coordinator."""

    def __init__(self, domain: str = "martial_arts"):
        self.config = get_config()
        self.domain = domain

        # Initialize components
        self.image_processor = ImageProcessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()
        self.japanese_processor = JapaneseProcessor()
        self.advanced_preprocessor = AdvancedImagePreprocessor()

        # Initialize OCR engines
        self.tesseract_engine = TesseractEngine()
        self.easyocr_engine = EasyOCREngine()

        # Initialize post-processor with domain
        self.postprocessor = OCRPostProcessor(domain=domain)

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

    def process_document(self, image_path: str, document_id: Optional[int] = None) -> ProcessingResult:
        """Process a complete document with OCR and analysis."""
        start_time = time.time()

        try:
            logger.info(f"Starting OCR processing for: {image_path}")

            # Load and preprocess image - use standalone function
            original_image = load_image(image_path)  # Changed from self.image_processor.load_image
            processed_image = self.image_processor.preprocess_for_ocr(original_image)

            # Check if advanced preprocessing needed
            initial_test = self._quick_ocr_test(processed_image)
            if initial_test['needs_advanced_preprocessing']:
                logger.info("Applying advanced preprocessing")
                preprocessing_result = self.advanced_preprocessor.preprocess_for_ocr(
                    image_path, document_type='auto'
                )
                if preprocessing_result.get('image') is not None:
                    processed_image = preprocessing_result['image']

            # Analyze layout
            text_regions = self.layout_analyzer.detect_text_regions(processed_image)
            image_regions = self.layout_analyzer.detect_image_regions(processed_image)

            # Extract images if needed
            extracted_images = self._extract_images(original_image, image_regions, image_path)

            # Run OCR
            if self._should_use_full_page_ocr(text_regions, processed_image):
                logger.info("Using full-page OCR")
                ocr_results = self._run_full_page_ocr(processed_image)
            else:
                ocr_results = self._process_text_regions(processed_image, text_regions)

            # Select best result
            best_result = self._select_best_ocr_result(ocr_results)

            # Post-process the text
            cleaned_text = self.postprocessor.clean_text(
                best_result.text,
                confidence=best_result.confidence,
                boxes=best_result.bounding_boxes
            )

            # Additional cleaning
            final_text, _ = self.text_cleaner.clean_text(cleaned_text)

            # Language detection and processing
            has_japanese = self._has_kana_or_kanji(final_text)

            if has_japanese:
                language_segments = self.language_detector.segment_by_language(final_text)
                japanese_result = self.japanese_processor.process_text(final_text)
            else:
                language_segments = [{'text': final_text, 'language': 'en'}]
                japanese_result = None

            # Generate statistics and outputs
            text_stats = TextStatistics.get_stats(final_text)
            overall_confidence = self._calculate_overall_confidence(ocr_results, japanese_result)
            quality_score = self._calculate_quality_score(best_result, text_stats, overall_confidence)

            # Generate output formats
            html_content = self._generate_html_content(final_text, japanese_result)
            markdown_content = self._generate_markdown_content(final_text, Path(image_path).stem)

            # Build result
            processing_time = time.time() - start_time

            return ProcessingResult(
                document_id=document_id,
                page_id=None,
                ocr_results=ocr_results,
                best_ocr_result=best_result,
                raw_text=best_result.text,
                cleaned_text=final_text,
                text_regions=text_regions,
                image_regions=image_regions,
                extracted_images=extracted_images,
                japanese_result=japanese_result,
                language_segments=language_segments,
                text_statistics=text_stats,
                overall_confidence=overall_confidence,
                quality_score=quality_score,
                processing_time=processing_time,
                html_content=html_content,
                markdown_content=markdown_content,
                processing_metadata={
                    'processing_date': datetime.now().isoformat(),
                    'image_path': image_path,
                    'engines_used': [r.engine for r in ocr_results],
                    'domain': self.domain,
                    'has_japanese': has_japanese,
                }
            )

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise

    def _quick_ocr_test(self, image: np.ndarray) -> Dict[str, Any]:
        """Quick OCR test to determine if advanced preprocessing is needed."""
        if pytesseract is None:
            return {
                'text': '',
                'confidence': 0.0,
                'needs_advanced_preprocessing': True
            }

        try:
            # Try a quick OCR with basic settings
            text = pytesseract.image_to_string(image, config='--psm 3')
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            confidences = [float(c) for c in data['conf'] if float(c) > 0]
            avg_conf = np.mean(confidences) if confidences else 0.0

            # Determine if we need advanced preprocessing
            needs_preprocessing = (
                len(text.strip()) < 20 or  # Very little text
                avg_conf < 50 or  # Low confidence
                len(confidences) < 5  # Very few detected words
            )

            return {
                'text': text,
                'confidence': avg_conf,
                'needs_advanced_preprocessing': needs_preprocessing
            }
        except Exception as e:
            logger.debug(f"Quick OCR test failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'needs_advanced_preprocessing': True
            }

    def _should_use_full_page_ocr(self, text_regions: List[ImageRegion], image: np.ndarray) -> bool:
        """Determine if we should skip regions and use full page OCR instead."""
        # If no regions or very few regions found
        if len(text_regions) <= 2:
            logger.info(f"Using full-page OCR: only {len(text_regions)} text regions found")
            return True

        # If regions cover less than 20% of the page
        image_area = image.shape[0] * image.shape[1]
        regions_area = sum(r.area for r in text_regions)
        coverage = regions_area / image_area if image_area > 0 else 0
        if coverage < 0.2:
            logger.info(f"Using full-page OCR: regions only cover {coverage:.1%} of page")
            return True

        return False

    def _run_full_page_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run full page OCR with multiple PSM modes."""
        ocr_results = []

        if self.tesseract_engine.available:
            # Try multiple PSM modes for full page
            for psm in ["11", "3", "6"]:
                try:
                    result = self.tesseract_engine.process_image(image, psm=psm)
                    result.metadata = {'full_page': True, 'psm': psm}
                    ocr_results.append(result)
                except Exception as e:
                    logger.debug(f"PSM {psm} failed: {e}")

        if self.easyocr_engine.available and len(ocr_results) == 0:
            try:
                result = self.easyocr_engine.process_image(image)
                result.metadata = {'full_page': True}
                ocr_results.append(result)
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")

        if not ocr_results:
            # Return a default empty result
            ocr_results.append(OCRResult(
                text="",
                confidence=0.0,
                processing_time=0.0,
                engine='none',
                bounding_boxes=[],
                metadata={}
            ))

        return ocr_results

    def _process_text_regions(self, image: np.ndarray, text_regions: List[ImageRegion]) -> List[OCRResult]:
        """Process text regions."""
        if not text_regions:
            return self._run_full_page_ocr(image)

        # For now, just run full page OCR
        # You can re-implement the complex region logic from the original if needed
        return self._run_full_page_ocr(image)

    def _has_kana_or_kanji(self, s: str) -> bool:
        """Check if text contains Japanese characters."""
        return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u9FFF\uFF66-\uFF9D]", s or ""))

    def _select_best_ocr_result(self, results: List[OCRResult]) -> OCRResult:
        """Select the best OCR result from multiple engines."""
        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=0.0,
                engine='none',
                bounding_boxes=[],
                metadata={}
            )

        if len(results) == 1:
            return results[0]

        # Score each result
        scored = []
        for result in results:
            score = result.confidence * 0.5  # Base on confidence

            # Bonus for text length
            if len(result.text) > 100:
                score += 0.2

            # Bonus for reasonable word count
            words = result.text.split()
            if 10 < len(words) < 1000:
                score += 0.1

            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _extract_images(self, image: np.ndarray, image_regions: List[ImageRegion],
                        base_path: str) -> List[Dict[str, Any]]:
        """Extract images from regions."""
        extracted_images = []
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
                                      japanese_result: Optional[Any]) -> float:
        """Calculate overall confidence score."""
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
        """Calculate quality score for the OCR result."""
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

        return min(1.0, score)

    def _generate_html_content(self, text: str, japanese_result: Optional[Any]) -> str:
        """Generate HTML content."""
        html = text or ""
        try:
            if japanese_result:
                html = self.japanese_processor.get_japanese_markup(html, japanese_result)
            else:
                html = html.replace("\r\n", "\n").replace("\r", "\n")
                html = html.replace("\n\n", "</p><p>").replace("\n", "<br>")
                html = f"<p>{html}</p>"
        except Exception:
            html = (text or "").replace("\r\n", "\n").replace("\r", "\n")
            html = html.replace("\n\n", "</p><p>").replace("\n", "<br>")
            html = f"<p>{html}</p>"

        return f'<div class="ocr-content"><div class="text-content">{html}</div></div>'

    def _generate_markdown_content(self, text: str, title: str) -> str:
        """Generate markdown content."""
        from utils import TextFormatter
        formatter = TextFormatter()
        return formatter.to_markdown(text or "", title or "Document")