"""
OCR Processor for Martial Arts OCR
Coordinates OCR engines, image processing, and Japanese text analysis.
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

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
from utils.image_utils import ImageProcessor, LayoutAnalyzer, ImageRegion, save_image, extract_image_region
from utils.text_utils import TextCleaner, LanguageDetector, TextStatistics
from processors.japanese_processor import JapaneseProcessor, JapaneseProcessingResult

logger = logging.getLogger(__name__)


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


class TesseractEngine:
    """Tesseract OCR engine wrapper."""

    def __init__(self):
        self.config = get_config().OCR_ENGINES['tesseract']
        self.available = TESSERACT_AVAILABLE and self.config.get('enabled', True)

        if self.available:
            self._verify_installation()

    def _verify_installation(self):
        """Verify Tesseract installation and language support."""
        try:
            # Test basic functionality
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")

            # Check available languages
            available_langs = pytesseract.get_languages()
            required_langs = self.config.get('languages', ['eng'])

            missing_langs = set(required_langs) - set(available_langs)
            if missing_langs:
                logger.warning(f"Missing Tesseract languages: {missing_langs}")

        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
            self.available = False

    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with Tesseract OCR."""
        if not self.available:
            raise RuntimeError("Tesseract is not available")

        start_time = time.time()

        try:
            # Prepare Tesseract configuration
            langs = '+'.join(self.config.get('languages', ['eng']))
            config = self.config.get('config', '--psm 6')

            # Run OCR
            ocr_data = pytesseract.image_to_data(
                image,
                lang=langs,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=langs,
                config=config
            )

            # Calculate confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Create bounding boxes
            bounding_boxes = []
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 0:
                    bounding_boxes.append({
                        'text': ocr_data['text'][i],
                        'confidence': int(ocr_data['conf'][i]),
                        'x': int(ocr_data['left'][i]),
                        'y': int(ocr_data['top'][i]),
                        'width': int(ocr_data['width'][i]),
                        'height': int(ocr_data['height'][i])
                    })

            processing_time = time.time() - start_time

            return OCRResult(
                text=text,
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                processing_time=processing_time,
                engine='tesseract',
                bounding_boxes=bounding_boxes,
                metadata={
                    'languages': langs,
                    'config': config,
                    'word_count': len([word for word in ocr_data['text'] if word.strip()])
                }
            )

        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            raise


class EasyOCREngine:
    """EasyOCR engine wrapper."""

    def __init__(self):
        self.config = get_config().OCR_ENGINES['easyocr']
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

    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with EasyOCR."""
        if not self.available or not self.reader:
            raise RuntimeError("EasyOCR is not available")

        start_time = time.time()

        try:
            # Run OCR
            results = self.reader.readtext(image)

            # Extract text and calculate confidence
            text_lines = []
            confidences = []
            bounding_boxes = []

            for (bbox, text, conf) in results:
                confidence_threshold = self.config.get('confidence_threshold', 0.5)
                if conf >= confidence_threshold:
                    text_lines.append(text)
                    confidences.append(conf)

                    # Convert bbox to standard format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))

                    bounding_boxes.append({
                        'text': text,
                        'confidence': conf,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    })

            full_text = '\n'.join(text_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

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
                    'confidence_threshold': self.config.get('confidence_threshold', 0.5)
                }
            )

        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
            raise


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

    def process_document(self, image_path: str, document_id: int = None) -> ProcessingResult:
        """
        Process a complete document with OCR and analysis.

        Args:
            image_path: Path to the image file
            document_id: Optional database document ID

        Returns:
            ProcessingResult with comprehensive analysis
        """
        start_time = time.time()

        try:
            logger.info(f"Starting OCR processing for: {image_path}")

            # Load and preprocess image
            original_image = self.image_processor.load_image(image_path)
            processed_image = self.image_processor.preprocess_for_ocr(original_image)

            # Analyze layout
            text_regions = self.layout_analyzer.detect_text_regions(processed_image)
            image_regions = self.layout_analyzer.detect_image_regions(processed_image)

            # Merge overlapping regions
            all_regions = text_regions + image_regions
            merged_regions = self.layout_analyzer.merge_overlapping_regions(all_regions)

            # Separate text and image regions after merging
            final_text_regions = [r for r in merged_regions if r.region_type == 'text']
            final_image_regions = [r for r in merged_regions if r.region_type == 'image']

            # Extract images for preservation
            extracted_images = self._extract_images(original_image, final_image_regions, image_path)

            # Perform OCR on text regions or full image
            if final_text_regions:
                ocr_results = self._process_text_regions(processed_image, final_text_regions)
            else:
                # Process entire image if no specific text regions detected
                ocr_results = self._process_full_image(processed_image)

            # Select best OCR result
            best_result = self._select_best_ocr_result(ocr_results)

            # Clean and process text
            cleaned_text, cleaning_stats = self.text_cleaner.clean_text(best_result.text)

            # Language detection and segmentation
            language_segments = self.language_detector.segment_by_language(cleaned_text)

            # Japanese text processing
            japanese_result = None
            if any(seg.language == 'ja' for seg in language_segments):
                japanese_result = self.japanese_processor.process_text(cleaned_text)

            # Generate statistics
            text_stats = TextStatistics.get_stats(cleaned_text)

            # Calculate quality metrics
            overall_confidence = self._calculate_overall_confidence(ocr_results, japanese_result)
            quality_score = self._calculate_quality_score(best_result, text_stats, overall_confidence)

            # Generate output formats
            html_content = self._generate_html_content(cleaned_text, japanese_result)
            markdown_content = self._generate_markdown_content(cleaned_text, Path(image_path).stem)

            # Create processing metadata
            processing_time = time.time() - start_time
            processing_metadata = {
                'processing_date': datetime.now().isoformat(),
                'image_path': image_path,
                'engines_used': [result.engine for result in ocr_results],
                'layout_regions': {
                    'text_regions': len(final_text_regions),
                    'image_regions': len(final_image_regions)
                },
                'cleaning_stats': cleaning_stats.to_dict(),
                'processing_config': {
                    'tesseract_enabled': self.tesseract_engine.available,
                    'easyocr_enabled': self.easyocr_engine.available,
                    'japanese_processing': japanese_result is not None
                }
            }

            result = ProcessingResult(
                document_id=document_id,
                page_id=None,  # Single page for now
                ocr_results=ocr_results,
                best_ocr_result=best_result,
                raw_text=best_result.text,
                cleaned_text=cleaned_text,
                text_regions=final_text_regions,
                image_regions=final_image_regions,
                extracted_images=extracted_images,
                japanese_result=japanese_result,
                language_segments=[seg.to_dict() for seg in language_segments],
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

    def _process_text_regions(self, image: np.ndarray, text_regions: List[ImageRegion]) -> List[OCRResult]:
        """Process individual text regions with OCR."""
        ocr_results = []

        for i, region in enumerate(text_regions):
            logger.debug(f"Processing text region {i+1}/{len(text_regions)}")

            # Extract region from image
            region_image = extract_image_region(image, region)

            # Process with available engines
            region_results = self._run_ocr_engines(region_image, f"region_{i}")
            ocr_results.extend(region_results)

        return ocr_results

    def _process_full_image(self, image: np.ndarray) -> List[OCRResult]:
        """Process the entire image with OCR."""
        logger.debug("Processing full image")
        return self._run_ocr_engines(image, "full_image")

    def _run_ocr_engines(self, image: np.ndarray, region_id: str) -> List[OCRResult]:
        """Run OCR engines with content-aware selection."""
        results = []

        # Quick content analysis for engine selection
        content_hints = self._analyze_image_content(image)

        # Determine optimal engine order based on content
        engine_priority = self._get_engine_priority(content_hints)

        for engine_name in engine_priority:
            try:
                if engine_name == 'tesseract' and self.tesseract_engine.available:
                    result = self.tesseract_engine.process_image(image)
                    result.metadata['region_id'] = region_id
                    result.metadata['content_hints'] = content_hints
                    results.append(result)
                    logger.debug(
                        f"Tesseract processed {region_id}: {len(result.text)} chars, {result.confidence:.2f} confidence")

                elif engine_name == 'easyocr' and self.easyocr_engine.available:
                    result = self.easyocr_engine.process_image(image)
                    result.metadata['region_id'] = region_id
                    result.metadata['content_hints'] = content_hints
                    results.append(result)
                    logger.debug(
                        f"EasyOCR processed {region_id}: {len(result.text)} chars, {result.confidence:.2f} confidence")

            except Exception as e:
                logger.warning(f"{engine_name} failed for {region_id}: {e}")
                continue

        # After the engine loop, add:
        if not results:
            logger.warning(f"All OCR engines failed for {region_id}")
            # Return minimal result instead of crashing
            return [OCRResult(text="", confidence=0.0, processing_time=0.0, engine="failed")]

        # If primary engine worked well, skip secondary for performance
        if results and results[0].confidence > 0.8:
            logger.debug(f"High confidence result from {results[0].engine}, skipping additional engines")
            return [results[0]]

        return results

    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image content to guide engine selection."""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            height, width = gray.shape

            # Calculate basic characteristics
            edge_density = self._calculate_edge_density(gray)
            aspect_ratio = width / max(height, 1)

            # Estimate character density patterns
            char_patterns = self._estimate_character_patterns(gray)

            return {
                'edge_density': edge_density,
                'aspect_ratio': aspect_ratio,
                'character_patterns': char_patterns,
                'image_size': (width, height),
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
        """Calculate edge density for content analysis."""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    def _estimate_character_patterns(self, gray: np.ndarray) -> Dict[str, int]:
        """Estimate character patterns to detect likely script type."""
        try:
            # Find contours
            contours, _ = cv2.findContours(255 - gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            square_chars = 0
            tall_chars = 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 5:  # Ignore tiny contours
                    aspect_ratio = w / h
                    if 0.7 <= aspect_ratio <= 1.3:  # Square-ish (Japanese)
                        square_chars += 1
                    elif aspect_ratio < 0.7:  # Tall (English)
                        tall_chars += 1

            return {
                'square_chars': square_chars,
                'tall_chars': tall_chars,
                'total_chars': square_chars + tall_chars
            }

        except Exception:
            return {'square_chars': 0, 'tall_chars': 0, 'total_chars': 0}

    def _get_engine_priority(self, content_hints: Dict[str, Any]) -> List[str]:
        """Determine engine priority based on content analysis."""
        # For Japanese/mixed content, prefer EasyOCR
        if content_hints.get('likely_japanese', False):
            return ['easyocr', 'tesseract']

        # For dense text with high edge density, prefer Tesseract
        elif content_hints.get('edge_density', 0) > 0.15:
            return ['tesseract', 'easyocr']

        # For low-quality/sparse content, try both but EasyOCR first
        else:
            return ['easyocr', 'tesseract']

    def _select_best_ocr_result(self, ocr_results: List[OCRResult]) -> OCRResult:
        """Select best OCR result with improved logic."""
        if not ocr_results:
            raise ValueError("No OCR results to select from")

        if len(ocr_results) == 1:
            return ocr_results[0]

        # Score each result with content awareness
        scored_results = []
        for result in ocr_results:
            score = self._score_ocr_result(result)
            scored_results.append((score, result))

        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        best_result = scored_results[0][1]

        # Log selection reasoning for debugging
        logger.debug(f"OCR result selection:")
        for score, result in scored_results:
            logger.debug(
                f"  {result.engine}: score={score:.3f}, conf={result.confidence:.3f}, chars={len(result.text)}")

        logger.debug(f"Selected: {best_result.engine} (score: {scored_results[0][0]:.3f})")

        return best_result

    def _score_ocr_result(self, result: OCRResult) -> float:
        """Score OCR result with content-aware logic."""
        score = 0.0
        text = result.text.strip()
        text_length = len(text)

        if text_length == 0:
            return 0.0

        # FIXED: Base confidence (weighted heavily)
        score += result.confidence * 0.5

        # FIXED: Content quality over raw length
        words = text.split()
        word_count = len(words)

        # Reward meaningful word structure
        if word_count > 3:
            avg_word_length = text_length / word_count
            if 2 <= avg_word_length <= 12:  # Reasonable word lengths
                score += 0.2
            elif 1 <= avg_word_length <= 20:
                score += 0.1

        # FIXED: Smarter character analysis for Japanese content
        content_hints = result.metadata.get('content_hints', {})
        if content_hints.get('likely_japanese', False):
            # For Japanese content, don't penalize special characters
            japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u9fff')
            if japanese_chars > 0:
                score += 0.15  # Bonus for successfully reading Japanese
        else:
            # For English content, mild penalty for excessive special chars
            if text_length > 0:
                special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / text_length
                if special_char_ratio > 0.5:  # Very high threshold
                    score -= 0.1

        # FIXED: Content-aware engine preferences
        engine = result.engine
        if content_hints.get('likely_japanese', False) and engine == 'easyocr':
            score += 0.1  # EasyOCR better for Japanese
        elif not content_hints.get('likely_japanese', False) and engine == 'tesseract':
            score += 0.05  # Slight Tesseract preference for English

        # FIXED: Confidence consistency check
        if result.confidence > 0.8 and word_count > 5:
            score += 0.1  # Bonus for high-confidence meaningful results

        return max(0.0, min(1.0, score))

    def _extract_images(self, image: np.ndarray, image_regions: List[ImageRegion],
                       base_path: str) -> List[Dict[str, Any]]:
        """Extract and save image regions."""
        extracted_images = []
        base_name = Path(base_path).stem

        for i, region in enumerate(image_regions):
            try:
                # Extract image region
                extracted_img = extract_image_region(image, region)

                # Generate filename
                img_filename = f"{base_name}_image_{i+1}.png"
                img_path = get_config().EXTRACTED_CONTENT_DIR / img_filename

                # Save extracted image
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
                    logger.debug(f"Extracted image: {img_filename}")

            except Exception as e:
                logger.warning(f"Failed to extract image region {i}: {e}")

        return extracted_images

    def _calculate_overall_confidence(self, ocr_results: List[OCRResult],
                                    japanese_result: Optional[JapaneseProcessingResult]) -> float:
        """Calculate overall confidence score."""
        if not ocr_results:
            return 0.0

        # Average OCR confidence
        ocr_confidences = [result.confidence for result in ocr_results]
        avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences)

        # Japanese processing confidence (if applicable)
        japanese_confidence = japanese_result.confidence_score if japanese_result else 1.0

        # Combined confidence (weighted average)
        if japanese_result:
            overall_confidence = (avg_ocr_confidence * 0.7) + (japanese_confidence * 0.3)
        else:
            overall_confidence = avg_ocr_confidence

        return min(1.0, overall_confidence)

    def _calculate_quality_score(self, best_result: OCRResult, text_stats: Dict,
                               overall_confidence: float) -> float:
        """Calculate overall quality score for the processing result."""
        score = 0.0

        # Base confidence component (40%)
        score += overall_confidence * 0.4

        # Text length component (20%)
        char_count = text_stats.get('characters', 0)
        if char_count > 200:
            score += 0.2
        elif char_count > 50:
            score += 0.15
        elif char_count > 10:
            score += 0.1

        # Word count component (15%)
        word_count = text_stats.get('words', 0)
        if word_count > 50:
            score += 0.15
        elif word_count > 20:
            score += 0.1
        elif word_count > 5:
            score += 0.05

        # Sentence structure component (15%)
        sentences = text_stats.get('sentences', 0)
        if sentences > 0 and word_count > 0:
            avg_sentence_length = word_count / sentences
            if 8 <= avg_sentence_length <= 25:  # Reasonable sentence length
                score += 0.15
            elif 5 <= avg_sentence_length <= 35:
                score += 0.1

        # Language detection component (10%)
        japanese_ratio = text_stats.get('language_ratio', {}).get('japanese', 0)
        if japanese_ratio > 0:
            score += 0.05  # Bonus for detected Japanese
        if text_stats.get('language_ratio', {}).get('english', 0) > 0.5:
            score += 0.05  # Bonus for substantial English

        return min(1.0, score)

    def _generate_html_content(self, text: str, japanese_result: Optional[JapaneseProcessingResult]) -> str:
        """Generate HTML content with Japanese markup."""
        html_content = text

        if japanese_result:
            # Apply Japanese markup with tooltips
            html_content = self.japanese_processor.get_japanese_markup(text, japanese_result)

        # Basic HTML formatting
        html_content = html_content.replace('\n\n', '</p><p>')
        html_content = html_content.replace('\n', '<br>')
        html_content = f'<p>{html_content}</p>'

        # Wrap in proper HTML structure
        html_template = f"""
        <div class="ocr-content">
            <div class="text-content">
                {html_content}
            </div>
        </div>
        """

        return html_template.strip()

    def _generate_markdown_content(self, text: str, title: str) -> str:
        """Generate Markdown content."""
        from utils.text_utils import TextFormatter

        formatter = TextFormatter()
        return formatter.to_markdown(text, title)

    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of available OCR engines."""
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

    def process_text_only(self, text: str) -> ProcessingResult:
        """Process text without OCR (for testing or text-only processing)."""
        start_time = time.time()

        try:
            # Create dummy OCR result
            dummy_ocr = OCRResult(
                text=text,
                confidence=1.0,
                processing_time=0.0,
                engine='text_input'
            )

            # Clean and process text
            cleaned_text, cleaning_stats = self.text_cleaner.clean_text(text)

            # Language detection and segmentation
            language_segments = self.language_detector.segment_by_language(cleaned_text)

            # Japanese text processing
            japanese_result = None
            if any(seg.language == 'ja' for seg in language_segments):
                japanese_result = self.japanese_processor.process_text(cleaned_text)

            # Generate statistics
            text_stats = TextStatistics.get_stats(cleaned_text)

            # Generate output formats
            html_content = self._generate_html_content(cleaned_text, japanese_result)
            markdown_content = self._generate_markdown_content(cleaned_text, "Text Input")

            processing_time = time.time() - start_time

            return ProcessingResult(
                document_id=None,
                page_id=None,
                ocr_results=[dummy_ocr],
                best_ocr_result=dummy_ocr,
                raw_text=text,
                cleaned_text=cleaned_text,
                text_regions=[],
                image_regions=[],
                extracted_images=[],
                japanese_result=japanese_result,
                language_segments=[seg.to_dict() for seg in language_segments],
                text_statistics=text_stats,
                overall_confidence=1.0,
                quality_score=0.9,  # High score for text input
                processing_time=processing_time,
                html_content=html_content,
                markdown_content=markdown_content,
                processing_metadata={
                    'processing_date': datetime.now().isoformat(),
                    'input_type': 'text_only',
                    'cleaning_stats': cleaning_stats.to_dict()
                }
            )

        except Exception as e:
            logger.error(f"Text-only processing failed: {e}")
            raise


# Utility functions for easy access
def process_document(image_path: str, document_id: int = None) -> ProcessingResult:
    """
    Convenient function to process a document with OCR.

    Args:
        image_path: Path to the image file
        document_id: Optional database document ID

    Returns:
        ProcessingResult with comprehensive analysis
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not Path(image_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        raise ValueError(f"Unsupported image format: {image_path}")

    processor = OCRProcessor()
    return processor.process_document(image_path, document_id)


def process_text(text: str) -> ProcessingResult:
    """
    Convenient function to process text without OCR.

    Args:
        text: Input text to process

    Returns:
        ProcessingResult with text analysis
    """
    processor = OCRProcessor()
    return processor.process_text_only(text)


def get_ocr_engines_status() -> Dict[str, Any]:
    """
    Get status of available OCR engines.

    Returns:
        Dictionary with engine availability and configuration
    """
    processor = OCRProcessor()
    return processor.get_engine_status()


if __name__ == "__main__":
    # Test the OCR processor
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        print("OCR Processor Test")
        print("=" * 50)

        # Check engine status
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