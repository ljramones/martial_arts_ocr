"""
Content Extractor for Martial Arts OCR
Extracts and separates text and image content from scanned documents.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import json
from datetime import datetime

from config import get_config
from utils.image_utils import ImageProcessor, LayoutAnalyzer, ImageRegion, save_image, extract_image_region

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted image from the document."""
    region: ImageRegion
    image_data: np.ndarray
    image_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    image_type: str = "unknown"  # "diagram", "photo", "illustration", "chart"
    confidence: float = 0.0
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'region': self.region.to_dict(),
            'image_path': self.image_path,
            'thumbnail_path': self.thumbnail_path,
            'image_type': self.image_type,
            'confidence': self.confidence,
            'description': self.description,
            'width': self.region.width,
            'height': self.region.height,
            'x': self.region.x,
            'y': self.region.y
        }


@dataclass
class ExtractedText:
    """Represents extracted text from the document."""
    region: ImageRegion
    text_content: str
    language: str = "unknown"
    confidence: float = 0.0
    text_type: str = "paragraph"  # "paragraph", "title", "caption", "list"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'region': self.region.to_dict(),
            'text_content': self.text_content,
            'language': self.language,
            'confidence': self.confidence,
            'text_type': self.text_type,
            'word_count': len(self.text_content.split()) if self.text_content else 0,
            'char_count': len(self.text_content) if self.text_content else 0
        }


@dataclass
class ContentExtractionResult:
    """Results from content extraction process."""
    original_image_path: str
    extracted_text: List[ExtractedText]
    extracted_images: List[ExtractedImage]
    layout_regions: List[ImageRegion]
    processing_time: float
    confidence_score: float
    has_mixed_content: bool
    extraction_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_image_path': self.original_image_path,
            'extracted_text': [text.to_dict() for text in self.extracted_text],
            'extracted_images': [img.to_dict() for img in self.extracted_images],
            'layout_regions': [region.to_dict() for region in self.layout_regions],
            'processing_time': self.processing_time,
            'confidence_score': self.confidence_score,
            'has_mixed_content': self.has_mixed_content,
            'extraction_metadata': self.extraction_metadata,
            'summary': {
                'total_text_regions': len(self.extracted_text),
                'total_image_regions': len(self.extracted_images),
                'total_characters': sum(len(text.text_content) for text in self.extracted_text),
                'total_words': sum(len(text.text_content.split()) for text in self.extracted_text),
                'mixed_content': self.has_mixed_content
            }
        }


class ContentExtractor:
    """Main content extraction class."""

    def __init__(self):
        self.config = get_config()
        self.image_processor = ImageProcessor()
        self.layout_analyzer = LayoutAnalyzer()

        # Content extraction settings
        self.min_text_area = self.config.LAYOUT_DETECTION.get('text_block_min_area', 1000)
        self.min_image_area = self.config.LAYOUT_DETECTION.get('image_block_min_area', 2500)
        self.overlap_threshold = 0.3

    def extract_content(self, image_path: str, output_dir: Optional[str] = None) -> ContentExtractionResult:
        """
        Extract text and image content from a document image.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save extracted images (optional)

        Returns:
            ContentExtractionResult with all extracted content
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting content extraction for: {image_path}")

            # Load and preprocess image
            original_image = self.image_processor.load_image(image_path)
            processed_image = self.image_processor.preprocess_for_ocr(original_image)

            # Analyze layout to identify regions
            text_regions = self.layout_analyzer.detect_text_regions(processed_image)
            image_regions = self.layout_analyzer.detect_image_regions(processed_image)

            # Merge overlapping regions
            all_regions = text_regions + image_regions
            merged_regions = self.layout_analyzer.merge_overlapping_regions(all_regions)

            # Classify regions more accurately
            classified_regions = self._classify_regions(processed_image, merged_regions)

            # Extract text content
            extracted_text = self._extract_text_content(processed_image, classified_regions['text'])

            # Extract image content
            extracted_images = self._extract_image_content(
                original_image,
                classified_regions['image'],
                output_dir
            )

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_overall_confidence(extracted_text, extracted_images)
            has_mixed_content = len(extracted_text) > 0 and len(extracted_images) > 0

            # Create metadata
            extraction_metadata = {
                'processing_date': datetime.now().isoformat(),
                'image_dimensions': {
                    'width': original_image.shape[1],
                    'height': original_image.shape[0],
                    'channels': original_image.shape[2] if len(original_image.shape) > 2 else 1
                },
                'preprocessing_applied': {
                    'deskew': self.config.IMAGE_PROCESSING.get('deskew', True),
                    'denoise': self.config.IMAGE_PROCESSING.get('denoise', True),
                    'contrast_enhancement': self.config.IMAGE_PROCESSING.get('enhance_contrast', True)
                },
                'region_detection': {
                    'total_regions_found': len(all_regions),
                    'regions_after_merge': len(merged_regions),
                    'text_regions': len(classified_regions['text']),
                    'image_regions': len(classified_regions['image'])
                }
            }

            result = ContentExtractionResult(
                original_image_path=image_path,
                extracted_text=extracted_text,
                extracted_images=extracted_images,
                layout_regions=merged_regions,
                processing_time=processing_time,
                confidence_score=confidence_score,
                has_mixed_content=has_mixed_content,
                extraction_metadata=extraction_metadata
            )

            logger.info(f"Content extraction completed in {processing_time:.2f}s")
            logger.info(f"Found {len(extracted_text)} text regions and {len(extracted_images)} image regions")

            return result

        except Exception as e:
            logger.error(f"Content extraction failed for {image_path}: {e}")
            raise

    def _classify_regions(self, image: np.ndarray, regions: List[ImageRegion]) -> Dict[str, List[ImageRegion]]:
        """
        Classify regions into text and image categories with simplified, robust logic.

        Args:
            image: Preprocessed image
            regions: List of detected regions

        Returns:
            Dictionary with 'text' and 'image' region lists
        """
        if not regions:
            return {'text': [], 'image': []}

        text_regions = []
        image_regions = []

        # Calculate global statistics for relative thresholds
        all_areas = [r.area for r in regions]
        all_aspect_ratios = [r.width / max(r.height, 1) for r in regions]

        median_area = np.median(all_areas)
        median_aspect_ratio = np.median(all_aspect_ratios)

        for region in regions:
            try:
                # Extract region from image
                region_image = extract_image_region(image, region)

                # Calculate simplified features
                features = self._calculate_simple_features(region_image, region, median_area, median_aspect_ratio)

                # Classify based on simplified logic
                classification = self._simple_classification(features)

                if classification == 'text':
                    region.region_type = 'text'
                    text_regions.append(region)
                else:
                    region.region_type = 'image'
                    image_regions.append(region)

            except Exception as e:
                logger.warning(f"Failed to classify region {region}: {e}")
                # Default to image for failed regions (safer choice)
                region.region_type = 'image'
                image_regions.append(region)

        # Filter by minimum area requirements
        text_regions = [r for r in text_regions if r.area >= self.min_text_area]
        image_regions = [r for r in image_regions if r.area >= self.min_image_area]

        logger.debug(f"Classified {len(text_regions)} text regions and {len(image_regions)} image regions")

        return {
            'text': text_regions,
            'image': image_regions
        }

    def _calculate_simple_features(self, region_image: np.ndarray, region: ImageRegion,
                                   median_area: float, median_aspect_ratio: float) -> Dict[str, float]:
        """
        Calculate simplified features for region classification.
        Uses only 3 robust indicators instead of 9+ complex metrics.

        Args:
            region_image: Image data of the region
            region: Region metadata
            median_area: Median area of all regions (for relative thresholds)
            median_aspect_ratio: Median aspect ratio of all regions

        Returns:
            Dictionary of calculated features
        """
        try:
            # Convert to grayscale if needed
            if len(region_image.shape) == 3:
                gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = region_image

            height, width = gray.shape

            # FEATURE 1: Aspect ratio relative to document norm
            aspect_ratio = width / max(height, 1)
            relative_aspect_ratio = aspect_ratio / max(median_aspect_ratio, 0.1)

            # FEATURE 2: Edge density (simplified Canny edge detection)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height) if width * height > 0 else 0

            # FEATURE 3: Area relative to document norm
            relative_area = region.area / max(median_area, 1)

            return {
                'relative_aspect_ratio': relative_aspect_ratio,
                'edge_density': edge_density,
                'relative_area': relative_area,
                'absolute_area': region.area,
                'width': width,
                'height': height
            }

        except Exception as e:
            logger.warning(f"Simple feature calculation failed: {e}")
            return {
                'relative_aspect_ratio': 1.0,
                'edge_density': 0.0,
                'relative_area': 1.0,
                'absolute_area': region.area,
                'width': region.width,
                'height': region.height
            }

    def _simple_classification(self, features: Dict[str, float]) -> str:
        """
        Classify a region as text or image using simplified, robust logic.
        Uses relative thresholds that adapt to document characteristics.

        Args:
            features: Calculated region features

        Returns:
            'text' or 'image'
        """
        # TEXT INDICATORS (each worth 1 point)
        text_score = 0

        # 1. Text tends to be wider than tall (relative to document norm)
        if features['relative_aspect_ratio'] > 1.2:  # 20% wider than document average
            text_score += 1

        # 2. Text has high edge density (lots of character edges)
        if features['edge_density'] > 0.08:  # Simplified threshold
            text_score += 1

        # 3. Text regions are typically smaller than large images
        if features['relative_area'] < 2.0:  # Less than 2x the document average
            text_score += 1

        # IMAGE INDICATORS (negative points)
        image_score = 0

        # 1. Very large regions are likely images
        if features['relative_area'] > 3.0:  # More than 3x document average
            image_score += 1

        # 2. Very low edge density suggests solid image areas
        if features['edge_density'] < 0.02:
            image_score += 1

        # 3. Square regions more likely to be images/diagrams
        if 0.8 <= features['relative_aspect_ratio'] <= 1.2:  # Nearly square
            image_score += 0.5

        # DECISION LOGIC
        final_score = text_score - image_score

        # Default to text if unclear (better for OCR to try than miss)
        return 'text' if final_score >= 0 else 'image'







    def _extract_text_content(self, image: np.ndarray, text_regions: List[ImageRegion]) -> List[ExtractedText]:
        """
        Extract text content from identified text regions.

        Args:
            image: Preprocessed image
            text_regions: List of text regions

        Returns:
            List of extracted text objects
        """
        extracted_texts = []

        for region in text_regions:
            try:
                # Extract region image
                region_image = extract_image_region(image, region)

                # For now, create placeholder text content
                # This will be processed by OCR processor later
                text_content = f"[Text region at ({region.x}, {region.y}) - {region.width}x{region.height}]"

                # Detect language hints
                language = self._detect_language_hints(region_image)

                # Estimate confidence based on image quality
                confidence = self._estimate_text_quality(region_image)

                # Determine text type
                text_type = self._classify_text_type(region)

                extracted_text = ExtractedText(
                    region=region,
                    text_content=text_content,
                    language=language,
                    confidence=confidence,
                    text_type=text_type
                )

                extracted_texts.append(extracted_text)

            except Exception as e:
                logger.warning(f"Failed to extract text from region {region}: {e}")
                continue

        return extracted_texts

    def _extract_image_content(self, image: np.ndarray, image_regions: List[ImageRegion],
                               output_dir: Optional[str] = None) -> List[ExtractedImage]:
        """
        Extract and save image content from identified image regions.

        Args:
            image: Original image
            image_regions: List of image regions
            output_dir: Directory to save extracted images

        Returns:
            List of extracted image objects
        """
        extracted_images = []

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for i, region in enumerate(image_regions):
            try:
                # Extract region image
                region_image = extract_image_region(image, region)

                # Classify image type
                image_type = self._classify_image_type(region_image, region)

                # Generate description
                description = self._generate_image_description(region, image_type)

                # Save image if output directory provided
                image_path = None
                thumbnail_path = None

                if output_dir:
                    # Save full image
                    image_filename = f"extracted_image_{i + 1:03d}.jpg"
                    image_path = output_path / image_filename
                    save_image(region_image, str(image_path))

                    # Create thumbnail
                    thumbnail_filename = f"thumb_extracted_image_{i + 1:03d}.jpg"
                    thumbnail_path = output_path / thumbnail_filename
                    thumbnail = cv2.resize(region_image, (200, 200), interpolation=cv2.INTER_AREA)
                    save_image(thumbnail, str(thumbnail_path))

                    # Store relative paths
                    image_path = str(image_path.relative_to(Path(output_dir).parent))
                    thumbnail_path = str(thumbnail_path.relative_to(Path(output_dir).parent))

                # Calculate confidence
                confidence = self._estimate_image_quality(region_image)

                extracted_image = ExtractedImage(
                    region=region,
                    image_data=region_image,
                    image_path=image_path,
                    thumbnail_path=thumbnail_path,
                    image_type=image_type,
                    confidence=confidence,
                    description=description
                )

                extracted_images.append(extracted_image)

            except Exception as e:
                logger.warning(f"Failed to extract image from region {region}: {e}")
                continue

        return extracted_images

    def _detect_language_hints(self, region_image: np.ndarray) -> str:
        """Simplified language detection based on character shapes."""
        try:
            gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY) if len(region_image.shape) == 3 else region_image

            # Find contours
            contours, _ = cv2.findContours(255 - gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 'unknown'

            # Simple aspect ratio analysis
            aspect_ratios = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 5:  # Ignore tiny contours
                    aspect_ratios.append(w / h)

            if not aspect_ratios:
                return 'unknown'

            avg_aspect_ratio = np.mean(aspect_ratios)

            # Simple heuristic: Japanese characters more square, English more tall
            if 0.7 <= avg_aspect_ratio <= 1.3:
                return 'japanese'
            elif avg_aspect_ratio < 0.7:
                return 'english'
            else:
                return 'mixed'

        except Exception:
            return 'unknown'

    def _estimate_text_quality(self, region_image: np.ndarray) -> float:
        """Simplified text quality estimation."""
        try:
            gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY) if len(region_image.shape) == 3 else region_image

            # Simple sharpness measure
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)

            # Simple contrast measure
            contrast = gray.std() / 255.0

            # Combine with equal weights
            return (sharpness_score + contrast) / 2.0

        except Exception:
            return 0.5

    def _classify_text_type(self, region: ImageRegion) -> str:
        """
        Classify the type of text based on region characteristics.

        Args:
            region: Text region

        Returns:
            Text type ('title', 'paragraph', 'caption', 'list')
        """
        # Simple heuristics based on region size and aspect ratio
        aspect_ratio = region.width / region.height if region.height > 0 else 0

        # Title: wide and relatively short
        if aspect_ratio > 5.0 and region.height < 100:
            return 'title'
        # Caption: narrow and short
        elif region.height < 50:
            return 'caption'
        # List: narrow and tall
        elif aspect_ratio < 2.0 and region.height > 200:
            return 'list'
        # Default to paragraph
        else:
            return 'paragraph'

    def _classify_image_type(self, region_image: np.ndarray, region: ImageRegion) -> str:
        """
        Classify the type of image content.

        Args:
            region_image: Image data
            region: Region metadata

        Returns:
            Image type ('diagram', 'photo', 'illustration', 'chart')
        """
        try:
            gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY) if len(region_image.shape) == 3 else region_image

            # Calculate features for classification
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze geometric shapes
            geometric_shapes = 0
            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Count polygonal shapes (diagrams often have geometric shapes)
                if len(approx) <= 8 and cv2.contourArea(contour) > 500:
                    geometric_shapes += 1

            # Classification heuristics
            if geometric_shapes > 3 and edge_density > 0.1:
                return 'diagram'
            elif edge_density < 0.05:
                return 'photo'
            elif geometric_shapes > 1:
                return 'illustration'
            else:
                return 'chart'

        except Exception:
            return 'unknown'

    def _generate_image_description(self, region: ImageRegion, image_type: str) -> str:
        """
        Generate a description for an extracted image.

        Args:
            region: Image region
            image_type: Classified image type

        Returns:
            Description string
        """
        position = f"Position: ({region.x}, {region.y})"
        size = f"Size: {region.width} × {region.height} pixels"
        type_desc = f"Type: {image_type.title()}"

        return f"{type_desc}. {position}. {size}."

    def _estimate_image_quality(self, region_image: np.ndarray) -> float:
        """
        Estimate image quality for extracted images.

        Args:
            region_image: Image data

        Returns:
            Quality score between 0 and 1
        """
        try:
            gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY) if len(region_image.shape) == 3 else region_image

            # Calculate sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)

            # Calculate noise level (inverse of quality)
            noise_level = np.std(gray) / 255.0
            noise_score = 1.0 - min(noise_level, 1.0)

            # Size score (larger images generally better quality)
            size_score = min((gray.shape[0] * gray.shape[1]) / 10000.0, 1.0)

            # Combine scores
            quality_score = (sharpness_score * 0.5 + noise_score * 0.3 + size_score * 0.2)

            return max(0.0, min(1.0, quality_score))

        except Exception:
            return 0.5

    def _calculate_overall_confidence(self, extracted_text: List[ExtractedText],
                                      extracted_images: List[ExtractedImage]) -> float:
        """
        Calculate overall confidence score for the extraction process.

        Args:
            extracted_text: List of extracted text regions
            extracted_images: List of extracted images

        Returns:
            Overall confidence score between 0 and 1
        """
        if not extracted_text and not extracted_images:
            return 0.0

        text_confidence = np.mean([text.confidence for text in extracted_text]) if extracted_text else 0.0
        image_confidence = np.mean([img.confidence for img in extracted_images]) if extracted_images else 0.0

        # Weight by number of regions
        text_weight = len(extracted_text) / (len(extracted_text) + len(extracted_images))
        image_weight = len(extracted_images) / (len(extracted_text) + len(extracted_images))

        overall_confidence = text_confidence * text_weight + image_confidence * image_weight

        return overall_confidence

    def save_extraction_result(self, result: ContentExtractionResult, output_path: str) -> None:
        """
        Save extraction result to JSON file.

        Args:
            result: Extraction result to save
            output_path: Path to save JSON file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Extraction result saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save extraction result: {e}")
            raise

    def load_extraction_result(self, input_path: str) -> ContentExtractionResult:
        """
        Load extraction result from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Loaded ContentExtractionResult
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct objects from dictionaries
            extracted_text = []
            for text_data in data['extracted_text']:
                region = ImageRegion(**text_data['region'])
                text = ExtractedText(
                    region=region,
                    text_content=text_data['text_content'],
                    language=text_data['language'],
                    confidence=text_data['confidence'],
                    text_type=text_data['text_type']
                )
                extracted_text.append(text)

            extracted_images = []
            for img_data in data['extracted_images']:
                region = ImageRegion(**img_data['region'])
                img = ExtractedImage(
                    region=region,
                    image_data=np.array([]),  # Image data not stored in JSON
                    image_path=img_data['image_path'],
                    thumbnail_path=img_data['thumbnail_path'],
                    image_type=img_data['image_type'],
                    confidence=img_data['confidence'],
                    description=img_data['description']
                )
                extracted_images.append(img)

            layout_regions = [ImageRegion(**region_data) for region_data in data['layout_regions']]

            result = ContentExtractionResult(
                original_image_path=data['original_image_path'],
                extracted_text=extracted_text,
                extracted_images=extracted_images,
                layout_regions=layout_regions,
                processing_time=data['processing_time'],
                confidence_score=data['confidence_score'],
                has_mixed_content=data['has_mixed_content'],
                extraction_metadata=data['extraction_metadata']
            )

            logger.info(f"Extraction result loaded from: {input_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to load extraction result: {e}")
            raise


def extract_content_from_image(image_path: str, output_dir: Optional[str] = None) -> ContentExtractionResult:
    """
    Convenience function to extract content from an image.

    Args:
        image_path: Path to the image file
        output_dir: Directory to save extracted images (optional)

    Returns:
        ContentExtractionResult with extracted content
    """
    extractor = ContentExtractor()
    return extractor.extract_content(image_path, output_dir)


def batch_extract_content(image_paths: List[str], output_base_dir: str) -> List[ContentExtractionResult]:
    """
    Extract content from multiple images in batch.

    Args:
        image_paths: List of image file paths
        output_base_dir: Base directory for output

    Returns:
        List of ContentExtractionResult objects
    """
    extractor = ContentExtractor()
    results = []

    for i, image_path in enumerate(image_paths):
        try:
            # Create individual output directory for each image
            image_name = Path(image_path).stem
            output_dir = Path(output_base_dir) / f"extraction_{i + 1:03d}_{image_name}"

            result = extractor.extract_content(image_path, str(output_dir))

            # Save result to JSON
            json_path = output_dir / "extraction_result.json"
            extractor.save_extraction_result(result, str(json_path))

            results.append(result)

            logger.info(f"Processed {i + 1}/{len(image_paths)}: {image_path}")

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue

    logger.info(f"Batch extraction completed. Processed {len(results)}/{len(image_paths)} images.")
    return results


class ContentExtractionValidator:
    """Validator for content extraction results."""

    @staticmethod
    def validate_extraction_result(result: ContentExtractionResult) -> Dict[str, Any]:
        """
        Validate an extraction result and provide quality metrics.

        Args:
            result: ContentExtractionResult to validate

        Returns:
            Dictionary with validation results and quality metrics
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {},
            'recommendations': []
        }

        try:
            # Basic validation
            if result.confidence_score < 0 or result.confidence_score > 1:
                validation['errors'].append("Confidence score out of range [0, 1]")
                validation['is_valid'] = False

            if result.processing_time < 0:
                validation['errors'].append("Processing time cannot be negative")
                validation['is_valid'] = False

            # Quality metrics
            total_regions = len(result.extracted_text) + len(result.extracted_images)

            validation['quality_metrics'] = {
                'total_regions': total_regions,
                'text_regions': len(result.extracted_text),
                'image_regions': len(result.extracted_images),
                'mixed_content': result.has_mixed_content,
                'overall_confidence': result.confidence_score,
                'processing_time': result.processing_time,
                'content_density': total_regions / max(result.processing_time, 0.1)  # regions per second
            }

            # Warnings and recommendations
            if result.confidence_score < 0.5:
                validation['warnings'].append("Low overall confidence score")
                validation['recommendations'].append("Consider improving image quality or preprocessing")

            if total_regions == 0:
                validation['warnings'].append("No content regions detected")
                validation['recommendations'].append("Check image quality and layout detection settings")

            if result.processing_time > 30:
                validation['warnings'].append("Long processing time")
                validation['recommendations'].append("Consider image size optimization")

            if not result.has_mixed_content and len(result.extracted_images) == 0:
                validation['recommendations'].append("Document appears to be text-only")

            # Text-specific validation
            for i, text in enumerate(result.extracted_text):
                if text.confidence < 0.3:
                    validation['warnings'].append(f"Low confidence for text region {i + 1}")

                if not text.text_content or len(text.text_content.strip()) == 0:
                    validation['warnings'].append(f"Empty text content in region {i + 1}")

            # Image-specific validation
            for i, image in enumerate(result.extracted_images):
                if image.confidence < 0.3:
                    validation['warnings'].append(f"Low confidence for image region {i + 1}")

                if image.region.area < 1000:
                    validation['warnings'].append(f"Very small image region {i + 1}")

        except Exception as e:
            validation['errors'].append(f"Validation failed: {str(e)}")
            validation['is_valid'] = False

        return validation

    @staticmethod
    def generate_quality_report(result: ContentExtractionResult) -> str:
        """
        Generate a human-readable quality report.

        Args:
            result: ContentExtractionResult to analyze

        Returns:
            Formatted quality report string
        """
        validation = ContentExtractionValidator.validate_extraction_result(result)

        report = []
        report.append("Content Extraction Quality Report")
        report.append("=" * 40)

        # Summary
        report.append(f"Overall Status: {'PASS' if validation['is_valid'] else 'FAIL'}")
        report.append(f"Confidence Score: {result.confidence_score:.1%}")
        report.append(f"Processing Time: {result.processing_time:.2f} seconds")
        report.append("")

        # Content breakdown
        metrics = validation['quality_metrics']
        report.append("Content Analysis:")
        report.append(f"  Total Regions: {metrics['total_regions']}")
        report.append(f"  Text Regions: {metrics['text_regions']}")
        report.append(f"  Image Regions: {metrics['image_regions']}")
        report.append(f"  Mixed Content: {'Yes' if metrics['mixed_content'] else 'No'}")
        report.append(f"  Content Density: {metrics['content_density']:.1f} regions/second")
        report.append("")

        # Errors
        if validation['errors']:
            report.append("Errors:")
            for error in validation['errors']:
                report.append(f"  ❌ {error}")
            report.append("")

        # Warnings
        if validation['warnings']:
            report.append("Warnings:")
            for warning in validation['warnings']:
                report.append(f"  ⚠️  {warning}")
            report.append("")

        # Recommendations
        if validation['recommendations']:
            report.append("Recommendations:")
            for rec in validation['recommendations']:
                report.append(f"  💡 {rec}")
            report.append("")

        # Detailed breakdown
        if result.extracted_text:
            report.append("Text Regions:")
            for i, text in enumerate(result.extracted_text):
                report.append(f"  {i + 1}. {text.text_type.title()} - Confidence: {text.confidence:.1%}")
                if text.language != 'unknown':
                    report.append(f"     Language: {text.language}")
            report.append("")

        if result.extracted_images:
            report.append("Image Regions:")
            for i, image in enumerate(result.extracted_images):
                report.append(f"  {i + 1}. {image.image_type.title()} - Confidence: {image.confidence:.1%}")
                report.append(f"     Size: {image.region.width}×{image.region.height}")
            report.append("")

        return "\n".join(report)


# Example usage and testing functions
def main():
    """Example usage of the content extractor."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract content from martial arts documents')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--output-dir', help='Output directory for extracted content')
    parser.add_argument('--save-result', help='Path to save extraction result JSON')
    parser.add_argument('--report', action='store_true', help='Generate quality report')

    args = parser.parse_args()

    try:
        # Extract content
        result = extract_content_from_image(args.image_path, args.output_dir)

        # Save result if requested
        if args.save_result:
            extractor = ContentExtractor()
            extractor.save_extraction_result(result, args.save_result)

        # Generate report if requested
        if args.report:
            report = ContentExtractionValidator.generate_quality_report(result)
            print(report)
        else:
            # Simple summary
            print(f"Extraction completed successfully!")
            print(f"Found {len(result.extracted_text)} text regions and {len(result.extracted_images)} image regions")
            print(f"Overall confidence: {result.confidence_score:.1%}")
            print(f"Processing time: {result.processing_time:.2f} seconds")

    except Exception as e:
        print(f"Extraction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
