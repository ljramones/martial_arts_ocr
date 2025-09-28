"""
Image processing utilities for Martial Arts OCR.
Handles image preprocessing, enhancement, and manipulation for better OCR results.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class ImageRegion:
    """Represents a rectangular region in an image."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    region_type: str = "unknown"  # "text", "image", "diagram", etc.

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height,
            'confidence': self.confidence, 'type': self.region_type, 'area': self.area
        }


@dataclass
class ImageInfo:
    """Information about an image file."""
    width: int
    height: int
    channels: int
    dtype: str
    file_size: int
    format: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width, 'height': self.height, 'channels': self.channels,
            'dtype': self.dtype, 'file_size': self.file_size, 'format': self.format
        }


class ImageProcessor:
    """Main image processing class for OCR preprocessing."""

    def __init__(self):
        self.config = config.IMAGE_PROCESSING

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load with OpenCV (BGR format)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            logger.debug(f"Loaded image: {path.name} ({image.shape})")
            return image

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def get_image_info(self, image_path: str) -> ImageInfo:
        """Get detailed information about an image."""
        try:
            path = Path(image_path)

            # Load with PIL for format info
            with Image.open(path) as pil_img:
                format_info = pil_img.format

            # Load with OpenCV for detailed analysis
            cv_img = self.load_image(image_path)

            return ImageInfo(
                width=cv_img.shape[1],
                height=cv_img.shape[0],
                channels=cv_img.shape[2] if len(cv_img.shape) > 2 else 1,
                dtype=str(cv_img.dtype),
                file_size=path.stat().st_size,
                format=format_info or "unknown"
            )

        except Exception as e:
            logger.error(f"Failed to get image info for {image_path}: {e}")
            raise

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline optimized for OCR."""
        try:
            processed = image.copy()

            # Convert to RGB if needed
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # Resize if configured
            if self.config.get('resize_factor', 1.0) != 1.0:
                processed = self.resize_image(processed, self.config['resize_factor'])

            # Convert to grayscale for processing
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed

            # Apply preprocessing steps
            if self.config.get('deskew', True):
                gray = self.deskew_image(gray)

            if self.config.get('denoise', True):
                gray = self.denoise_image(gray)

            if self.config.get('enhance_contrast', True):
                gray = self.enhance_contrast(gray)

            # Ensure image is in valid range
            processed = self.normalize_image(gray)

            logger.debug("Image preprocessing completed")
            return processed

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image

    def resize_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Resize image by the given factor."""
        if factor == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = int(width * factor)
        new_height = int(height * factor)

        # Use appropriate interpolation
        interpolation = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        logger.debug(f"Resized image: {width}x{height} -> {new_width}x{new_height}")
        return resized

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew/rotation in the image."""
        try:
            # Find text lines using HoughLinesP
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                    minLineLength=100, maxLineGap=10)

            if lines is None or len(lines) == 0:
                return image

            # Calculate angles of all lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # Get median angle (more robust than mean)
            median_angle = np.median(angles)

            # Only correct if angle is significant (> 0.5 degrees)
            if abs(median_angle) > 0.5:
                # Rotate image
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

                # Calculate new dimensions to avoid cropping
                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_width = int((height * sin) + (width * cos))
                new_height = int((height * cos) + (width * sin))

                # Adjust translation
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]

                deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
                return deskewed

            return image

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from the image."""
        try:
            # Apply Non-local Means Denoising
            if len(image.shape) == 2:  # Grayscale
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            else:  # Color
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            logger.debug("Applied denoising")
            return denoised

        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better OCR."""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            logger.debug("Enhanced contrast with CLAHE")
            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to 0-255 range."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized

    def create_binary_image(self, image: np.ndarray) -> np.ndarray:
        """Create binary (black and white) version for OCR."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Use adaptive thresholding for better results with varying lighting
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            logger.debug("Created binary image")
            return binary

        except Exception as e:
            logger.warning(f"Binary conversion failed: {e}")
            return image


class LayoutAnalyzer:
    """Analyze image layout to detect text and image regions."""

    def __init__(self):
        self.config = config.LAYOUT_DETECTION

    def detect_text_regions(self, image: np.ndarray) -> List[ImageRegion]:
        """Detect regions containing text."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Use MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=self.config.get('text_block_min_area', 1000),
                _max_area=int(gray.shape[0] * gray.shape[1] * 0.3)
            )

            regions, _ = mser.detectRegions(gray)
            text_regions = []

            for region in regions:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

                # Filter by size and aspect ratio
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                if (area >= self.config.get('text_block_min_area', 1000) and
                        0.1 <= aspect_ratio <= 20):  # Reasonable aspect ratios for text

                    text_regions.append(ImageRegion(
                        x=x, y=y, width=w, height=h,
                        region_type="text",
                        confidence=0.8  # MSER is generally reliable for text
                    ))

            logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions

        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def detect_image_regions(self, image: np.ndarray) -> List[ImageRegion]:
        """Detect regions containing images/diagrams."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Find contours that might be images
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_regions = []
            min_area = self.config.get('image_block_min_area', 2500)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check if region looks like an image (not too elongated)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:
                        image_regions.append(ImageRegion(
                            x=x, y=y, width=w, height=h,
                            region_type="image",
                            confidence=0.6
                        ))

            logger.debug(f"Detected {len(image_regions)} image regions")
            return image_regions

        except Exception as e:
            logger.error(f"Image region detection failed: {e}")
            return []

    def merge_overlapping_regions(self, regions: List[ImageRegion],
                                  overlap_threshold: float = 0.3) -> List[ImageRegion]:
        """Merge overlapping regions to avoid duplicates."""
        if not regions:
            return []

        merged = []
        remaining = regions.copy()

        while remaining:
            current = remaining.pop(0)
            overlapping = [current]

            # Find overlapping regions
            i = 0
            while i < len(remaining):
                if self._calculate_overlap(current, remaining[i]) > overlap_threshold:
                    overlapping.append(remaining.pop(i))
                else:
                    i += 1

            # Merge overlapping regions
            if len(overlapping) > 1:
                merged_region = self._merge_regions(overlapping)
                merged.append(merged_region)
            else:
                merged.append(current)

        return merged

    def _calculate_overlap(self, region1: ImageRegion, region2: ImageRegion) -> float:
        """Calculate overlap ratio between two regions."""
        x1_min, y1_min, x1_max, y1_max = region1.bbox
        x2_min, y2_min, x2_max, y2_max = region2.bbox

        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection_area = x_overlap * y_overlap

        # Calculate union
        union_area = region1.area + region2.area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def _merge_regions(self, regions: List[ImageRegion]) -> ImageRegion:
        """Merge multiple regions into one."""
        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)

        # Use the most common type and highest confidence
        types = [r.region_type for r in regions]
        most_common_type = max(set(types), key=types.count)
        max_confidence = max(r.confidence for r in regions)

        return ImageRegion(
            x=min_x, y=min_y,
            width=max_x - min_x, height=max_y - min_y,
            region_type=most_common_type,
            confidence=max_confidence
        )


# Utility functions
def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """Save image to file with specified quality."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert from RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Save with quality settings
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif path.suffix.lower() == '.png':
            compression = int((100 - quality) / 10)  # Convert to PNG compression (0-9)
            cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        else:
            cv2.imwrite(str(path), image_bgr)

        logger.debug(f"Saved image to: {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False


def extract_image_region(image: np.ndarray, region: ImageRegion) -> np.ndarray:
    """Extract a specific region from an image."""
    try:
        x, y, x2, y2 = region.bbox

        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        x2 = max(x, min(x2, width))
        y2 = max(y, min(y2, height))

        extracted = image[y:y2, x:x2]
        logger.debug(f"Extracted region: {region.region_type} at ({x},{y}) size {x2 - x}x{y2 - y}")
        return extracted

    except Exception as e:
        logger.error(f"Failed to extract region: {e}")
        return image


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (200, 300)) -> np.ndarray:
    """Create a thumbnail of the image."""
    try:
        height, width = image.shape[:2]
        target_width, target_height = size

        # Calculate scaling to maintain aspect ratio
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        logger.debug(f"Created thumbnail: {width}x{height} -> {new_width}x{new_height}")
        return thumbnail

    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return image


def validate_image_file(file_path: str) -> bool:
    """Validate if file is a supported image format."""
    try:
        path = Path(file_path)
        if not path.exists():
            return False

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        if path.suffix.lower() not in valid_extensions:
            return False

        # Try to load the image
        with Image.open(path) as img:
            img.verify()  # Verify it's a valid image

        return True

    except Exception:
        return False