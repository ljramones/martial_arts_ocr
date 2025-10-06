"""
Image preprocessing operations for OCR.

This module contains the main image preprocessing pipeline optimized for OCR
on martial arts documents, including deskewing, denoising, contrast enhancement,
and specialized binarization methods.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

from config import get_config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Main image processing class for OCR preprocessing.

    Provides a comprehensive preprocessing pipeline optimized for
    typewritten and printed martial arts documents.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the image processor.

        Args:
            config_override: Optional dictionary to override default config values
        """
        config = get_config()
        self.config = config.IMAGE_PROCESSING

        # Apply any config overrides
        if config_override:
            self.config.update(config_override)

    def preprocess_for_ocr(self, image: np.ndarray,
                           apply_deskew: Optional[bool] = None,
                           apply_denoise: Optional[bool] = None) -> np.ndarray:
        """
        Apply comprehensive preprocessing pipeline optimized for OCR.

        This pipeline is specifically tuned for dense, typewritten pages
        common in martial arts manuals and documentation.

        Args:
            image: Input image (BGR or grayscale)
            apply_deskew: Override deskew setting from config
            apply_denoise: Override denoise setting from config

        Returns:
            Preprocessed binary image optimized for OCR
        """
        try:
            img = image.copy()

            # 1. Convert BGR to RGB if needed
            if img.ndim == 3 and img.shape[2] == 3:
                # Assume BGR input, convert to RGB for processing
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. Optional resize (helps with typewriter dots)
            scale = float(self.config.get('resize_factor', 1.2))
            if scale and abs(scale - 1.0) > 1e-3:
                img = self.resize_image(img, scale)

            # 3. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

            # 4. Apply deskewing if enabled
            should_deskew = apply_deskew if apply_deskew is not None else self.config.get('deskew', True)
            if should_deskew:
                gray = self.deskew_image(gray)

            # 5. Perspective correction (de-keystoning)
            gray = self._apply_perspective_correction(gray)

            # 6. Denoise if enabled
            should_denoise = apply_denoise if apply_denoise is not None else self.config.get('denoise', True)
            if should_denoise:
                gray = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)

            # 7. Apply adaptive binarization
            binary = self._apply_adaptive_binarization(gray)

            # 8. Light morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 9. Unsharp masking for edge enhancement
            binary = self._apply_unsharp_mask(binary)

            return self.normalize_image(binary)

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image

    def _apply_perspective_correction(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply perspective correction to remove keystone distortion.

        Detects the page contour and applies homography transformation
        to create a rectangular, properly oriented page.

        Args:
            gray: Grayscale image

        Returns:
            Perspective-corrected image
        """
        try:
            # Detect edges
            edges = cv2.Canny(gray, 60, 160)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

            # Find contours
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return gray

            # Find largest contour (likely the page)
            page = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(page, True)
            approx = cv2.approxPolyDP(page, 0.02 * peri, True)

            # Check if we found a quadrilateral that's large enough
            min_area_ratio = 0.25  # Page should be at least 25% of image
            if len(approx) != 4:
                return gray
            if cv2.contourArea(approx) < min_area_ratio * (gray.shape[0] * gray.shape[1]):
                return gray

            # Order the corners properly
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)

            # Calculate target dimensions
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])

            W = int(max(widthA, widthB))
            H = int(max(heightA, heightB))

            # Limit maximum size
            max_height = 2400
            if H > max_height:
                scale = max_height / H
                W = int(W * scale)
                H = int(H * scale)

            # Define destination points
            dst = np.array([
                [0, 0],
                [W - 1, 0],
                [W - 1, H - 1],
                [0, H - 1]
            ], dtype=np.float32)

            # Apply perspective transformation
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (W, H),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)

            logger.debug(f"Applied perspective correction: {gray.shape} -> {warped.shape}")
            return warped

        except Exception as e:
            logger.debug(f"Perspective correction skipped: {e}")
            return gray

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left.

        Args:
            pts: 4x2 array of corner points

        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff to identify corners
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()

        rect[0] = pts[np.argmin(s)]  # Top-left (smallest sum)
        rect[2] = pts[np.argmax(s)]  # Bottom-right (largest sum)
        rect[1] = pts[np.argmin(d)]  # Top-right (smallest diff)
        rect[3] = pts[np.argmax(d)]  # Bottom-left (largest diff)

        return rect

    def _apply_adaptive_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization using Sauvola's method.

        Sauvola binarization is particularly effective for documents
        with varying lighting and typewritten text.

        Args:
            gray: Grayscale image

        Returns:
            Binary image
        """
        try:
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Sauvola binarization
            window_size = 25
            k = 0.2  # Tuning parameter
            R = 128.0  # Dynamic range

            # Calculate local mean and standard deviation
            mean = cv2.boxFilter(enhanced, ddepth=-1, ksize=(window_size, window_size))
            sqmean = cv2.boxFilter((enhanced * enhanced).astype(np.float32),
                                   ddepth=-1, ksize=(window_size, window_size))

            # Calculate variance and standard deviation
            var = np.maximum(sqmean - mean.astype(np.float32) ** 2, 0)
            std = np.sqrt(var)

            # Sauvola threshold
            thresh = (mean.astype(np.float32) * (1 + k * ((std / R) - 1))).astype(np.uint8)
            binary = (enhanced > thresh).astype(np.uint8) * 255

            return binary

        except Exception as e:
            logger.debug(f"Sauvola binarization failed, using fallback: {e}")
            # Fallback to Gaussian adaptive threshold
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10)

    def _apply_unsharp_mask(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges.

        Args:
            image: Input image
            strength: Strength of the sharpening effect

        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, strength, blurred, -0.5, 0)
        return sharpened

    def correct_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Flatten background illumination using division normalization.

        Particularly useful for camera-captured images with uneven lighting.

        Args:
            gray: Grayscale image

        Returns:
            Image with corrected illumination
        """
        try:
            # Estimate background with large Gaussian blur
            kernel_size = max(15, int(0.02 * max(gray.shape)))
            if kernel_size % 2 == 0:
                kernel_size += 1

            background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

            # Avoid division by zero
            background = np.clip(background, 1, 255).astype(np.float32)

            # Divide and normalize
            normalized = (gray.astype(np.float32) / background) * 128.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

            return normalized

        except Exception as e:
            logger.debug(f"Illumination correction skipped: {e}")
            return gray

    def resize_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Resize image by the given factor.

        Args:
            image: Input image
            factor: Scaling factor (>1 for upscale, <1 for downscale)

        Returns:
            Resized image
        """
        if abs(factor - 1.0) < 1e-6:
            return image

        height, width = image.shape[:2]
        new_width = max(1, int(width * factor))
        new_height = max(1, int(height * factor))

        # Use appropriate interpolation
        interpolation = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        logger.debug(f"Resized image: {width}x{height} -> {new_width}x{new_height}")
        return resized

    def deskew_image(self, image: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
        """
        Correct skew/rotation in the image.

        Detects dominant line angles and rotates to align with horizontal/vertical.
        Includes guardrails to prevent false corrections.

        Args:
            image: Input image (grayscale)
            max_angle: Maximum rotation angle to apply (degrees)

        Returns:
            Deskewed image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=100, minLineLength=100, maxLineGap=10)

            if lines is None or len(lines) == 0:
                return image

            # Calculate angles of all lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

                # Normalize to [-90, 90]
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                angles.append(angle)

            if not angles:
                return image

            # Use median angle for robustness
            median_angle = float(np.median(angles))

            # Apply guardrails
            if abs(median_angle) > 85:
                logger.debug(f"Deskew: ignoring near-vertical angle {median_angle:.2f}°")
                return image

            if abs(median_angle) > max_angle:
                logger.debug(f"Deskew: angle {median_angle:.2f}° exceeds maximum")
                return image

            if abs(median_angle) <= 0.5:
                return image

            # Apply rotation with canvas expansion
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            # Calculate new canvas size to avoid cropping
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust translation to center image
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            deskewed = cv2.warpAffine(image, M, (new_w, new_h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

            logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
            return deskewed

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image

    def denoise_image(self, image: np.ndarray, strength: str = 'medium') -> np.ndarray:
        """
        Remove noise from the image.

        Args:
            image: Input image
            strength: Denoising strength ('light', 'medium', 'strong')

        Returns:
            Denoised image
        """
        try:
            # Set parameters based on strength
            params = {
                'light': (6, 7, 21),
                'medium': (10, 7, 21),
                'strong': (15, 10, 25)
            }
            h, template_window, search_window = params.get(strength, params['medium'])

            if len(image.shape) == 2:
                # Grayscale
                denoised = cv2.fastNlMeansDenoising(image, None, h,
                                                    template_window, search_window)
            else:
                # Color
                denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                           template_window, search_window)

            logger.debug(f"Applied {strength} denoising")
            return denoised

        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image

    def enhance_contrast(self, image: np.ndarray,
                         clip_limit: float = 2.0,
                         grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Input image (grayscale)
            clip_limit: Threshold for contrast limiting
            grid_size: Size of grid for histogram equalization

        Returns:
            Contrast-enhanced image
        """
        try:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            logger.debug("Enhanced contrast with CLAHE")
            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to 0-255 range (uint8).

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        return image

    def create_binary_image(self, image: np.ndarray,
                            method: str = 'gaussian',
                            block_size: int = 11,
                            c_value: int = 2) -> np.ndarray:
        """
        Create binary (black and white) version for OCR.

        Args:
            image: Input image
            method: Binarization method ('gaussian', 'mean', 'otsu')
            block_size: Size of pixel neighborhood for adaptive methods
            c_value: Constant subtracted from weighted mean

        Returns:
            Binary image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if method == 'otsu':
                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'mean':
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, block_size, c_value)
            else:  # gaussian
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, block_size, c_value)

            logger.debug(f"Created binary image using {method} method")
            return binary

        except Exception as e:
            logger.warning(f"Binary conversion failed: {e}")
            return image


# Standalone preprocessing functions for specific use cases

def preprocess_for_captions_np(np_img: np.ndarray) -> np.ndarray:
    """
    Aggressive but safe binarization for short/sparse English captions.

    Optimized for figure captions, labels, and other sparse text elements
    that may have thin strokes typical of typewritten documents.

    Args:
        np_img: Input image array

    Returns:
        Binary image optimized for caption OCR
    """
    # Convert to grayscale if needed
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_img.copy()

    # Upscale to help with thin typewritten glyphs
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    # Bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    # Adaptive threshold with parameters tuned for sparse text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)

    return binary


def preprocess_for_fullpage_np(np_img: np.ndarray) -> np.ndarray:
    """
    Full-page document binarization for dense typed pages.

    Fallback preprocessing method for complete document pages with
    dense text content.

    Args:
        np_img: Input image array

    Returns:
        Binary image optimized for full-page OCR
    """
    # Convert to grayscale if needed
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_img.copy()

    # Upscale for better character definition
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    # Light Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold optimized for dense text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)

    return binary


def preprocess_for_japanese_np(np_img: np.ndarray) -> np.ndarray:
    """
    Preprocessing optimized for Japanese text (kanji, hiragana, katakana).

    Japanese characters have more complex strokes and require different
    preprocessing parameters than English text.

    Args:
        np_img: Input image array

    Returns:
        Binary image optimized for Japanese OCR
    """
    # Convert to grayscale if needed
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np_img.copy()

    # Moderate upscale - Japanese characters need clarity but not too much
    gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Light denoising
    gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)

    # Adaptive threshold with parameters tuned for complex characters
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 12)

    return binary