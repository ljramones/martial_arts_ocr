# processors/image_preprocessor.py

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Tuple, Optional, Dict, Any
import logging
import re


class AdvancedImagePreprocessor:
    """
    Advanced preprocessing for challenging scanned documents.
    Handles various scan quality issues including:
    - Poor contrast
    - Skewed text
    - Noise and artifacts
    - Faded or light text
    - Dark backgrounds
    - Mixed quality regions
    - Arbitrary rotation angles
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def preprocess_for_ocr(self, image_path: str,
                           document_type: str = 'auto') -> Dict[str, Any]:
        """
        Comprehensive preprocessing pipeline
        """
        self.logger.info(f"Starting preprocessing for {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        original = image.copy()

        # ROTATION CORRECTION - do this first before other preprocessing
        image = self.detect_and_correct_rotation(image)

        # Step 1: Initial assessment
        quality_metrics = self.assess_image_quality(image)
        self.logger.info(f"Quality metrics: {quality_metrics}")

        # Step 2: Apply appropriate preprocessing based on assessment
        if quality_metrics['needs_enhancement']:
            processed = self.apply_enhancement_pipeline(image, quality_metrics)
        else:
            processed = image

        # Step 3: Try multiple preprocessing variants
        variants = self.create_preprocessing_variants(processed)

        # Step 4: Test each variant with OCR
        best_result = self.select_best_variant(variants, original)

        return best_result

    def detect_and_correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct ANY rotation angle automatically.
        First checks for major rotations (90/180/270), then fine-tunes.
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # FIRST: Check if we need a major rotation (90/180/270) using OCR confidence
            best_major_rotation = 0
            best_major_score = 0

            # Test major rotations
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    test_img = gray
                elif angle == 90:
                    test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    test_img = cv2.rotate(gray, cv2.ROTATE_180)
                else:  # 270
                    test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Quick OCR test on center region
                h, w = test_img.shape[:2]
                sample = test_img[h // 4:3 * h // 4, w // 4:3 * w // 4]

                try:
                    data = pytesseract.image_to_data(sample, output_type=pytesseract.Output.DICT,
                                                     config='--psm 3')
                    confidences = [float(c) for c in data['conf'] if float(c) > 0]
                    text = ' '.join([str(t) for t in data['text'] if t])

                    if confidences and len(text) > 20:
                        avg_conf = np.mean(confidences)
                        # Check for English words as additional validation
                        english_words = len(re.findall(r'\b(?:the|and|of|to|in|is|are|was|were|for|with|that|this)\b',
                                                       text.lower()))
                        score = avg_conf * (1 + english_words * 0.1)

                        self.logger.debug(f"Major rotation test {angle}°: confidence={avg_conf:.2f}, score={score:.2f}")

                        if score > best_major_score:
                            best_major_score = score
                            best_major_rotation = angle
                except Exception as e:
                    self.logger.debug(f"Major rotation test {angle}° failed: {e}")

            # Apply best major rotation if needed
            if best_major_rotation != 0:
                self.logger.info(f"Applying major rotation: {best_major_rotation}°")
                if best_major_rotation == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                elif best_major_rotation == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    gray = cv2.rotate(gray, cv2.ROTATE_180)
                elif best_major_rotation == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # SECOND: Fine-tune with small rotation detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                    minLineLength=100, maxLineGap=10)

            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    angles.append(angle)

                median_angle = np.median(angles)

                # Only apply small corrections now (text should already be roughly horizontal)
                if abs(median_angle) <= 15:
                    rotation_angle = -median_angle
                    if abs(rotation_angle) > 0.5:
                        self.logger.info(f"Fine-tuning rotation by {rotation_angle:.2f} degrees")
                        image = self.rotate_image_any_angle(image, rotation_angle)

        except Exception as e:
            self.logger.error(f"Rotation detection failed: {e}")

        return image


    def rotate_image_any_angle(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by any angle with canvas expansion to prevent cropping.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new canvas size to fit rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix for new canvas
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate with expanded canvas
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def find_best_rotation_by_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Test different rotation angles and find the one with best OCR results.
        Used as fallback when line detection doesn't work.
        """
        best_angle = 0
        best_score = 0
        best_image = image

        # Test range of angles (coarse search first)
        test_angles = list(range(-180, 180, 30))  # Every 30 degrees

        for angle in test_angles:
            rotated = self.rotate_image_any_angle(image, angle)

            # Test OCR on a sample region
            h, w = rotated.shape[:2]
            # Use center region for testing
            y1, y2 = h // 3, 2 * h // 3
            x1, x2 = w // 3, 2 * w // 3
            sample = rotated[y1:y2, x1:x2]

            try:
                # Quick OCR test
                data = pytesseract.image_to_data(sample, output_type=pytesseract.Output.DICT,
                                                 config='--psm 3')

                # Score based on confidence and text length
                confidences = [float(c) for c in data['conf'] if float(c) > 0]
                text = ' '.join([str(t) for t in data['text'] if t])

                if confidences and len(text) > 10:
                    avg_conf = np.mean(confidences)
                    word_count = len(text.split())

                    # Combined score: confidence * log(word count)
                    score = avg_conf * (1 + np.log(word_count + 1))

                    if score > best_score:
                        best_score = score
                        best_angle = angle
                        best_image = rotated

            except Exception as e:
                self.logger.debug(f"OCR test at {angle}° failed: {e}")
                continue

        # Fine-tune if we found a good angle
        if best_angle != 0 and best_score > 0:
            self.logger.info(f"Coarse rotation found: {best_angle}° (score: {best_score:.2f})")

            # Fine search around best angle
            fine_angles = range(best_angle - 10, best_angle + 11, 2)
            original_best = best_angle

            for angle in fine_angles:
                if angle == original_best:
                    continue

                rotated = self.rotate_image_any_angle(image, angle)
                h, w = rotated.shape[:2]
                sample = rotated[h // 3:2 * h // 3, w // 3:2 * w // 3]

                try:
                    data = pytesseract.image_to_data(sample, output_type=pytesseract.Output.DICT,
                                                     config='--psm 3')
                    confidences = [float(c) for c in data['conf'] if float(c) > 0]
                    text = ' '.join([str(t) for t in data['text'] if t])

                    if confidences and len(text) > 10:
                        avg_conf = np.mean(confidences)
                        word_count = len(text.split())
                        score = avg_conf * (1 + np.log(word_count + 1))

                        if score > best_score:
                            best_score = score
                            best_angle = angle
                            best_image = rotated

                except Exception:
                    continue

            self.logger.info(f"Final rotation: {best_angle}° (score: {best_score:.2f})")

        return best_image

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess image quality and determine needed preprocessing
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        metrics = {
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'contrast': gray.max() - gray.min(),
            'noise_level': self.estimate_noise(gray),
            'is_binary': self.is_binary_image(gray),
            'has_skew': self.detect_skew(gray),
            'blur_level': self.estimate_blur(gray)
        }

        # Determine if enhancement needed
        metrics['needs_enhancement'] = (
                metrics['mean_brightness'] < 100 or
                metrics['mean_brightness'] > 200 or
                metrics['contrast'] < 50 or
                metrics['noise_level'] > 0.1 or
                metrics['blur_level'] > 100
        )

        return metrics

    def apply_enhancement_pipeline(self, image: np.ndarray,
                                   metrics: Dict[str, Any]) -> np.ndarray:
        """
        Apply enhancement based on quality metrics
        """
        result = image.copy()

        # Convert to grayscale if not already
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # 1. Denoise if needed
        if metrics['noise_level'] > 0.1:
            self.logger.info("Applying denoising")
            result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)

        # 2. Adjust brightness/contrast
        if metrics['mean_brightness'] < 100 or metrics['mean_brightness'] > 200:
            self.logger.info("Adjusting brightness/contrast")
            result = self.adjust_brightness_contrast(result)

        # 3. Sharpen if blurry
        if metrics['blur_level'] > 100:
            self.logger.info("Applying sharpening")
            result = self.sharpen_image(result)

        # 4. Deskew if needed (small skew correction, not full rotation)
        if metrics.get('has_skew', False):
            self.logger.info("Correcting skew")
            result = self.deskew_image(result)

        return result

    def create_preprocessing_variants(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create multiple preprocessing variants to find best OCR results
        """
        variants = {}

        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Variant 1: Simple threshold
        _, variants['simple_threshold'] = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Variant 2: Adaptive threshold
        variants['adaptive_threshold'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Variant 3: CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants['clahe'] = clahe.apply(gray)

        # Variant 4: Morphological cleaning
        kernel = np.ones((1, 1), np.uint8)
        variants['morphological'] = cv2.morphologyEx(
            variants['simple_threshold'], cv2.MORPH_CLOSE, kernel
        )

        # Variant 5: Inverted (for dark backgrounds)
        variants['inverted'] = cv2.bitwise_not(variants['simple_threshold'])

        # Variant 6: Heavy preprocessing
        enhanced = self.heavy_preprocessing(gray)
        variants['heavy'] = enhanced

        # Variant 7: Original grayscale
        variants['original'] = gray

        return variants

    def heavy_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """
        Aggressive preprocessing for very poor quality scans
        """
        # 1. Denoise
        denoised = cv2.medianBlur(gray, 3)

        # 2. Background removal
        bg_removed = self.remove_background(denoised)

        # 3. Enhance contrast
        enhanced = cv2.equalizeHist(bg_removed)

        # 4. Binarize with careful threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return cleaned

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background using rolling ball algorithm approximation
        """
        # Create structure element (rolling ball)
        kernel_size = 50
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Estimate background
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Subtract background
        result = cv2.subtract(image, background)

        # Normalize
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result.astype(np.uint8)

    def select_best_variant(self, variants: Dict[str, np.ndarray],
                            original: np.ndarray) -> Dict[str, Any]:
        """
        Test each variant with OCR and select best result
        """
        best_result = {
            'image': None,
            'text': '',
            'confidence': 0.0,
            'variant_used': 'none',
            'all_results': {}
        }

        for variant_name, variant_image in variants.items():
            try:
                # Run OCR on variant
                ocr_data = pytesseract.image_to_data(
                    variant_image,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 3'  # Fully automatic page segmentation
                )

                # Extract text and calculate confidence
                text = ' '.join([str(word) for word in ocr_data['text'] if word])
                confidences = [float(c) for c in ocr_data['conf'] if float(c) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0

                # Store result
                best_result['all_results'][variant_name] = {
                    'text_length': len(text),
                    'confidence': avg_confidence,
                    'text_preview': text[:100] if text else ''
                }

                # Update best if this is better
                if len(text) > len(best_result['text']) and avg_confidence > 30:
                    best_result['image'] = variant_image
                    best_result['text'] = text
                    best_result['confidence'] = avg_confidence
                    best_result['variant_used'] = variant_name

            except Exception as e:
                self.logger.warning(f"Failed to OCR variant {variant_name}: {e}")

        # If still no good result, try with different PSM modes
        if not best_result['text']:
            best_result = self.try_alternative_psm_modes(variants['original'])

        return best_result

    def try_alternative_psm_modes(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Try different Tesseract PSM modes for difficult documents
        """
        psm_modes = {
            3: "Fully automatic",
            4: "Single column",
            6: "Uniform block",
            11: "Sparse text",
            12: "Sparse text with OSD"
        }

        best_result = {
            'text': '',
            'confidence': 0,
            'psm_used': None
        }

        for psm, description in psm_modes.items():
            try:
                self.logger.info(f"Trying PSM {psm}: {description}")
                text = pytesseract.image_to_string(
                    image,
                    config=f'--psm {psm}'
                )

                if len(text) > len(best_result['text']):
                    best_result['text'] = text
                    best_result['psm_used'] = psm

            except Exception as e:
                self.logger.warning(f"PSM {psm} failed: {e}")

        return best_result

    def adjust_brightness_contrast(self, image: np.ndarray,
                                   target_mean: int = 127) -> np.ndarray:
        """
        Automatically adjust brightness and contrast
        """
        current_mean = np.mean(image)

        # Calculate adjustment factors
        brightness_delta = target_mean - current_mean

        # Apply adjustments
        adjusted = cv2.convertScaleAbs(image, alpha=1.2, beta=brightness_delta)

        return adjusted

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen blurry text
        """
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct small skews in scans (< 15 degrees)
        For larger rotations, use detect_and_correct_rotation instead
        """
        # Find text angle
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = 90 + angle

        # Only correct small skews here
        if abs(angle) > 15:
            return image

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate image noise level
        """
        # Use Laplacian variance as noise metric
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var() / 10000

    def estimate_blur(self, image: np.ndarray) -> float:
        """
        Estimate blur level
        """
        # Variance of Laplacian - lower values indicate more blur
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def is_binary_image(self, image: np.ndarray) -> bool:
        """
        Check if image is already binary
        """
        unique = np.unique(image)
        return len(unique) <= 2

    def detect_skew(self, image: np.ndarray) -> bool:
        """
        Detect if image has small skew (not major rotation)
        """
        # Simple heuristic - check if text lines are not horizontal
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                angles.append(angle)

            # Check if dominant angle is slightly off horizontal
            dominant_angle = np.median(angles)
            # Small skew detection (5-15 degrees off)
            return 5 < abs(dominant_angle - 90) < 15

        return False