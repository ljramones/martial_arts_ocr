"""
Layout analysis and region detection for documents.

This module provides sophisticated layout analysis capabilities for detecting
text regions, diagrams, photos, and other content areas in martial arts documents.
Uses multiple detection strategies to handle various types of content.
"""
import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from scipy.signal import find_peaks

from config import get_config
from .core_image import ImageRegion

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Analyze image layout to detect text and image regions.

    Uses multiple detection strategies to identify different types of content
    including text blocks, line art diagrams, photographs, and illustrations.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the layout analyzer.

        Args:
            config_override: Optional dictionary to override default config values
        """
        config = get_config()
        self.config = config.LAYOUT_DETECTION

        # Apply any config overrides
        if config_override:
            self.config.update(config_override)

    def detect_text_regions(self, image: np.ndarray,
                            min_area: Optional[int] = None,
                            max_area_ratio: float = 0.3) -> List[ImageRegion]:
        """
        Detect regions containing text using MSER (Maximally Stable Extremal Regions).

        MSER is particularly effective for detecting text regions as text typically
        forms stable regions across multiple threshold levels.

        Args:
            image: Input image (BGR or grayscale)
            min_area: Minimum area for text regions (pixels)
            max_area_ratio: Maximum area as ratio of image size

        Returns:
            List of ImageRegion objects representing text areas
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Ensure uint8
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Configure MSER parameters
            min_area = min_area or int(self.config.get('text_block_min_area', 1000))
            max_area = int(gray.shape[0] * gray.shape[1] * max_area_ratio)

            # Create MSER detector
            mser = cv2.MSER_create(
                delta=5,  # Delta for threshold levels
                min_area=min_area,  # Minimum component area
                max_area=max_area,  # Maximum component area
                max_variation=0.25,  # Maximum variation in region
                min_diversity=0.2,  # Minimum diversity
                max_evolution=200,  # Maximum number of evolution steps
                area_threshold=1.01,  # Area threshold
                min_margin=0.003,  # Minimum margin
                edge_blur_size=5  # Edge blur kernel size
            )

            # Detect regions
            regions, _ = mser.detectRegions(gray)
            text_regions: List[ImageRegion] = []

            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                area = w * h
                aspect_ratio = (w / h) if h > 0 else 0.0

                # Filter by area and aspect ratio
                if (area >= min_area) and (0.1 <= aspect_ratio <= 20.0):
                    text_regions.append(ImageRegion(
                        x=x, y=y, width=w, height=h,
                        region_type="text",
                        confidence=0.8
                    ))

            # Merge overlapping text regions
            text_regions = self.merge_overlapping_regions(text_regions, overlap_threshold=0.5)

            logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions

        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def detect_image_regions(self, image: np.ndarray) -> List[ImageRegion]:
        """Detect regions containing images (diagrams, photos, illustrations)."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            all_regions = []

            # Try figure-specific detection first
            figure_regions = self._detect_figure_regions(gray)
            all_regions.extend(figure_regions)
            logger.debug(f"Figure detection found {len(figure_regions)} regions")

            # If no figures found, try contour detection
            if len(figure_regions) == 0:
                contour_regions = self._detect_by_contours(gray)
                all_regions.extend(contour_regions)
                logger.debug(f"Contour detection found {len(contour_regions)} regions")

            # Remove overlaps and filter
            merged_regions = self._remove_overlapping_regions(all_regions)
            filtered_regions = self._filter_text_regions(gray, merged_regions)

            logger.info(f"Total image regions detected: {len(filtered_regions)}")
            return filtered_regions

        except Exception as e:
            logger.error(f"Image region detection failed: {e}")
            return []

    def _detect_by_contours(self, gray: np.ndarray) -> List[ImageRegion]:
        """Detect line art and diagrams using contour analysis."""
        try:
            regions = []
            h, w = gray.shape

            # Use Canny edge detection with lower thresholds for line art
            edges = cv2.Canny(gray, 30, 90)

            # Dilate to connect diagram components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # Parameters for filtering
            min_area = 15000  # Minimum size for a diagram
            max_area = int(h * w * 0.4)  # Maximum 40% of page

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = width * height

                if not (min_area < area < max_area):
                    continue

                aspect_ratio = width / height

                # Diagrams typically have moderate aspect ratios
                if not (0.3 < aspect_ratio < 3.5):
                    continue

                # Extract ROI for detailed analysis
                roi = gray[y:y + height, x:x + width]

                # Calculate text-like features to EXCLUDE text regions
                # Text regions have many small, similar-sized components
                _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)

                if num_labels > 2:  # Exclude background
                    component_areas = stats[1:, cv2.CC_STAT_AREA]

                    # Text characteristics: many small, uniform components
                    median_comp_area = np.median(component_areas) if len(component_areas) > 0 else 0
                    num_small_components = np.sum(component_areas < 200)

                    # If it looks like text, skip it
                    if num_small_components > 30 and median_comp_area < 150:
                        logger.debug(f"Skipping text-like region at ({x},{y})")
                        continue

                # Check for line structures (diagrams have clear lines)
                roi_edges = cv2.Canny(roi, 50, 150)
                lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50,
                                        minLineLength=30, maxLineGap=10)

                # Calculate features
                edge_density = np.sum(roi_edges > 0) / area
                has_significant_lines = lines is not None and len(lines) > 5

                # Calculate white space ratio (diagrams have more white space than text)
                white_ratio = np.sum(roi > 200) / area

                # Accept if it has diagram characteristics
                if (has_significant_lines or
                        (0.02 < edge_density < 0.15 and white_ratio > 0.6) or
                        (x < w * 0.6 and white_ratio > 0.7)):  # Left side bias for figures

                    logger.info(f"Diagram found at ({x},{y}): area={area}, "
                                f"edge_density={edge_density:.3f}, white_ratio={white_ratio:.2f}")

                    regions.append(ImageRegion(
                        x=int(x), y=int(y),
                        width=int(width), height=int(height),
                        region_type="diagram",
                        confidence=0.85
                    ))

            # If we found multiple regions, keep only the largest ones
            if len(regions) > 2:
                regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

            return regions

        except Exception as e:
            logger.error(f"Contour detection failed: {e}")
            return []

    def _detect_figure_regions(self, gray: np.ndarray) -> List[ImageRegion]:
        """Detect regions that look like figures/diagrams based on spatial isolation."""
        try:
            h, w = gray.shape
            regions = []

            # Binarize the image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Apply morphological operations to connect figure components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = width * height

                # Size constraints for figures
                if area < 10000 or area > h * w * 0.5:
                    continue

                # Position heuristic: figures often on left half of page
                if x > w * 0.7:  # Too far right, likely text
                    continue

                # Aspect ratio check
                aspect = width / height
                if not (0.4 < aspect < 3.0):
                    continue

                # Check isolation (figures are usually separated from text)
                margin = 20
                x_start = max(0, x - margin)
                y_start = max(0, y - margin)
                x_end = min(w, x + width + margin)
                y_end = min(h, y + height + margin)

                surrounding = gray[y_start:y_end, x_start:x_end]

                # Calculate the density of content around the region
                _, surrounding_binary = cv2.threshold(surrounding, 200, 255, cv2.THRESH_BINARY)
                surrounding_white_ratio = np.sum(surrounding_binary == 255) / surrounding.size

                # Figures are usually surrounded by white space
                if surrounding_white_ratio > 0.75:
                    logger.info(f"Figure detected at ({x},{y}): area={area}, isolation={surrounding_white_ratio:.2f}")
                    regions.append(ImageRegion(
                        x=x, y=y, width=width, height=height,
                        region_type="figure",
                        confidence=0.9
                    ))

            return regions

        except Exception as e:
            logger.error(f"Figure detection failed: {e}")
            return []


    def _detect_by_variance(self, gray: np.ndarray) -> List[ImageRegion]:
        """Detect photo-like regions using variance analysis."""
        try:
            regions = []
            h, w = gray.shape

            # Use larger window and stride to reduce overlaps
            window_size = max(150, min(h, w) // 8)  # Increased from 100 and //10
            stride = window_size  # Changed from window_size // 2 to prevent overlaps

            detected_areas = []

            for y in range(0, h - window_size, stride):
                for x in range(0, w - window_size, stride):
                    roi = gray[y:y + window_size, x:x + window_size]

                    # Calculate variance (photos have moderate variance)
                    variance = np.var(roi)
                    if not (100 < variance < 5000):
                        continue

                    # Check for gradients (photos have smooth gradients)
                    grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_smoothness = np.std(grad_x) * np.std(grad_y)

                    if gradient_smoothness > 50:
                        # Check for significant overlap with existing regions
                        is_new = True
                        for existing in detected_areas:
                            ex, ey, ew, eh = existing
                            # Check if centers are too close
                            cx_new = x + window_size // 2
                            cy_new = y + window_size // 2
                            cx_old = ex + ew // 2
                            cy_old = ey + eh // 2
                            distance = np.sqrt((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2)

                            if distance < window_size:  # Too close to existing region
                                is_new = False
                                break

                        if is_new:
                            # Expand to find full image bounds
                            expanded = self._expand_region(gray, x, y,
                                                           window_size, window_size)
                            if expanded.area > 20000 and expanded.width > 0 and expanded.height > 0:
                                regions.append(expanded)
                                detected_areas.append((expanded.x, expanded.y,
                                                       expanded.width, expanded.height))
                                logger.debug(f"Variance region: area={expanded.area}, "
                                             f"variance={variance:.1f}")

            return regions

        except Exception as e:
            logger.error(f"Variance detection failed: {e}")
            return []

    def _detect_uniform_regions(self, gray: np.ndarray) -> List[ImageRegion]:
        """
        Detect large uniform regions that might be images or backgrounds.

        Uses morphological operations to find large connected regions.

        Args:
            gray: Grayscale image

        Returns:
            List of detected uniform regions
        """
        try:
            regions = []
            h, w = gray.shape

            # Use morphological operations to find large uniform areas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Threshold to binary
            _, binary = cv2.threshold(closed, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find large contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            min_area = max(30000, int(h * w * 0.03))  # At least 3% of image

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = width * height
                aspect_ratio = width / height

                if area > min_area and 0.3 < aspect_ratio < 3.0:
                    roi = gray[y:y + height, x:x + width]

                    # Check if it's not text by looking at variance patterns
                    local_std = np.std(roi)
                    if 10 < local_std < 100:  # Moderate variation, not text
                        logger.debug(f"Uniform region: area={area}, std={local_std:.1f}")
                        regions.append(ImageRegion(
                            x=x, y=y, width=width, height=height,
                            region_type="image",
                            confidence=0.7
                        ))

            return regions

        except Exception as e:
            logger.error(f"Uniform region detection failed: {e}")
            return []

    def _expand_region(self, gray: np.ndarray, x: int, y: int,
                       width: int, height: int) -> ImageRegion:
        """Expand a region to find its full boundaries."""
        try:
            h, w = gray.shape

            # Ensure starting region is within bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)

            if width <= 0 or height <= 0:
                return ImageRegion(x, y, 1, 1, confidence=0.5, region_type="image")

            roi = gray[y:y + height, x:x + width]

            # Calculate mean intensity of the initial region
            mean_intensity = np.mean(roi)
            threshold = 30  # Intensity difference threshold

            # Expand in each direction (with bounds checking)
            new_x = x
            while new_x > 0:
                col = gray[y:min(y + height, h), max(0, new_x - 1):new_x]
                if col.size > 0 and abs(np.mean(col) - mean_intensity) > threshold:
                    break
                new_x -= 1

            new_x2 = min(x + width, w)
            while new_x2 < w:
                col = gray[y:min(y + height, h), new_x2:min(new_x2 + 1, w)]
                if col.size > 0 and abs(np.mean(col) - mean_intensity) > threshold:
                    break
                new_x2 += 1

            new_y = y
            while new_y > 0:
                row = gray[max(0, new_y - 1):new_y, new_x:new_x2]
                if row.size > 0 and abs(np.mean(row) - mean_intensity) > threshold:
                    break
                new_y -= 1

            new_y2 = min(y + height, h)
            while new_y2 < h:
                row = gray[new_y2:min(new_y2 + 1, h), new_x:new_x2]
                if row.size > 0 and abs(np.mean(row) - mean_intensity) > threshold:
                    break
                new_y2 += 1

            # Final bounds check
            final_width = max(1, new_x2 - new_x)
            final_height = max(1, new_y2 - new_y)

            return ImageRegion(
                x=new_x, y=new_y,
                width=final_width, height=final_height,
                region_type="image", confidence=0.75
            )

        except Exception as e:
            logger.error(f"Region expansion failed: {e}")
            return ImageRegion(x, y, max(1, width), max(1, height),
                               confidence=0.5, region_type="image")

    def _filter_text_regions(self, gray: np.ndarray,
                             regions: List[ImageRegion]) -> List[ImageRegion]:
        """
        Filter out regions that look like text blocks.

        Uses multiple heuristics to identify and remove text-heavy regions
        from the list of detected image regions.

        Args:
            gray: Grayscale image
            regions: List of regions to filter

        Returns:
            Filtered list of regions
        """
        filtered = []

        for region in regions:
            x, y, w, h = region.x, region.y, region.width, region.height

            # Ensure region is within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            roi = gray[y:y + h, x:x + w]
            is_text = False

            # Check 1: Horizontal projection profile (text has regular line spacing)
            try:
                horizontal_proj = np.sum(roi < 128, axis=1)
                if len(horizontal_proj) > 10:
                    peaks, _ = find_peaks(horizontal_proj, distance=10)
                    if len(peaks) > 10:  # Many lines
                        peak_distances = np.diff(peaks)
                        if len(peak_distances) > 0:
                            # Check for regular spacing
                            spacing_std = np.std(peak_distances)
                            if spacing_std < 5:
                                is_text = True
                                logger.debug(f"Region rejected: regular line spacing "
                                             f"(std={spacing_std:.1f})")
            except Exception as e:
                logger.debug(f"Line spacing check failed: {e}")

            # Check 2: Connected components analysis
            if not is_text:
                try:
                    _, binary = cv2.threshold(roi, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        binary, connectivity=8)

                    if num_labels > 50:  # Many components
                        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                        if len(areas) > 0:
                            median_area = np.median(areas)
                            mean_area = np.mean(areas)
                            if median_area < 100 and mean_area > 0:
                                std_ratio = np.std(areas) / mean_area
                                if std_ratio < 1.5:  # Similar sizes = letters
                                    is_text = True
                                    logger.debug(f"Region rejected: uniform small components "
                                                 f"(median_area={median_area:.0f})")
                except Exception as e:
                    logger.debug(f"Component analysis failed: {e}")

            # Check 3: Density and structure patterns
            if not is_text:
                try:
                    fill_ratio = np.sum(roi < 128) / (w * h)
                    if 0.4 < fill_ratio < 0.7 and w / h > 0.8:  # Dense, squarish
                        # Check for geometric patterns (lines)
                        edges = cv2.Canny(roi, 50, 150)
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                                minLineLength=30)
                        if lines is None or len(lines) < 5:
                            is_text = True
                            logger.debug(f"Region rejected: high density without structure "
                                         f"(fill={fill_ratio:.2f})")
                except Exception as e:
                    logger.debug(f"Density check failed: {e}")

            if not is_text:
                filtered.append(region)
                logger.info(f"Region accepted: area={region.area}, "
                            f"type={region.region_type}, confidence={region.confidence:.2f}")

        return filtered

    def _remove_overlapping_regions(self, regions: List[ImageRegion],
                                    iou_threshold: float = 0.3) -> List[ImageRegion]:
        """Remove overlapping regions, keeping the larger ones."""
        if len(regions) <= 1:
            return regions

        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r.area, reverse=True)

        kept_regions = []
        for region in regions:
            should_keep = True
            for kept in kept_regions:
                # Use lower threshold to be more aggressive about removing overlaps
                if region.iou(kept) > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept_regions.append(region)

        logger.debug(f"Reduced {len(regions)} regions to {len(kept_regions)} after overlap removal")
        return kept_regions

    def merge_overlapping_regions(self, regions: List[ImageRegion],
                                  overlap_threshold: float = 0.3) -> List[ImageRegion]:
        """
        Merge overlapping regions to avoid duplicates.

        Args:
            regions: List of regions to merge
            overlap_threshold: Minimum IoU to consider regions for merging

        Returns:
            List of merged regions
        """
        if not regions:
            return []

        merged = []
        remaining = regions.copy()

        while remaining:
            current = remaining.pop(0)
            overlapping = [current]

            i = 0
            while i < len(remaining):
                if current.iou(remaining[i]) > overlap_threshold:
                    overlapping.append(remaining.pop(i))
                else:
                    i += 1

            if len(overlapping) > 1:
                merged_region = self._merge_regions(overlapping)
                merged.append(merged_region)
            else:
                merged.append(current)

        return merged

    def _merge_regions(self, regions: List[ImageRegion]) -> ImageRegion:
        """
        Merge multiple regions into one encompassing region.

        Args:
            regions: List of regions to merge

        Returns:
            Single merged region
        """
        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)

        # Use most common type
        types = [r.region_type for r in regions]
        most_common_type = max(set(types), key=types.count)

        # Use maximum confidence
        max_confidence = max(r.confidence for r in regions)

        return ImageRegion(
            x=min_x, y=min_y,
            width=max_x - min_x, height=max_y - min_y,
            region_type=most_common_type,
            confidence=max_confidence
        )

    def analyze_page_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete page layout analysis.

        Detects all content regions and provides a structured layout analysis.

        Args:
            image: Input image

        Returns:
            Dictionary containing layout analysis results
        """
        text_regions = self.detect_text_regions(image)
        image_regions = self.detect_image_regions(image)

        # Calculate layout statistics
        total_area = image.shape[0] * image.shape[1]
        text_area = sum(r.area for r in text_regions)
        image_area = sum(r.area for r in image_regions)

        return {
            'text_regions': [r.to_dict() for r in text_regions],
            'image_regions': [r.to_dict() for r in image_regions],
            'statistics': {
                'num_text_regions': len(text_regions),
                'num_image_regions': len(image_regions),
                'text_coverage': text_area / total_area,
                'image_coverage': image_area / total_area,
                'page_size': (image.shape[1], image.shape[0])
            }
        }