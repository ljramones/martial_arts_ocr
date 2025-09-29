"""
Image processing utilities for Martial Arts OCR.
Handles image preprocessing, enhancement, and manipulation for better OCR results.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
        """Load image from file path (BGR for OpenCV), honoring EXIF orientation."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Use PIL to apply EXIF orientation, then convert to OpenCV BGR
            try:
                with Image.open(path) as pil:
                    pil = ImageOps.exif_transpose(pil)
                    pil = pil.convert("RGB")
                    arr = np.array(pil)  # RGB
                image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # -> BGR
            except Exception as pil_e:
                logger.warning(f"EXIF-aware load failed ({pil_e}); falling back to cv2.imread")
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
        """Apply preprocessing pipeline optimized for OCR on dense, typewritten pages."""
        try:
            img = image.copy()

            # 1) BGR->RGB once
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2) optional resize up (helps typewriter dots)
            scale = float(self.config.get('resize_factor', 1.2))
            if scale and abs(scale - 1.0) > 1e-3:
                img = self.resize_image(img, scale)

            # 3) grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

            # 4) mild deskew (your robust version)
            if self.config.get('deskew', True):
                gray = self.deskew_image(gray)

            # 5) perspective de-keystone (page contour → homography)
            try:
                g = gray
                # strong edges, close small gaps
                edges = cv2.Canny(g, 60, 160)
                edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    page = max(cnts, key=cv2.contourArea)
                    peri = cv2.arcLength(page, True)
                    approx = cv2.approxPolyDP(page, 0.02 * peri, True)
                    if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * (g.shape[0] * g.shape[1]):
                        # order corners
                        pts = approx.reshape(4, 2).astype(np.float32)
                        s = pts.sum(axis=1);
                        d = np.diff(pts, axis=1).ravel()
                        rect = np.zeros((4, 2), dtype=np.float32)
                        rect[0] = pts[np.argmin(s)]  # top-left
                        rect[2] = pts[np.max(s.argmax(), initial=0)]  # bottom-right (safe)
                        rect[1] = pts[np.argmin(d)]  # top-right
                        rect[3] = pts[np.argmax(d)]  # bottom-left
                        # target size: keep aspect; push to ~2400px tall max
                        widthA = np.linalg.norm(rect[2] - rect[3])
                        widthB = np.linalg.norm(rect[1] - rect[0])
                        heightA = np.linalg.norm(rect[1] - rect[2])
                        heightB = np.linalg.norm(rect[0] - rect[3])
                        W = int(max(widthA, widthB))
                        H = int(max(heightA, heightB))
                        scaleH = min(2400 / max(H, 1), 1.0)
                        W, H = int(W * scaleH), int(H * scaleH)
                        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(rect, dst)
                        gray = cv2.warpPerspective(g, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            except Exception as _:
                pass  # fall back gracefully

            # 6) denoise (light)
            if self.config.get('denoise', True):
                gray = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)

            # 7) contrast + local binarization for tiny mono glyphs
            # CLAHE (small tiles) → Sauvola (typewritten-friendly)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g2 = clahe.apply(gray)
            try:
                # Sauvola
                mean = cv2.boxFilter(g2, ddepth=-1, ksize=(25, 25))
                sqmean = cv2.boxFilter((g2 * g2).astype(np.float32), ddepth=-1, ksize=(25, 25))
                var = np.maximum(sqmean - mean.astype(np.float32) ** 2, 0)
                std = cv2.sqrt(var)
                R = 128.0
                k = 0.2
                thresh = (mean.astype(np.float32) * (1 + k * ((std / R) - 1))).astype(np.uint8)
                bin_img = (g2 > thresh).astype(np.uint8) * 255
            except Exception:
                # Gaussian adaptive fallback
                bin_img = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 31, 10)

            # 8) light morphological connect (helps “i/j” dots & broken strokes)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 9) unsharp to crispen edges for OCR
            blurred = cv2.GaussianBlur(bin_img, (0, 0), 1.0)
            sharp = cv2.addWeighted(bin_img, 1.5, blurred, -0.5, 0)

            return self.normalize_image(sharp)

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image

    def correct_illumination(self, gray: np.ndarray) -> np.ndarray:
        """Flatten background with division normalization (robust for camera shots)."""
        try:
            # big blur → estimate background then divide
            r = max(15, int(0.02 * max(gray.shape)))  # ~2% of max dim
            if r % 2 == 0: r += 1
            bg = cv2.GaussianBlur(gray, (r, r), 0)
            # avoid divide-by-zero
            bg = np.clip(bg, 1, 255).astype(np.float32)
            norm = (gray.astype(np.float32) / bg) * 128.0
            norm = np.clip(norm, 0, 255).astype(np.uint8)
            return norm
        except Exception as e:
            logger.debug(f"Illumination correction skipped: {e}")
            return gray

    def resize_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Resize image by the given factor."""
        if factor == 1.0:
            return image

        height, width = image.shape[:2]
        new_width = max(1, int(width * factor))
        new_height = max(1, int(height * factor))

        interpolation = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        logger.debug(f"Resized image: {width}x{height} -> {new_width}x{new_height}")
        return resized

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct skew/rotation in the image (robust to false near-vertical angles)."""
        try:
            # Edge + line detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100, minLineLength=100, maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                return image

            # Collect line angles (degrees)
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
                # Normalize to [-90, 90] to treat verticals consistently
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                angles.append(angle)

            if not angles:
                return image

            median_angle = float(np.median(angles))

            # Guardrails
            if abs(median_angle) > 85:
                logger.debug(f"Deskew: ignoring near-vertical angle {median_angle:.2f}°")
                return image
            if abs(median_angle) > 30:
                logger.debug(f"Deskew: skipping large angle {median_angle:.2f}° as likely false")
                return image
            if abs(median_angle) <= 0.5:
                return image

            # Rotate with canvas expansion to avoid cropping
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Translate to keep image centered
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            deskewed = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            logger.debug(f"Deskewed image by {median_angle:.2f} degrees (lines={len(lines)})")
            return deskewed

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from the image."""
        try:
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
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            logger.debug("Enhanced contrast with CLAHE")
            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to 0-255 range (uint8)."""
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        return image

    def create_binary_image(self, image: np.ndarray) -> np.ndarray:
        """Create binary (black and white) version for OCR."""
        try:
            if len(image.shape) == 3:
                # Be defensive: try RGB first, then BGR if needed
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except cv2.error:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

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
            # Convert to grayscale robustly
            if len(image.shape) == 3:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except cv2.error:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Ensure uint8
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # MSER
            min_area = int(self.config.get('text_block_min_area', 1000))
            max_area = int(gray.shape[0] * gray.shape[1] * 0.3)
            # delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size
            mser = cv2.MSER_create(5, min_area, max_area, 0.25, 0.2, 200, 1.01, 0.003, 5)

            regions, _ = mser.detectRegions(gray)
            text_regions: List[ImageRegion] = []

            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                area = w * h
                aspect_ratio = (w / h) if h > 0 else 0.0

                if (area >= min_area) and (0.1 <= aspect_ratio <= 20.0):
                    text_regions.append(ImageRegion(
                        x=x, y=y, width=w, height=h,
                        region_type="text",
                        confidence=0.8
                    ))

            logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions

        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def detect_image_regions(self, image: np.ndarray) -> List[ImageRegion]:
        """Detect regions containing images/diagrams."""
        try:
            if len(image.shape) == 3:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except cv2.error:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_regions: List[ImageRegion] = []
            min_area = self.config.get('image_block_min_area', 2500)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = (w / h) if h > 0 else 0.0
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

            i = 0
            while i < len(remaining):
                if self._calculate_overlap(current, remaining[i]) > overlap_threshold:
                    overlapping.append(remaining.pop(i))
                else:
                    i += 1

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

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection_area = x_overlap * y_overlap

        union_area = region1.area + region2.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def _merge_regions(self, regions: List[ImageRegion]) -> ImageRegion:
        """Merge multiple regions into one."""
        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)

        types = [r.region_type for r in regions]
        most_common_type = max(set(types), key=types.count)
        max_confidence = max(r.confidence for r in regions)

        return ImageRegion(
            x=min_x, y=min_y,
            width=max_x - min_x, height=max_y - min_y,
            region_type=most_common_type,
            confidence=max_confidence
        )


# ---------- Line-merging utilities (group small boxes into lines) ----------

@dataclass
class _Box:
    x: int; y: int; w: int; h: int
    @property
    def x2(self): return self.x + self.w
    @property
    def y2(self): return self.y + self.h

def _y_overlap_ratio(a: _Box, b: _Box) -> float:
    top = max(a.y, b.y)
    bot = min(a.y2, b.y2)
    inter = max(0, bot - top)
    return inter / max(1, min(a.h, b.h))

def _merge_horizontal(a: _Box, b: _Box) -> _Box:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2, y2 = max(a.x2, b.x2), max(a.y2, b.y2)
    return _Box(x1, y1, x2 - x1, y2 - y1)

def merge_regions_into_lines(regions: List[ImageRegion],
                             max_gap_px: int = 28,
                             min_y_overlap: float = 0.45) -> List[ImageRegion]:
    """
    Greedy left-to-right line merge:
    - Two regions are merged if their vertical overlap ratio >= min_y_overlap
      and the horizontal gap between them <= max_gap_px.
    - Produces wider, line-level boxes that OCR much better than tiny blobs.
    """
    if not regions:
        return []

    # sort by y, then x (top-to-bottom, then left-to-right)
    regs = sorted(regions, key=lambda r: (r.y, r.x))
    lines: List[List[ImageRegion]] = []

    for r in regs:
        placed = False
        for line in lines:
            last = line[-1]
            # vertical overlap ratio based on min height
            y1a, y2a = last.y, last.y + last.height
            y1b, y2b = r.y, r.y + r.height
            overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
            min_h = max(1, min(last.height, r.height))
            y_overlap_ratio = overlap / min_h

            # horizontal gap (r is to the right because of sort, but be safe)
            gap = r.x - (last.x + last.width)
            if y_overlap_ratio >= min_y_overlap and gap <= max_gap_px:
                line.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])

    # fuse each line into one region
    merged: List[ImageRegion] = []
    for line in lines:
        x1 = min(rr.x for rr in line)
        y1 = min(rr.y for rr in line)
        x2 = max(rr.x + rr.width for rr in line)
        y2 = max(rr.y + rr.height for rr in line)
        merged.append(ImageRegion(
            x=x1, y=y1, width=x2 - x1, height=y2 - y1,
            confidence=max(rr.confidence for rr in line),
            region_type="text"
        ))

    return merged

def _post_ocr_fixups(self, text: str) -> str:
    """
    Light, deterministic cleanups:
    - join hyphenated line breaks: 'thrust-\ning' -> 'thrusting'
    - merge soft wraps when the next line continues the same sentence
    - drop consecutive duplicate lines (from overlapping regions)
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

    # 2) merge soft wraps (if next line looks like a continuation)
    merged = []
    for ln in out:
        if merged and ln and ln[0].islower() and not merged[-1].endswith(('.', '!', '?', ':', ';')):
            merged[-1] = (merged[-1] + ' ' + ln).strip()
        else:
            merged.append(ln)

    # 3) drop immediate duplicates
    dedup = []
    for ln in merged:
        if not dedup or ln != dedup[-1]:
            dedup.append(ln)

    # 4) collapse excess blank lines
    final_lines = []
    for ln in dedup:
        if ln.strip() == "":
            if final_lines and final_lines[-1].strip() == "":
                continue
        final_lines.append(ln)

    return "\n".join(final_lines).strip()


# ----------------------------- Utility functions -----------------------------

def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """Save image to file with specified quality."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # If it's RGB, convert to BGR for cv2.imwrite; if grayscale, keep as is
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        if path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))])
        elif path.suffix.lower() == '.png':
            # PNG compression is 0..9 (higher = smaller)
            compression = int(np.clip((100 - quality) / 10, 0, 9))
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

        scale = min(target_width / width, target_height / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

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

        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        if path.suffix.lower() not in valid_extensions:
            return False

        with Image.open(path) as img:
            img.verify()

        return True

    except Exception:
        return False
