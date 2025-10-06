"""
Image input/output operations for the Martial Arts OCR system.

This module handles all image file operations including loading, saving,
validation, and basic manipulation like thumbnail creation and region extraction.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import logging
from typing import Optional, Tuple, Union

from .core_image import ImageInfo, ImageRegion

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path, returning BGR array for OpenCV.

    Automatically handles EXIF orientation metadata to ensure images
    are correctly oriented regardless of camera/phone orientation.

    Args:
        image_path: Path to the image file

    Returns:
        numpy array in BGR format (OpenCV convention)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded
    """
    try:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Use PIL to apply EXIF orientation, then convert to OpenCV BGR
        try:
            with Image.open(path) as pil_img:
                # Handle EXIF orientation
                pil_img = ImageOps.exif_transpose(pil_img)

                # Convert to RGB if necessary
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to numpy array (RGB format)
                arr = np.array(pil_img)

                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        except Exception as pil_error:
            logger.warning(f"EXIF-aware load failed ({pil_error}); falling back to cv2.imread")
            image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        logger.debug(f"Loaded image: {path.name} (shape: {image.shape}, dtype: {image.dtype})")
        return image

    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def save_image(image: np.ndarray, output_path: Union[str, Path],
               quality: int = 95, create_dirs: bool = True) -> bool:
    """
    Save image to file with specified quality.

    Supports JPEG, PNG, and other common formats. Automatically handles
    color space conversion from RGB to BGR for OpenCV compatibility.

    Args:
        image: Image array to save
        output_path: Path where the image will be saved
        quality: JPEG quality (1-100) or PNG compression (0-9, auto-converted)
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        True if save was successful, False otherwise
    """
    try:
        path = Path(output_path)

        # Create parent directories if requested
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Handle color space conversion if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB input, convert to BGR for cv2.imwrite
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Set compression parameters based on file format
        suffix = path.suffix.lower()

        if suffix in ['.jpg', '.jpeg']:
            # JPEG quality parameter
            quality_param = int(np.clip(quality, 1, 100))
            success = cv2.imwrite(str(path), image_bgr,
                                  [cv2.IMWRITE_JPEG_QUALITY, quality_param])

        elif suffix == '.png':
            # PNG compression level (0-9, where 9 is max compression)
            compression = int(np.clip((100 - quality) / 10, 0, 9))
            success = cv2.imwrite(str(path), image_bgr,
                                  [cv2.IMWRITE_PNG_COMPRESSION, compression])

        elif suffix in ['.tiff', '.tif']:
            # TIFF with LZW compression
            success = cv2.imwrite(str(path), image_bgr,
                                  [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        else:
            # Default save for other formats
            success = cv2.imwrite(str(path), image_bgr)

        if success:
            file_size = path.stat().st_size
            logger.debug(f"Saved image to: {path} (size: {file_size:,} bytes)")
        else:
            logger.error(f"cv2.imwrite failed for: {path}")

        return bool(success)

    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if file is a supported image format.

    Checks both file extension and actual file content to ensure
    the file is a valid, readable image.

    Args:
        file_path: Path to the file to validate

    Returns:
        True if the file is a valid image, False otherwise
    """
    try:
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            logger.debug(f"File does not exist: {file_path}")
            return False

        # Check if it's a file (not directory)
        if not path.is_file():
            logger.debug(f"Path is not a file: {file_path}")
            return False

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif',
                            '.bmp', '.webp', '.jp2', '.j2k'}
        if path.suffix.lower() not in valid_extensions:
            logger.debug(f"Invalid extension: {path.suffix}")
            return False

        # Try to open and verify the image
        try:
            with Image.open(path) as img:
                img.verify()  # Verify it's a valid image

            # verify() closes the file, so we need to reopen for additional checks
            with Image.open(path) as img:
                # Check for minimum size
                if img.width < 10 or img.height < 10:
                    logger.debug(f"Image too small: {img.width}x{img.height}")
                    return False

        except Exception as e:
            logger.debug(f"PIL verification failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"Validation error for {file_path}: {e}")
        return False


def get_image_info(image_path: Union[str, Path]) -> ImageInfo:
    """
    Get detailed information about an image file.

    Extracts comprehensive metadata including dimensions, format,
    color space, and file size.

    Args:
        image_path: Path to the image file

    Returns:
        ImageInfo object containing image metadata

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be read
    """
    try:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Get format and basic info from PIL
        with Image.open(path) as pil_img:
            format_info = pil_img.format or "unknown"
            mode = pil_img.mode

            # Extract DPI if available
            dpi = None
            if hasattr(pil_img, 'info') and 'dpi' in pil_img.info:
                dpi_info = pil_img.info['dpi']
                if isinstance(dpi_info, tuple) and len(dpi_info) >= 2:
                    dpi = int((dpi_info[0] + dpi_info[1]) / 2)

            # Map PIL mode to color space
            color_space_map = {
                'L': 'grayscale',
                'RGB': 'RGB',
                'RGBA': 'RGBA',
                'CMYK': 'CMYK',
                'YCbCr': 'YCbCr',
                'LAB': 'LAB',
                'HSV': 'HSV',
                '1': 'binary'
            }
            color_space = color_space_map.get(mode, mode)

        # Load with OpenCV for detailed analysis
        cv_img = load_image(image_path)

        # Determine number of channels and dtype
        if len(cv_img.shape) == 2:
            channels = 1
        else:
            channels = cv_img.shape[2]

        return ImageInfo(
            width=cv_img.shape[1],
            height=cv_img.shape[0],
            channels=channels,
            dtype=str(cv_img.dtype),
            file_size=path.stat().st_size,
            format=format_info,
            dpi=dpi,
            color_space=color_space
        )

    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
        raise


def create_thumbnail(image: np.ndarray,
                     size: Tuple[int, int] = (200, 300),
                     maintain_aspect: bool = True,
                     resample: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Create a thumbnail of the image.

    Args:
        image: Source image array
        size: Target size as (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        resample: OpenCV interpolation method

    Returns:
        Thumbnail image array
    """
    try:
        height, width = image.shape[:2]
        target_width, target_height = size

        if maintain_aspect:
            # Calculate scale to fit within target size while maintaining aspect ratio
            scale = min(target_width / width, target_height / height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
        else:
            # Direct resize to target dimensions
            new_width = target_width
            new_height = target_height

        # Use INTER_AREA for downscaling, INTER_CUBIC for upscaling
        if new_width < width and new_height < height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = resample

        thumbnail = cv2.resize(image, (new_width, new_height),
                               interpolation=interpolation)

        logger.debug(f"Created thumbnail: {width}x{height} -> {new_width}x{new_height}")
        return thumbnail

    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return image


def extract_image_region(image: np.ndarray, region: ImageRegion,
                         padding: int = 0,
                         safe_crop: bool = True) -> np.ndarray:
    """
    Extract a specific region from an image.

    Args:
        image: Source image array
        region: Region to extract
        padding: Additional pixels to include around the region
        safe_crop: If True, clip coordinates to image bounds

    Returns:
        Extracted region as image array

    Raises:
        ValueError: If region is outside image bounds and safe_crop is False
    """
    try:
        x1, y1, x2, y2 = region.bbox

        # Add padding if specified
        if padding > 0:
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding

        height, width = image.shape[:2]

        if safe_crop:
            # Clip coordinates to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(x1, min(x2, width))
            y2 = max(y1, min(y2, height))
        else:
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError(f"Region {region.bbox} exceeds image bounds {width}x{height}")

        extracted = image[y1:y2, x1:x2]

        # Ensure we return a valid image
        if extracted.size == 0:
            logger.warning(f"Extracted region is empty, returning 1x1 image")
            extracted = np.zeros((1, 1, image.shape[2] if len(image.shape) > 2 else 1),
                                 dtype=image.dtype)

        logger.debug(f"Extracted region: {region.region_type} at ({x1},{y1}) "
                     f"size {x2 - x1}x{y2 - y1}")
        return extracted

    except Exception as e:
        logger.error(f"Failed to extract region: {e}")
        raise


def resize_to_height(image: np.ndarray, target_height: int,
                     interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to a specific height while maintaining aspect ratio.

    Args:
        image: Source image array
        target_height: Desired height in pixels
        interpolation: OpenCV interpolation method

    Returns:
        Resized image array
    """
    height, width = image.shape[:2]
    scale = target_height / height
    new_width = max(1, int(width * scale))

    return cv2.resize(image, (new_width, target_height), interpolation=interpolation)


def resize_to_width(image: np.ndarray, target_width: int,
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to a specific width while maintaining aspect ratio.

    Args:
        image: Source image array
        target_width: Desired width in pixels
        interpolation: OpenCV interpolation method

    Returns:
        Resized image array
    """
    height, width = image.shape[:2]
    scale = target_width / width
    new_height = max(1, int(height * scale))

    return cv2.resize(image, (target_width, new_height), interpolation=interpolation)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Source image array (BGR or RGB)

    Returns:
        Grayscale image array
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        # Assume BGR format (OpenCV convention)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")