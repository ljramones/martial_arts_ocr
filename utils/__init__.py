"""Image processing utilities for Martial Arts OCR."""

# Import main classes
from .image_preprocessing import ImageProcessor
from .image_layout import LayoutAnalyzer

# Import core data classes
from .core_image import ImageRegion, ImageInfo

# Import commonly used functions
from .image_io import (
    load_image,
    save_image,
    validate_image_file,
    get_image_info,
    create_thumbnail,
    extract_image_region
)

from .image_preprocessing import (
    preprocess_for_captions_np,
    preprocess_for_fullpage_np
)

from .image_regions import (
    merge_regions_into_lines,
    post_ocr_fixups
)

__all__ = [
    # Classes
    'ImageProcessor',
    'LayoutAnalyzer',
    'ImageRegion',
    'ImageInfo',
    # I/O functions
    'load_image',
    'save_image',
    'validate_image_file',
    'get_image_info',
    'create_thumbnail',
    'extract_image_region',
    # Preprocessing functions
    'preprocess_for_captions_np',
    'preprocess_for_fullpage_np',
    # Region utilities
    'merge_regions_into_lines',
    'post_ocr_fixups'
]
