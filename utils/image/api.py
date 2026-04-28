# utils/image/api.py
"""
High-level convenience API for image IO and common operations.

This module wraps lower-level utils so callers don't need to import from
many places. It also provides a couple of safe one-liners used across
your OCR codebase.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

# IO
from utils.image.io.read import load_image, imread_bytes
from utils.image.io.write import save_image, imencode
from utils.image.io.validate import (
    validate_image_file, validate_image_bytes, ValidationReport
)
from utils.image.io.meta import get_image_meta, probe_array, export_image_info, ImageMeta

# OPS
from utils.image.ops.resize import (
    resize_to_width, resize_to_height, resize_long_edge, resize_scale,
    fit_into, fill_into, thumbnail as resize_thumbnail, letterbox
)
from utils.image.ops.thumb import (
    make_thumb, thumb_from_file, thumb_from_bytes,
    thumb_file_to_file, thumb_bytes_to_bytes
)
from utils.image.ops.extract import (
    extract_region, extract_many, extract_polygon_region, clamp_bbox,
    expand_bbox, translate_region, scale_region, region_to_slice
)
from utils.image.ops.tone import (
    to_gray, adjust_brightness_contrast, gamma_correct, auto_contrast_gray,
    rescale_intensity_gray, clahe_gray, clahe_bgr_lab, white_balance_grayworld_bgr,
    unsharp_mask_gray
)

ColorMode = Literal["bgr", "rgb", "gray"]
PathLike = Union[str, Path]

__all__ = [
    # IO
    "load_image", "imread_bytes", "save_image", "imencode",
    "validate_image_file", "validate_image_bytes", "ValidationReport",
    "get_image_meta", "probe_array", "export_image_info", "ImageMeta",
    # Resize & thumbs
    "resize_to_width", "resize_to_height", "resize_long_edge", "resize_scale",
    "fit_into", "fill_into", "resize_thumbnail", "letterbox",
    "make_thumb", "thumb_from_file", "thumb_from_bytes",
    "thumb_file_to_file", "thumb_bytes_to_bytes",
    # Extract
    "extract_region", "extract_many", "extract_polygon_region",
    "clamp_bbox", "expand_bbox", "translate_region", "scale_region", "region_to_slice",
    # Tone
    "to_gray", "adjust_brightness_contrast", "gamma_correct", "auto_contrast_gray",
    "rescale_intensity_gray", "clahe_gray", "clahe_bgr_lab",
    "white_balance_grayworld_bgr", "unsharp_mask_gray",
    # Helpers
    "ensure_valid_then_load", "load_and_resize_long_edge", "quick_thumb_to_file",
]

def ensure_valid_then_load(path: PathLike, *, mode: ColorMode = "bgr") -> np.ndarray:
    """Validate first; raise with a helpful message if invalid; then EXIF-aware load."""
    rpt = validate_image_file(path)
    if not rpt.is_valid:
        raise ValueError(f"Invalid image '{path}': {rpt.reason}")
    return load_image(path, mode=mode, use_exif=True)

def load_and_resize_long_edge(path: PathLike, long_edge: int, *, mode: ColorMode = "bgr") -> np.ndarray:
    """One-liner: validate → load → resize longest edge."""
    img = ensure_valid_then_load(path, mode=mode)
    return resize_long_edge(img, long_edge)

def quick_thumb_to_file(src: PathLike, dst: PathLike, size: Tuple[int, int] = (320, 320)) -> bool:
    """One-liner: load → make thumbnail → save (JPEG 90)."""
    return thumb_file_to_file(src, dst, max_size=size, pad_to_box=False, output_quality=90)
