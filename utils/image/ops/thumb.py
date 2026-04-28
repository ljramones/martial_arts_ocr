# utils/image/ops/thumb.py
"""
Thumbnail utilities for the Martial Arts OCR system.

Responsibilities:
- Create thumbnails from NumPy arrays
- Create thumbnails from files or bytes (with EXIF-aware loading via io.read)
- Optional letterbox padding to hit an exact rectangle
- Save to disk or encode to memory buffers

Conventions:
- Thumbnails preserve aspect by default.
- For exact WxH outputs, set pad_to_box=True (letterboxes with bg color).

Dependencies:
- utils.image.io.read.load_image, .imread_bytes
- utils.image.io.write.save_image, .imencode
- utils.image.ops.resize.thumbnail, .fit_into
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import cv2

# Local imports inside the package
from utils.image.io.read import load_image, imread_bytes
from utils.image.io.write import save_image, imencode
from utils.image.ops.resize import thumbnail as _resize_thumbnail, fit_into as _fit_into

logger = logging.getLogger(__name__)

ColorMode = Literal["bgr", "rgb", "gray"]
PathLike = str | Path

__all__ = [
    "make_thumb",
    "thumb_from_file",
    "thumb_from_bytes",
    "thumb_file_to_file",
    "thumb_bytes_to_bytes",
]


# ----------------------------- core array API ---------------------------------

def make_thumb(
    img: np.ndarray,
    max_size: Tuple[int, int] = (200, 300),
    *,
    pad_to_box: bool = False,
    bg: Tuple[int, int, int] = (0, 0, 0),
    prefer_area_for_downscale: bool = True,
) -> np.ndarray:
    """
    Make a thumbnail from an in-memory image.

    Args:
        img: NumPy ndarray (GRAY 2D or HxWxC).
        max_size: (max_width, max_height) bounding box.
        pad_to_box: If True, letterbox-pad to exactly max_size (contain).
        bg: Background color for padding (BGR).
        prefer_area_for_downscale: Use INTER_AREA when downscaling (crisper text).

    Returns:
        Thumbnail ndarray.
    """
    thumb = _resize_thumbnail(img, max_size, prefer_area_for_downscale=prefer_area_for_downscale)
    if pad_to_box:
        mw, mh = max_size
        thumb = _fit_into(thumb, (mw, mh), bg=bg, align="center")
    return thumb


# --------------------------- file/bytes front-ends ----------------------------

def thumb_from_file(
    path: PathLike,
    max_size: Tuple[int, int] = (200, 300),
    *,
    mode: ColorMode = "bgr",
    use_exif: bool = True,
    pad_to_box: bool = False,
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Load an image from disk and return a thumbnail ndarray.
    """
    arr = load_image(path, mode=mode, use_exif=use_exif)
    return make_thumb(arr, max_size, pad_to_box=pad_to_box, bg=bg)


def thumb_from_bytes(
    data: bytes,
    max_size: Tuple[int, int] = (200, 300),
    *,
    mode: ColorMode = "bgr",
    use_exif: bool = True,
    pad_to_box: bool = False,
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Decode an image from bytes and return a thumbnail ndarray.
    """
    arr = imread_bytes(data, mode=mode, use_exif=use_exif)
    return make_thumb(arr, max_size, pad_to_box=pad_to_box, bg=bg)


# ----------------------------- save/encode helpers ----------------------------

def thumb_file_to_file(
    in_path: PathLike,
    out_path: PathLike,
    max_size: Tuple[int, int] = (200, 300),
    *,
    pad_to_box: bool = False,
    bg: Tuple[int, int, int] = (0, 0, 0),
    input_mode: ColorMode = "bgr",
    output_quality: int = 90,
    png_compression: Optional[int] = None,
    progressive_jpeg: bool = True,
    tiff_lzw: bool = True,
) -> bool:
    """
    Make a thumbnail from a file and save to another file.

    Returns:
        True on success, False otherwise.
    """
    try:
        arr = load_image(in_path, mode=input_mode, use_exif=True)
        thumb = make_thumb(arr, max_size, pad_to_box=pad_to_box, bg=bg)
        ok = save_image(
            thumb,
            out_path,
            quality=output_quality,
            input_mode=input_mode,
            png_compression=png_compression,
            progressive_jpeg=progressive_jpeg,
            tiff_lzw=tiff_lzw,
            atomic=True,
        )
        if not ok:
            logger.error("Failed to save thumbnail: %s -> %s", in_path, out_path)
        return ok
    except Exception as e:
        logger.error("thumb_file_to_file error for %s -> %s: %s", in_path, out_path, e)
        return False


def thumb_bytes_to_bytes(
    data: bytes,
    *,
    max_size: Tuple[int, int] = (200, 300),
    pad_to_box: bool = False,
    bg: Tuple[int, int, int] = (0, 0, 0),
    input_mode: ColorMode = "bgr",
    out_ext: str = ".jpg",
    output_quality: int = 90,
    png_compression: Optional[int] = None,
    progressive_jpeg: bool = True,
    tiff_lzw: bool = True,
) -> tuple[bool, np.ndarray]:
    """
    Make a thumbnail from bytes and return an encoded buffer.

    Args:
        data: Encoded input image.
        out_ext: Output extension (e.g., '.jpg', '.png', '.webp').

    Returns:
        (ok, buf) where buf is a 1D uint8 array with encoded bytes.
    """
    try:
        arr = imread_bytes(data, mode=input_mode, use_exif=True)
        thumb = make_thumb(arr, max_size, pad_to_box=pad_to_box, bg=bg)
        ok, buf = imencode(
            out_ext,
            thumb,
            input_mode=input_mode,
            quality=output_quality,
            png_compression=png_compression,
            progressive_jpeg=progressive_jpeg,
            tiff_lzw=tiff_lzw,
        )
        if not ok:
            logger.error("thumb_bytes_to_bytes: imencode failed for ext=%s", out_ext)
        return ok, buf
    except Exception as e:
        logger.error("thumb_bytes_to_bytes error: %s", e)
        return False, np.empty((0,), dtype=np.uint8)
