# utils/image/io/meta.py
"""
Image metadata utilities for the Martial Arts OCR system.

Responsibilities:
- Inspect image files for metadata (format, size, mode, DPI, EXIF, ICC)
- Inspect NumPy arrays (shape, dtype, channels)
- Optional export to core ImageInfo (width, height, channels, dtype, file_size, format, dpi, color_space)

Notes:
- Uses Pillow for lightweight header/EXIF/DPI/ICC reads.
- Uses OpenCV only when you request dtype/channels via full decode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
from PIL import Image

# Optional, used only for nicer ICC descriptions if available
try:
    from PIL import ImageCms  # type: ignore
except Exception:
    ImageCms = None  # type: ignore

# Pull in loader if you want dtype/channels without re-implementing
try:
    from .read import load_image  # returns np.ndarray in BGR/RGB/GRAY
except Exception:
    load_image = None  # type: ignore

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ColorMode = Literal["bgr", "rgb", "gray"]


@dataclass(frozen=True)
class ImageMeta:
    # Geometry
    width: int
    height: int
    channels: Optional[int] = None  # for arrays or after decode
    aspect_ratio: Optional[float] = None  # width / height
    megapixels: Optional[float] = None

    # File/system
    path: Optional[str] = None
    file_size: Optional[int] = None  # bytes

    # Pixel info
    dtype: Optional[str] = None  # e.g., 'uint8'
    pil_mode: Optional[str] = None  # e.g., 'RGB', 'L', 'CMYK'
    color_space: Optional[str] = None  # friendly label (RGB, grayscale, etc.)
    has_alpha: Optional[bool] = None

    # Format & capture hints
    fmt: Optional[str] = None  # e.g., 'JPEG', 'PNG'
    dpi: Optional[int] = None
    exif_orientation: Optional[int] = None  # EXIF tag 274, 1..8
    exif_datetime: Optional[str] = None     # DateTimeOriginal if present
    icc_profile_desc: Optional[str] = None  # ICC description if parseable

    # Raw EXIF dict (optional, small subset)
    exif: Optional[Dict[str, Any]] = None


__all__ = [
    "ImageMeta",
    "get_image_meta",
    "probe_array",
    "export_image_info",
]


# --- Helpers -----------------------------------------------------------------

_COLOR_SPACE_MAP = {
    "1": "binary",
    "L": "grayscale",
    "LA": "grayscale+alpha",
    "P": "palette",
    "RGB": "RGB",
    "RGBA": "RGBA",
    "CMYK": "CMYK",
    "YCbCr": "YCbCr",
    "LAB": "LAB",
    "HSV": "HSV",
}

_EXIF_ORIENTATION_TAG = 274
_EXIF_DT_TAGS = (36867, 306)  # DateTimeOriginal, DateTime


def _extract_dpi(img: Image.Image) -> Optional[int]:
    info = getattr(img, "info", {}) or {}
    dpi_val = info.get("dpi")
    if isinstance(dpi_val, tuple) and len(dpi_val) >= 2:
        try:
            return int(round((dpi_val[0] + dpi_val[1]) / 2))
        except Exception:
            return None
    if isinstance(dpi_val, (int, float)):
        try:
            return int(round(dpi_val))
        except Exception:
            return None
    return None


def _extract_icc_description(img: Image.Image) -> Optional[str]:
    try:
        icc_bytes = img.info.get("icc_profile")
        if not icc_bytes:
            return None
        if ImageCms is None:
            # No ImageCms available; return a generic marker
            return "ICC profile (unknown description)"
        prof = ImageCms.ImageCmsProfile(bytes(icc_bytes))
        return ImageCms.getProfileName(prof) or "ICC profile"
    except Exception:
        return None


def _extract_exif_small(img: Image.Image) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        exif = img.getexif()
        if not exif:
            return out
        # Orientation
        ori = exif.get(_EXIF_ORIENTATION_TAG)
        if isinstance(ori, int):
            out["Orientation"] = ori
        # Datetime hints (prefer DateTimeOriginal)
        for tag in _EXIF_DT_TAGS:
            val = exif.get(tag)
            if val:
                out["DateTime"] = str(val)
                break
    except Exception:
        pass
    return out


def _shape_to_channels(arr: np.ndarray) -> int:
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return arr.shape[2]
    return 0


def _basic_dims(width: int, height: int) -> Tuple[float, float]:
    ar = (width / height) if height else 0.0
    mp = round((width * height) / 1_000_000.0, 3)
    return ar, mp


# --- Public API ---------------------------------------------------------------

def get_image_meta(
    file_path: PathLike,
    *,
    decode_for_dtype: bool = False,
    decode_mode: ColorMode = "bgr",
) -> ImageMeta:
    """
    Inspect an image file on disk.

    Args:
        file_path: path to the image.
        decode_for_dtype: if True, fully decode to NumPy to populate dtype/channels.
                          (Costs more I/O/CPU; uses .read.load_image if available.)
        decode_mode: when decoding, choose {"bgr","rgb","gray"} for dtype/channels.

    Returns:
        ImageMeta with best-effort fields populated.

    Raises:
        FileNotFoundError if path missing.
        ValueError if file cannot be opened at all.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    file_size = None
    try:
        file_size = path.stat().st_size
    except Exception:
        pass

    # Open with Pillow to avoid decoding full pixels unless requested
    try:
        with Image.open(path) as pil:
            width, height = pil.size
            fmt = pil.format or "unknown"
            pil_mode = pil.mode
            dpi = _extract_dpi(pil)
            icc_desc = _extract_icc_description(pil)
            exif_small = _extract_exif_small(pil)
            exif_orientation = exif_small.get("Orientation")
            exif_datetime = exif_small.get("DateTime")
            color_space = _COLOR_SPACE_MAP.get(pil_mode, pil_mode)
            has_alpha = "A" in (pil_mode or "")

            channels = None
            dtype = None
            if decode_for_dtype:
                if load_image is None:
                    raise RuntimeError("decode_for_dtype=True but io.read.load_image is unavailable.")
                arr = load_image(path, mode=decode_mode, use_exif=False)
                channels = _shape_to_channels(arr)
                dtype = str(arr.dtype)

            ar, mp = _basic_dims(width, height)

            return ImageMeta(
                width=width,
                height=height,
                channels=channels,
                aspect_ratio=round(ar, 6),
                megapixels=mp,
                path=str(path),
                file_size=file_size,
                dtype=dtype,
                pil_mode=pil_mode,
                color_space=color_space,
                has_alpha=has_alpha,
                fmt=fmt,
                dpi=dpi,
                exif_orientation=exif_orientation,
                exif_datetime=exif_datetime,
                icc_profile_desc=icc_desc,
                exif={"Orientation": exif_orientation, "DateTime": exif_datetime} if (exif_orientation or exif_datetime) else None,
            )
    except Exception as e:
        logger.error("Failed to read metadata for %s: %s", file_path, e)
        raise ValueError(f"Cannot read image metadata: {file_path}") from e


def probe_array(arr: np.ndarray) -> ImageMeta:
    """
    Inspect an in-memory NumPy image (no file-related fields).

    Returns:
        ImageMeta with geometry/channels/dtype and derived fields filled.
    """
    if arr is None or not isinstance(arr, np.ndarray):
        raise ValueError("probe_array expects a NumPy ndarray.")

    h, w = arr.shape[:2]
    ch = _shape_to_channels(arr)
    ar, mp = _basic_dims(w, h)

    # Heuristic pil_mode/color_space from shape
    if ch == 1:
        pil_mode = "L"
        color_space = "grayscale"
        has_alpha = False
    elif ch == 3:
        pil_mode = "RGB"
        color_space = "RGB"
        has_alpha = False
    elif ch == 4:
        pil_mode = "RGBA"
        color_space = "RGBA"
        has_alpha = True
    else:
        pil_mode = None
        color_space = None
        has_alpha = None

    return ImageMeta(
        width=w,
        height=h,
        channels=ch,
        aspect_ratio=round(ar, 6),
        megapixels=mp,
        dtype=str(arr.dtype),
        pil_mode=pil_mode,
        color_space=color_space,
        has_alpha=has_alpha,
    )


# Optional: export to your existing ImageInfo type -----------------------------

def export_image_info(meta: ImageMeta):
    """
    Convert ImageMeta -> utils.image.regions.core_image.ImageInfo (if importable).
    Returns an ImageInfo instance or raises ImportError if type unavailable.
    """
    # Local import to avoid hard dependency at import time
    from utils.image.regions.core_image import ImageInfo  # type: ignore

    # Fallback assumptions if channels/dtype were not decoded
    channels = meta.channels if meta.channels is not None else (
        1 if (meta.color_space or "").startswith("gray") else 3
    )
    dtype = meta.dtype or "uint8"

    return ImageInfo(
        width=meta.width,
        height=meta.height,
        channels=channels,
        dtype=dtype,
        file_size=meta.file_size or 0,
        format=meta.fmt or "unknown",
        dpi=meta.dpi,
        color_space=meta.color_space or (meta.pil_mode or "unknown"),
    )
