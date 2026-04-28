# utils/image/io/validate.py
"""
Image validation utilities for the Martial Arts OCR system.

Responsibilities:
- Validate image files on disk (existence, extension, decodability, min size)
- Validate in-memory bytes
- Lightweight format/extension sniffing helpers

Primary APIs:
- validate_image_file(path, ...)
- validate_image_bytes(data, ...)
- is_supported_extension(ext)
- sniff_image_format(path | bytes)

Notes:
- Uses Pillow's verify() to catch truncated/corrupt files, then re-opens for size/mode.
- Accepts a conservative default set of extensions used in this project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".jp2", ".j2k",
)

# Heuristic floor to avoid obviously bogus inputs
DEFAULT_MIN_SIZE: Tuple[int, int] = (10, 10)  # (w, h)


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    reason: str = "ok"
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    mode: Optional[str] = None
    size_bytes: Optional[int] = None
    dpi: Optional[int] = None


__all__ = [
    "ValidationReport",
    "validate_image_file",
    "validate_image_bytes",
    "is_supported_extension",
    "sniff_image_format",
]


def is_supported_extension(ext: str, allowed_extensions: Optional[Iterable[str]] = None) -> bool:
    """
    Case-insensitive extension check (ext may include or exclude the dot).
    """
    if not ext:
        return False
    if not ext.startswith("."):
        ext = "." + ext
    allowed = tuple(e.lower() for e in (allowed_extensions or DEFAULT_EXTENSIONS))
    return ext.lower() in allowed


def _extract_dpi(pil_img: Image.Image) -> Optional[int]:
    info = getattr(pil_img, "info", {}) or {}
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


def _validate_opened_image(pil_img: Image.Image, min_size: Tuple[int, int]) -> Tuple[bool, str, int, int, str, str, Optional[int]]:
    w, h = pil_img.size
    fmt = pil_img.format or "unknown"
    mode = pil_img.mode
    dpi = _extract_dpi(pil_img)

    min_w, min_h = min_size
    if w < max(1, min_w) or h < max(1, min_h):
        return (False, f"too small ({w}x{h}) < {min_w}x{min_h}", w, h, fmt, mode, dpi)

    return (True, "ok", w, h, fmt, mode, dpi)


def validate_image_file(
    file_path: PathLike,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    require_extension_match: bool = True,
    min_size: Tuple[int, int] = DEFAULT_MIN_SIZE,
) -> ValidationReport:
    """
    Validate an image at a filesystem path.

    Args:
        file_path: path to the image.
        allowed_extensions: whitelist; defaults to DEFAULT_EXTENSIONS.
        require_extension_match: if True, reject files with disallowed extensions.
        min_size: (min_width, min_height).

    Returns:
        ValidationReport with details.
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return ValidationReport(False, reason="not found")

        if not path.is_file():
            return ValidationReport(False, reason="not a file")

        if require_extension_match:
            if not is_supported_extension(path.suffix, allowed_extensions):
                return ValidationReport(False, reason=f"unsupported extension {path.suffix!r}")

        size_bytes: Optional[int]
        try:
            size_bytes = path.stat().st_size
        except Exception:
            size_bytes = None

        # First pass: verify() to catch corruption without decoding pixels fully.
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            logger.debug("PIL verify() failed for %s: %s", path, e)
            return ValidationReport(False, reason=f"corrupt or unreadable: {e}", size_bytes=size_bytes)

        # Re-open for metadata (verify() closes the file)
        try:
            with Image.open(path) as img2:
                ok, reason, w, h, fmt, mode, dpi = _validate_opened_image(img2, min_size)
        except Exception as e:
            return ValidationReport(False, reason=f"failed to reopen after verify: {e}", size_bytes=size_bytes)

        return ValidationReport(
            ok, reason, width=w, height=h, format=fmt, mode=mode, size_bytes=size_bytes, dpi=dpi
        )

    except Exception as e:
        logger.error("Validation error for %s: %s", file_path, e)
        return ValidationReport(False, reason=f"exception: {e}")


def validate_image_bytes(
    data: bytes,
    *,
    min_size: Tuple[int, int] = DEFAULT_MIN_SIZE,
) -> ValidationReport:
    """
    Validate an image supplied as encoded bytes (e.g., from HTTP/S3).

    Args:
        data: encoded image bytes.
        min_size: (min_width, min_height).

    Returns:
        ValidationReport with details (size_bytes is len(data)).
    """
    try:
        from io import BytesIO

        size_bytes = len(data)

        # verify() equivalent path for bytes
        try:
            with Image.open(BytesIO(data)) as img:
                img.verify()
        except Exception as e:
            return ValidationReport(False, reason=f"bytes corrupt or unreadable: {e}", size_bytes=size_bytes)

        # Re-open for metadata
        try:
            with Image.open(BytesIO(data)) as img2:
                ok, reason, w, h, fmt, mode, dpi = _validate_opened_image(img2, min_size)
        except Exception as e:
            return ValidationReport(False, reason=f"bytes failed reopen after verify: {e}", size_bytes=size_bytes)

        return ValidationReport(
            ok, reason, width=w, height=h, format=fmt, mode=mode, size_bytes=size_bytes, dpi=dpi
        )

    except Exception as e:
        logger.error("Validation error for bytes: %s", e)
        return ValidationReport(False, reason=f"exception: {e}")


def sniff_image_format(source: Union[PathLike, bytes]) -> Optional[str]:
    """
    Best-effort format sniffing using Pillow. Returns a short format string like
    'JPEG', 'PNG', 'TIFF', or None if undetectable.
    """
    try:
        if isinstance(source, (str, Path)):
            with Image.open(source) as img:
                return img.format
        else:
            from io import BytesIO
            with Image.open(BytesIO(source)) as img:
                return img.format
    except Exception:
        return None
