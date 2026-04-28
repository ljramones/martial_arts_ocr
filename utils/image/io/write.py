# utils/image/io/write.py
"""
Image writing utilities for the Martial Arts OCR system.

Responsibilities:
- Save NumPy images to disk with format-aware parameters
- Robust color-space handling (assumes OpenCV BGR by default)
- Optional atomic writes to avoid partial/corrupt files
- In-memory encoding for pipelines (HTTP, MinIO, etc.)

Conventions:
- Input arrays are assumed to be OpenCV BGR unless `input_mode` is specified.
- For grayscale, pass a 2D array or set input_mode="gray".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ColorMode = Literal["bgr", "rgb", "gray"]
PathLike = Union[str, Path]

__all__ = ["save_image", "imencode"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_bgr(img: np.ndarray, input_mode: ColorMode) -> np.ndarray:
    """
    Normalize incoming image to BGR for cv2.imwrite compatibility.
    - 2D arrays are returned as-is (treated as GRAY).
    - 3-channel arrays are converted as needed.
    """
    if img.ndim == 2:
        return img
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape for saving: {img.shape}")

    arr = img
    if arr.shape[2] == 4:
        # Drop alpha to ensure consistent 3-channel output where required.
        arr = arr[:, :, :3]

    if input_mode == "bgr":
        return arr
    if input_mode == "rgb":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if input_mode == "gray":
        # Caller passed a 3-channel array but claims "gray": convert to gray explicitly
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def _format_params_for_suffix(
    suffix: str,
    quality: int,
    progressive_jpeg: bool,
    png_compression: Optional[int],
    tiff_lzw: bool,
) -> Sequence[int]:
    """
    Build cv2.imwrite parameter list based on output format and knobs.
    Notes:
    - JPEG quality: 1..100
    - PNG compression: 0..9 (higher = smaller, slower)
    - TIFF: OpenCV uses numeric codes; LZW is '1' in many builds. If unsure, omit.
    - WEBP quality: 1..100
    """
    params: list[int] = []
    suf = suffix.lower()

    if suf in (".jpg", ".jpeg"):
        q = int(np.clip(quality, 1, 100))
        params.extend([cv2.IMWRITE_JPEG_QUALITY, q])
        # Progressive & optimize flags are available in most OpenCV builds.
        try:
            if progressive_jpeg:
                params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
            params.extend([cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        except Exception:
            pass

    elif suf == ".png":
        if png_compression is None:
            # Map 1..100 "quality" to PNG level 9..0 (inverse). Clamp to 0..9.
            level = int(np.clip(round((100 - quality) / 10), 0, 9))
        else:
            level = int(np.clip(png_compression, 0, 9))
        params.extend([cv2.IMWRITE_PNG_COMPRESSION, level])

    elif suf in (".tif", ".tiff"):
        # Use LZW if requested (value 1 in many OpenCV builds). If the build
        # doesn't support it, OpenCV will ignore/raise; caller will see False.
        if tiff_lzw:
            try:
                params.extend([cv2.IMWRITE_TIFF_COMPRESSION, 1])  # 1=LZW in OpenCV
            except Exception:
                pass

    elif suf == ".webp":
        q = int(np.clip(quality, 1, 100))
        try:
            params.extend([cv2.IMWRITE_WEBP_QUALITY, q])
        except Exception:
            pass

    # JP2 and others: let OpenCV defaults handle them.
    return params


def save_image(
    image: np.ndarray,
    output_path: PathLike,
    *,
    quality: int = 95,
    create_dirs: bool = True,
    input_mode: ColorMode = "bgr",
    progressive_jpeg: bool = True,
    png_compression: Optional[int] = None,
    tiff_lzw: bool = True,
    atomic: bool = True,
) -> bool:
    """
    Save an image to disk with format-aware parameters.

    Args:
        image: NumPy array (BGR by default). 2D for gray, HxWx3 for color.
        output_path: Destination path; suffix determines format.
        quality: JPEG/WEBP quality (1-100). For PNG, mapped to compression if
                 png_compression is None.
        create_dirs: Create parent directories as needed.
        input_mode: Color space of the input array: {"bgr","rgb","gray"}.
        progressive_jpeg: Try to write progressive JPEGs (ignored if unsupported).
        png_compression: Explicit PNG compression level 0..9 (overrides mapping).
        tiff_lzw: Try to apply LZW compression for TIFF outputs.
        atomic: Write to a temp file and replace, to avoid partial writes.

    Returns:
        True on success, False otherwise.
    """
    try:
        path = Path(output_path)
        if create_dirs:
            _ensure_parent(path)

        # Normalize image for OpenCV writer
        bgr = _to_bgr(image, input_mode)

        # Prepare per-format params
        params = _format_params_for_suffix(
            path.suffix, quality, progressive_jpeg, png_compression, tiff_lzw
        )

        # Decide final path (atomic temp then replace)
        if atomic:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            final_target = path
            target_path = tmp_path
        else:
            target_path = path
            final_target = None  # type: ignore[assignment]

        ok = cv2.imwrite(str(target_path), bgr, params)
        if not ok:
            logger.error("cv2.imwrite failed for: %s", target_path)
            # Clean up temp on failure
            if atomic and target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            return False

        # Atomic finalize
        if atomic and final_target is not None:
            try:
                os.replace(str(target_path), str(final_target))
            except Exception as e:
                logger.error("Atomic rename failed %s -> %s: %s", target_path, final_target, e)
                # Best effort cleanup: if replace failed and temp exists, remove temp
                try:
                    if target_path.exists():
                        target_path.unlink()
                except Exception:
                    pass
                return False

        out = final_target if atomic else target_path
        try:
            size = out.stat().st_size  # type: ignore[union-attr]
        except Exception:
            size = -1
        logger.debug("Saved image to %s (size: %s bytes)", out, f"{size:,}" if size >= 0 else "?")
        return True

    except Exception as e:
        logger.error("Failed to save image to %s: %s", output_path, e)
        return False


def imencode(
    ext: str,
    image: np.ndarray,
    *,
    input_mode: ColorMode = "bgr",
    quality: int = 95,
    png_compression: Optional[int] = None,
    progressive_jpeg: bool = True,
    tiff_lzw: bool = True,
) -> Tuple[bool, np.ndarray]:
    """
    Encode an image to a memory buffer (e.g., b".jpg", b".png") for HTTP/S3.

    Args:
        ext: File extension with dot (e.g., ".jpg", ".png", ".webp", ".tif").
        image: NumPy array.
        input_mode: Color space of the input array.
        quality: JPEG/WEBP quality (1-100). For PNG, mapped if png_compression is None.
        png_compression: Explicit PNG compression level 0..9.
        progressive_jpeg: Attempt progressive JPEG if supported.
        tiff_lzw: Try LZW compression when encoding TIFF.

    Returns:
        (ok, buf): ok indicates success; buf is a 1D uint8 array with encoded bytes.
    """
    bgr = _to_bgr(image, input_mode)
    params = _format_params_for_suffix(ext, quality, progressive_jpeg, png_compression, tiff_lzw)
    try:
        ok, buf = cv2.imencode(ext, bgr, params)
        if not ok:
            logger.error("cv2.imencode failed for ext=%s", ext)
        return ok, buf
    except Exception as e:
        logger.error("imencode error for ext=%s: %s", ext, e)
        return False, np.empty((0,), dtype=np.uint8)
