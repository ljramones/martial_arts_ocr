# utils/image/io/read.py
"""
Image reading utilities for the Martial Arts OCR system.

Responsibilities:
- Robust image loading from file paths or bytes
- EXIF-aware orientation correction (via Pillow)
- Output in caller-selected color space: BGR (OpenCV default), RGB, or GRAY
- Sensible fallbacks to cv2 when PIL fails

Notes:
- BGR is returned by default to match OpenCV conventions throughout the codebase.
- For 1-channel sources, GRAY returns a 2D array; BGR/RGB returns 3-channel arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Union, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

ColorMode = Literal["bgr", "rgb", "gray"]

__all__ = ["load_image", "imread_bytes"]


def _pil_open_exif(path: Path) -> Image.Image:
    """
    Open an image with Pillow and apply EXIF orientation if present.
    The caller is responsible for closing the returned PIL image (use context manager or .close()).
    """
    pil_img = Image.open(path)
    # Normalize orientation (handles rotations/mirrors from camera metadata)
    pil_img = ImageOps.exif_transpose(pil_img)
    return pil_img


def _to_numpy_from_pil(pil_img: Image.Image, mode: ColorMode) -> np.ndarray:
    """
    Convert a PIL image to a NumPy array in the requested color mode.
    """
    # Ensure an RGB-ish base for color conversions
    if mode in ("bgr", "rgb"):
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)  # RGB or RGBA -> ndarray
        if arr.ndim == 3 and arr.shape[2] == 4:
            # Drop alpha to keep consistent 3-channel output
            arr = arr[:, :, :3]

        if mode == "rgb":
            return arr
        else:  # "bgr"
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # GRAY path
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    return np.array(pil_img)


def _cv2_imread(path: Path, mode: ColorMode) -> Optional[np.ndarray]:
    """
    Fallback loader via OpenCV with the requested mode.
    """
    if mode == "gray":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img
    else:
        # cv2 reads as BGR by default
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if mode == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # "bgr"


def load_image(
    image_path: Union[str, Path],
    *,
    mode: ColorMode = "bgr",
    use_exif: bool = True,
) -> np.ndarray:
    """
    Load an image from disk with optional EXIF orientation handling.

    Args:
        image_path: Path-like to the image file.
        mode: Output color mode. One of {"bgr", "rgb", "gray"} (default "bgr").
        use_exif: If True, attempt EXIF-aware loading via Pillow first.

    Returns:
        np.ndarray representing the image in the requested mode.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Try EXIF-aware path via PIL first (recommended for phone/camera images)
    if use_exif:
        try:
            with _pil_open_exif(path) as pil_img:
                arr = _to_numpy_from_pil(pil_img, mode)
                logger.debug(
                    "Loaded (PIL+EXIF) %s -> shape=%s dtype=%s mode=%s",
                    path.name,
                    getattr(arr, "shape", None),
                    getattr(arr, "dtype", None),
                    mode,
                )
                return arr
        except Exception as pil_err:
            logger.warning(
                "EXIF-aware load failed for %s (%s); falling back to cv2.imread.",
                path,
                pil_err,
            )

    # Fallback to OpenCV
    img = _cv2_imread(path, mode)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    logger.debug(
        "Loaded (cv2) %s -> shape=%s dtype=%s mode=%s",
        path.name,
        getattr(img, "shape", None),
        getattr(img, "dtype", None),
        mode,
    )
    return img


def imread_bytes(
    data: bytes,
    *,
    mode: ColorMode = "bgr",
    use_exif: bool = True,
) -> np.ndarray:
    """
    Decode an image from a bytes buffer.

    Args:
        data: Encoded image bytes (e.g., JPEG/PNG/WebP).
        mode: Output color mode. One of {"bgr", "rgb", "gray"} (default "bgr").
        use_exif: If True, attempt EXIF-aware loading via Pillow first.

    Returns:
        np.ndarray in the requested color mode.

    Raises:
        ValueError: If the buffer cannot be decoded as an image.
    """
    if use_exif:
        try:
            from io import BytesIO

            with Image.open(BytesIO(data)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                arr = _to_numpy_from_pil(pil_img, mode)
                logger.debug(
                    "Loaded (PIL+EXIF bytes) -> shape=%s dtype=%s mode=%s",
                    getattr(arr, "shape", None),
                    getattr(arr, "dtype", None),
                    mode,
                )
                return arr
        except Exception as pil_err:
            logger.warning(
                "EXIF-aware bytes load failed (%s); falling back to cv2.imdecode.", pil_err
            )

    # cv2 fallback for bytes
    npbuf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(
        npbuf, cv2.IMREAD_GRAYSCALE if mode == "gray" else cv2.IMREAD_COLOR
    )
    if img is None:
        raise ValueError("Could not decode image from bytes buffer.")

    if mode == "rgb" and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.debug(
        "Loaded (cv2 bytes) -> shape=%s dtype=%s mode=%s",
        getattr(img, "shape", None),
        getattr(img, "dtype", None),
        mode,
    )
    return img
