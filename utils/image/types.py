# utils/image/types.py
"""
Lightweight type aliases for image arrays.

We keep this minimal to avoid pulling in extra deps.
These aliases are purely for readability in hints; no runtime checks.
"""

from __future__ import annotations

from typing import Tuple, TypeAlias, Protocol, runtime_checkable
import numpy as np


# ---- Image array aliases ----------------------------------------------------

# uint8 grayscale, shape (H, W)
GrayU8: TypeAlias = np.ndarray  # dtype=np.uint8, ndim=2

# uint8 BGR color, shape (H, W, 3) (OpenCV default channel order)
BGRU8: TypeAlias = np.ndarray   # dtype=np.uint8, ndim=3, channels=3

# Generic image (either gray or BGR). Use when function accepts both.
ImgU8: TypeAlias = np.ndarray   # dtype=np.uint8, ndim in {2,3}


# ---- Common geometry/shape aliases -----------------------------------------

SizeHW: TypeAlias = Tuple[int, int]   # (height, width)
PointXY: TypeAlias = Tuple[int, int]  # (x, y)


# ---- Optional protocol for debug sink --------------------------------------

@runtime_checkable
class DebugWriter(Protocol):
    """
    Minimal protocol for classes that accept debug image writes.
    Implementations should be tolerant to both Gray and BGR arrays.
    """
    def write(self, tag: str, mat: np.ndarray) -> None: ...
