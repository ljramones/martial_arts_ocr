# utils/image/debug_io.py
"""
Minimal debug sink for optional image/text dumps.

Usage:
    dbg = DebugSink("data/notebook_outputs/debug_output")          # or DebugSink(None) to disable
    dbg.write("chooser_gc_0", img_array)     # saves 0001_chooser_gc_0.png
    dbg.text("phase1_note", note_string)     # saves 0002_phase1_note.txt
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np


class DebugSink:
    """
    Writes debug artifacts when a directory is configured; no-ops otherwise.
    Thread-safe, sequentially numbered filenames for easy tracing.
    """

    def __init__(self, dirpath: Optional[str] = None, prefix: str = "", limit: Optional[int] = None):
        """
        Args:
            dirpath: Directory to write into. If falsy, all methods no-op.
            prefix:  Optional filename prefix, e.g. run id.
            limit:   Optional maximum number of files to emit (images + texts).
        """
        self.dir = dirpath or ""
        self.prefix = f"{prefix}_" if prefix else ""
        self.limit = int(limit) if limit is not None else None
        self._seq = 0
        self._lock = threading.Lock()
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    # ---- public API ---------------------------------------------------------

    def write(self, tag: str, mat: np.ndarray) -> None:
        """
        Save an image (gray or BGR). Silently no-ops if disabled or any error occurs.
        """
        if not self._can_emit():
            return
        try:
            m = self._as_bgr(mat)
            path = self._next_path(f"{tag}.png")
            if path:
                cv2.imwrite(path, m)
        except Exception:
            # Swallow debug errors by design
            pass

    def text(self, tag: str, content: str) -> None:
        """
        Save a small text note alongside images.
        """
        if not self._can_emit():
            return
        try:
            path = self._next_path(f"{tag}.txt")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content or "")
        except Exception:
            pass

    # ---- internals ----------------------------------------------------------

    def _can_emit(self) -> bool:
        if not self.dir:
            return False
        if self.limit is None:
            return True
        with self._lock:
            return self._seq < self.limit

    def _next_path(self, name: str) -> Optional[str]:
        if not self.dir:
            return None
        with self._lock:
            if self.limit is not None and self._seq >= self.limit:
                return None
            self._seq += 1
            fname = f"{self.prefix}{self._seq:04d}_{name}"
        return os.path.join(self.dir, fname)

    @staticmethod
    def _as_bgr(mat: np.ndarray) -> np.ndarray:
        """
        Convert gray/float/odd-channel arrays into uint8 BGR for saving.
        """
        if mat is None:
            raise ValueError("DebugSink._as_bgr: mat is None")

        img = mat
        # Normalize non-uint8 to 0..255
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3:
            if img.shape[2] == 3:
                return img  # assume already BGR
            # e.g., single-channel with shape (H,W,1) or 4-channel; squeeze or drop alpha
            if img.shape[2] == 1:
                return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Fallback: take first channel
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)
