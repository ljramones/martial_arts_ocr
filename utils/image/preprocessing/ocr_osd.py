# utils/image/ocr_osd.py
"""
Tesseract integration for orientation hints.

Provides:
- find_tesseract_bin() -> Optional[str]
- guess_tessdata_dir() -> Optional[str]
- osd_rotate_deg(gray, tess_bin, timeout_sec=5.0) -> (deg|None, conf|None)
- upright_0_vs_180(img_bgr_or_gray, tess_bin, timeout_sec=4.0) -> 0|180

Notes
-----
- All functions are defensive and return safe fallbacks on error.
- `upright_0_vs_180` uses a tiny central band OCR sniff (eng+jpn) and prefers
  the rotation with more alphabetic tokens, then higher mean confidence.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.image.shared_utils import _to_gray_u8


# ---------- discovery ----------

def find_tesseract_bin() -> Optional[str]:
    """
    Locate the `tesseract` executable. Returns None if not found.
    """
    path = shutil.which("tesseract")
    if path:
        return path
    for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
        if os.path.exists(p):
            return p
    return None


def guess_tessdata_dir() -> Optional[str]:
    """
    Best-effort guess of the tessdata directory (for OSD/lang files).
    Checks a repo-local `tessdata/` two levels up first, then common system paths.
    """
    # repo-local: <repo>/tessdata
    try:
        repo_tess = Path(__file__).resolve().parents[2] / "tessdata"
        if (repo_tess / "osd.traineddata").exists():
            return str(repo_tess)
    except Exception:
        pass

    for d in (
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
    ):
        try:
            if os.path.exists(os.path.join(d, "osd.traineddata")):
                return d
        except Exception:
            continue
    return None


# ---------- helpers ----------

def _env_with_tessdata() -> dict:
    env = os.environ.copy()
    td = guess_tessdata_dir()
    if td:
        env["TESSDATA_PREFIX"] = td
    return env


# ---------- OSD (orientation & script detection) ----------

def osd_rotate_deg(gray: np.ndarray, tess_bin: Optional[str], timeout_sec: float = 5.0) -> Tuple[Optional[int], Optional[float]]:
    """
    Run Tesseract OSD (PSM 0) to infer rotation in {0,90,180,270}.
    Returns (deg, confidence) or (None, None) on failure.
    """
    try:
        if not tess_bin:
            return (None, None)
        g = _to_gray_u8(gray)
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "osd.png")
            cv2.imwrite(img_path, g)
            cmd = [tess_bin, img_path, "stdout", "--psm", "0"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=_env_with_tessdata())
            text = (out.stdout or "") + "\n" + (out.stderr or "")
        m_deg = re.search(r"Rotate:\s*(\d+)", text)
        m_conf = re.search(r"Orientation confidence:\s*([0-9.]+)", text)
        deg = int(m_deg.group(1)) % 360 if m_deg else None
        conf = float(m_conf.group(1)) if m_conf else None
        if deg in (0, 90, 180, 270):
            return (deg, conf)
    except Exception:
        pass
    return (None, None)


# ---------- 0 vs 180 upright sniff ----------

def upright_0_vs_180(img_bgr_or_gray: np.ndarray,
                     tess_bin: Optional[str],
                     timeout_sec: float = 4.0,
                     languages: str = 'eng+jpn'
                     ) -> int:
    """
    Decide between 0 and 180 using a tiny OCR sniff over a central band.
    Returns 0 on any failure.
    """
    try:
        if not tess_bin:
            return 0
        g = _to_gray_u8(img_bgr_or_gray)
        H, W = g.shape[:2]
        if H < 40 or W < 40:
            return 0

        # central band avoids margins/headers/footers
        y1, y2 = int(H * 0.35), int(H * 0.65)
        x1, x2 = int(W * 0.10), int(W * 0.90)
        y1 = max(0, min(y1, H));
        y2 = max(0, min(y2, H))
        x1 = max(0, min(x1, W));
        x2 = max(0, min(x2, W))
        if y2 <= y1 or x2 <= x1:
            return 0

        crop = g[y1:y2, x1:x2]
        if crop.size == 0:
            return 0

        # Tesseract's reliability depends on clean, high-contrast input.
        # Binarizing the crop here is the most important change we can make.
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def _score(rot180: bool) -> tuple[int, float]:
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "s.png")
                crop_rot = cv2.rotate(crop, cv2.ROTATE_180) if rot180 else crop
                cv2.imwrite(p, crop_rot)
                try:
                    # <--- CHANGE: Use the languages parameter in the Tesseract command ---
                    cmd = [tess_bin, p, "stdout", "--psm", "6", "-l", languages, "tsv"]
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec,
                                         env=_env_with_tessdata())
                    tsv = out.stdout.splitlines()
                except Exception:
                    return (0, 0.0)

            confs, alpha = [], 0
            for line in tsv[1:]:
                cols = line.split('\t')
                if len(cols) > 11:
                    try:
                        c = float(cols[10])
                    except Exception:
                        c = -1.0
                    if c >= 0:
                        confs.append(c)
                    txt = cols[11]
                    if any(ch.isalpha() for ch in txt):
                        alpha += 1
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            return (alpha, mean_conf)

        a0, c0 = _score(False)
        a180, c180 = _score(True)

        if a0 != a180:
            return 0 if a0 > a180 else 180
        return 0 if c0 >= c180 else 180
    except Exception:
        return 0
