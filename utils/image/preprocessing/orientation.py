# utils/image/preprocessing/orientation.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np


@dataclass
class OrientationScores:
    contour_upright_ratio: Dict[int, float]
    line_anisotropy: Dict[int, float]
    footer_polarity: Dict[int, float]
    blended: Dict[int, float]


# ---------------------
# Small utilities
# ---------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _rotate_deg(gray: np.ndarray, deg: int) -> np.ndarray:
    d = ((deg % 360) + 360) % 360
    if d == 0:
        return gray
    if d == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Fallback (shouldn't happen)
    M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), float(d), 1.0)
    return cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC)


def _write_dbg(debug_dir: Optional[str], name: str, mat: np.ndarray) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, name), mat)


# ---------------------
# Preprocess helpers
# ---------------------

def _auto_trim_borders(gray: np.ndarray, pad: int = 6) -> np.ndarray:
    """
    Quickly trims thick dark borders/scan edges so they don't dominate projections.
    """
    # Otsu -> invert to highlight ink/page edges
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x = max(0, x - pad)
    y = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    return gray[y:y2, x:x2]


def _binarize_for_orientation(gray: np.ndarray) -> np.ndarray:
    """
    Binarization tuned for Latin text on noisy scans.
    Steps:
      1) CLAHE for local contrast
      2) Gentle blur
      3) Adaptive threshold (robust to uneven background)
      4) Otsu; intersect with adaptive to recover breaks
      5) Light open/close to clean specks and seal pinholes
    Returns a binary (ink=255) image.
    """
    h, w = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    g = cv2.GaussianBlur(g, (3, 3), 0)

    # Adaptive block size scaled to page size
    side = min(h, w)
    block_size = 31 if side < 1600 else 41  # try 41/51 if your pages are very large
    if block_size % 2 == 0:
        block_size += 1

    adap = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=8
    )

    # Otsu thresholding
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Intersection: keep pixels both methods consider ink
    mask = cv2.bitwise_and(adap, otsu)

    # Define the kernel needed for the morphology operations below
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Remove tiny specks / fill tiny holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)
    return mask

def _band_density(binary: np.ndarray, frac: float = 0.16) -> dict:
    """Ink density in outer bands (0..1)."""
    h, w = binary.shape
    band_h, band_w = max(1, int(h * frac)), max(1, int(w * frac))
    ink = lambda m: float(m.sum()) / (255.0 * m.size + 1e-6)
    return {
        "top":    ink(binary[:band_h, :]),
        "bottom": ink(binary[h-band_h:, :]),
        "left":   ink(binary[:, :band_w]),
        "right":  ink(binary[:, w-band_w:]),
    }

def _baseline_slope_sign(gray: np.ndarray) -> float:
    """
    Estimate sign of dominant near-horizontal stroke slope.
    Returns ~[-1, +1]; positive means 'as scanned' (heuristic).
    """
    # Edges then HoughLinesP; keep shortish, near-horizontal segments
    edges = cv2.Canny(gray, 60, 160, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.deg2rad(1.0), threshold=60,
                           minLineLength=40, maxLineGap=6)
    if segs is None or len(segs) < 10:
        return 0.0
    slopes = []
    for x1, y1, x2, y2 in segs[:,0,:]:
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))  # degrees
        if abs(ang) <= 10:  # near-horizontal
            slopes.append(ang)
    if not slopes:
        return 0.0
    mean_ang = float(np.median(slopes))
    # squash to [-1,1], preserving sign
    return float(np.tanh(np.radians(mean_ang) * 3.0))


# ---------------------
# Component extraction
# ---------------------

def _component_stats(binary: np.ndarray) -> Tuple[List[tuple], np.ndarray, np.ndarray]:
    """
    Extract candidate glyph components with shape/area filters.
    Returns:
      - filtered list of boxes/metrics
      - labels image (int32)
      - kept mask (uint8) where kept components are 255 (for debug)
    Each item in list: (x, y, bw, bh, area, ar, extent, solidity)
    """
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    area_img = h * w

    boxes: List[tuple] = []
    kept_mask = np.zeros_like(binary)

    # Tunables (dataset dependent)
    MIN_ABS_AREA = 16
    MIN_REL_AREA = 1e-6
    MAX_REL_AREA = 0.02  # stricter than 0.08; reject big diagrams/blocks
    AR_MIN, AR_MAX = 0.15, 6.5           # aspect = bh/bw
    EXTENT_MIN, EXTENT_MAX = 0.18, 0.95  # area / (bw*bh)
    SOLIDITY_MIN = 0.50                  # tighter than 0.25

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < MIN_ABS_AREA:
            continue

        rel = area / (area_img + 1e-9)
        if rel < MIN_REL_AREA or rel > MAX_REL_AREA:
            continue

        # --- isolate this component only using the labels image (critical fix) ---
        comp_roi = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255

        # Geom features
        extent = float(area) / float(bw * bh + 1e-6)
        ar = (bh + 1e-6) / (bw + 1e-6)

        cnts, _ = cv2.findContours(comp_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        a_sum = sum(cv2.contourArea(c) for c in cnts)
        hull = cv2.convexHull(np.vstack(cnts))
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = a_sum / hull_area

        # Heuristics tuned for glyph-like blobs
        if not (AR_MIN <= ar <= AR_MAX):
            continue
        if not (EXTENT_MIN <= extent <= EXTENT_MAX):
            continue
        if solidity < SOLIDITY_MIN:
            continue

        boxes.append((x, y, bw, bh, area, ar, extent, solidity))
        kept_mask[y:y + bh, x:x + bw][comp_roi > 0] = 255

    return boxes, labels, kept_mask


# ---------------------
# Scoring signals
# ---------------------

def _upright_ratio(boxes: List[tuple]) -> float:
    """
    The 'upright character' ratio: count of tall>wide over all boxes.
    """
    if not boxes:
        return 0.0
    tall = sum(1 for (_x, _y, bw, bh, *_rest) in boxes if bh > bw)
    return tall / max(1, len(boxes))


def _line_anisotropy(binary: np.ndarray) -> float:
    """
    Measures how strongly the page exhibits *horizontal* text lines.
    - Compute row and column projections.
    - Score = (row peakiness) - (column peakiness).
    Positive => more horizontal line structure (portrait text).
    """
    row = (binary.sum(axis=1).astype(np.float32)) / (255.0 * max(1, binary.shape[1]))
    col = (binary.sum(axis=0).astype(np.float32)) / (255.0 * max(1, binary.shape[0]))

    # Smooth then measure variance as a simple 'peakiness' proxy
    def peakiness(v: np.ndarray) -> float:
        v_s = cv2.GaussianBlur(v.reshape(-1, 1), (1, 9), 0).ravel()
        v_s = np.clip(v_s, 1e-6, None)
        return float(np.var(v_s))

    row_p = peakiness(row)
    col_p = peakiness(col)
    return row_p - col_p  # >0 => portrait; <0 => landscape


def _footer_polarity(binary: np.ndarray, orientation_deg: int) -> float:
    """
    Polarity tiebreak:
      - Portrait (0/180): expect more small components near bottom (page numbers/footers).
      - Landscape (90/270): expect more small components on the right (rotated bottom).
    Returns a signed score in [-1, +1] where positive means 'upright' for that orientation family.
    """
    h, w = binary.shape
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    smalls: List[tuple] = []
    max_small = max(200, int(0.001 * h * w))
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if 25 <= area <= max_small:
            smalls.append((x, y, bw, bh))

    if not smalls:
        return 0.0

    smalls = np.array(smalls, dtype=np.int32)
    xs = smalls[:, 0] + smalls[:, 2] / 2.0
    ys = smalls[:, 1] + smalls[:, 3] / 2.0

    margin = 0.15  # 15% band
    if orientation_deg in (0, 180):
        band = int(h * margin)
        bottom_count = np.sum(ys >= (h - band))
        top_count = np.sum(ys <= band)
        score = (bottom_count - top_count) / max(1, bottom_count + top_count)
        return float(score if orientation_deg == 0 else -score)
    else:
        band = int(w * margin)
        right_count = np.sum(xs >= (w - band))
        left_count = np.sum(xs <= band)
        score = (right_count - left_count) / max(1, right_count + left_count)
        return float(score if orientation_deg == 90 else -score)


def _blend_scores(upright_ratio: float, anisotropy: float, footer: float) -> float:
    """
    Weighted blend. The anisotropy is strong for portrait vs. landscape,
    while upright_ratio and footer help disambiguate 0 vs 180 / 90 vs 270.
    """
    # Squash anisotropy into ~[-1,1]
    an = float(np.tanh(anisotropy * 3.0))
    # Default weights; adjust if your 90/270 (landscape) confusions persist
    return 0.60 * an + 0.30 * upright_ratio + 0.10 * footer


# ---------------------
# Public API
# ---------------------

def choose_coarse_orientation(
    img: np.ndarray,
    debug: Optional[object] = None,       # DebugSink (facade passes this)
    debug_dir: Optional[str] = None,      # legacy path still accepted
    osd_hint: Optional[float] = None,     # 0..360
    proj_weight: float = 1.0,
    horiz_weight: float = 0.8,            # unused here but accepted
    **_unused,
) -> Tuple[int, Dict[str, Dict[int, float]]]:
    """
    Try {0,90,180,270} and return (best_deg, score_breakdown).
    Adds 'combo' (per-rotation blended) for facade's margin logic.
    """
    # ---- local helpers ----
    def _dbg_text(msg: str) -> None:
        try:
            if debug and hasattr(debug, "text"):
                debug.text("orientation_note", msg)
        except Exception:
            pass

    def _normalize_osd_hint(h) -> Optional[float]:
        if h is None: return None
        try:
            if isinstance(h, (int, float)): val = float(h)
            elif isinstance(h, str):        val = float(h.strip())
            elif isinstance(h, dict):
                for k in ("angle","deg","degrees","rotation","rot"):
                    if k in h: val = float(h[k]); break
                else: return None
            else:
                return None
            return (val % 360.0 + 360.0) % 360.0
        except Exception:
            return None

    def _prior_for(deg: int, hint_deg: Optional[float]) -> float:
        if hint_deg is None: return 0.0
        delta = float(((deg - hint_deg) + 540.0) % 360.0 - 180.0)
        return float(np.cos(np.deg2rad(delta)))  # [-1,1]

    PRIOR_WEIGHT = 0.05  # tiny nudge only

    # ---- pre-rotation ----
    gray0 = _to_gray(img)
    gray0 = _auto_trim_borders(gray0)
    hint_deg = _normalize_osd_hint(osd_hint)

    contour_upright: Dict[int, float] = {}
    line_aniso: Dict[int, float] = {}
    footer: Dict[int, float] = {}
    blended: Dict[int, float] = {}

    # cache per-rotation binary for tie-break usage
    rot_gray: Dict[int, np.ndarray] = {}
    for deg in (0, 90, 180, 270):
        g = _rotate_deg(gray0, deg)
        bin_img = _binarize_for_orientation(g)
        rot_gray[deg] = g

        boxes, _labels, kept_mask = _component_stats(bin_img)
        ur = float(_upright_ratio(boxes))
        an = float(_line_anisotropy(bin_img))
        pol = float(_footer_polarity(bin_img, deg))
        if np.isnan(ur): ur = 0.0
        if np.isnan(an): an = 0.0
        if np.isnan(pol): pol = 0.0

        score = float(_blend_scores(ur, an, pol)) + PRIOR_WEIGHT * _prior_for(deg, hint_deg)

        contour_upright[deg] = ur
        line_aniso[deg] = an
        footer[deg] = pol
        blended[deg] = score

        # Debug artifacts
        if debug and hasattr(debug, "write"):
            debug.write(f"rot{deg}_gray", g)
            debug.write(f"rot{deg}_bin",  bin_img)
            debug.write(f"rot{deg}_kept", kept_mask)
        else:
            _write_dbg(debug_dir, f"rot{deg}_gray.png", g)
            _write_dbg(debug_dir, f"rot{deg}_bin.png",  bin_img)
            _write_dbg(debug_dir, f"rot{deg}_kept.png", kept_mask)

    # ---- initial pick by blended score ----
    best_deg = int(max(blended, key=blended.get)) if blended else 0

    # ---- within-family tie-breaker using tiny OCR sniff (only if close) ----
    # threshold: how close is "too close" between opposing members
    CLOSE = 0.08

    try:
        from utils.image.preprocessing import ocr_osd
        tess = ocr_osd.find_tesseract_bin()
    except Exception:
        tess = None

    if tess:
        # portrait family tie-break (0 vs 180)
        if (abs(line_aniso.get(0, 0.0)) >= abs(line_aniso.get(90, 0.0))) and \
           (abs(line_aniso.get(0, 0.0)) >= abs(line_aniso.get(270, 0.0))):
            s0   = blended.get(0, 0.0)
            s180 = blended.get(180, 0.0)
            if abs(s0 - s180) < CLOSE:
                pick = ocr_osd.upright_0_vs_180(rot_gray[0], tess_bin=tess)  # returns 0 or 180
                _dbg_text(f"tie-break portrait by OCR: {pick} (s0={s0:.3f}, s180={s180:.3f})")
                best_deg = int(pick)

        # landscape family tie-break (90 vs 270) by rotating -90 and reusing 0 vs 180
        else:
            s90 = blended.get(90, 0.0)
            s270 = blended.get(270, 0.0)
            if abs(s90 - s270) < CLOSE:
                # Use the -90° view (i.e., counterclockwise 90) so that:
                #   original 90  -> 0
                #   original 270 -> 180
                g_minus90 = _rotate_deg(gray0, 270)  # -90
                pick0_180 = ocr_osd.upright_0_vs_180(g_minus90, tess_bin=tess)  # returns 0 or 180
                pick = 90 if pick0_180 == 0 else 270
                _dbg_text(f"tie-break landscape by OCR(-90)->mapped: {pick} (s90={s90:.3f}, s270={s270:.3f})")
                best_deg = int(pick)

    # ---- return with 'combo' for facade margin logic ----
    scores = OrientationScores(
        contour_upright_ratio=contour_upright,
        line_anisotropy=line_aniso,
        footer_polarity=footer,
        blended=blended,
    )

    return best_deg, {
        "contour_upright_ratio": scores.contour_upright_ratio,
        "line_anisotropy":       scores.line_anisotropy,
        "footer_polarity":       scores.footer_polarity,
        "blended":               scores.blended,
        "combo":                 scores.blended,  # <--- facade reads this
    }


