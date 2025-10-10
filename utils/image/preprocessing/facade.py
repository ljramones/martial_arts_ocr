# utils/image/facade.py
"""
Façade for OCR image preprocessing.

Public API preserved:
- class ImageProcessor
  - deskew_image(image, max_angle=30.0) -> np.ndarray     # Phase-1: orientation + small-angle deskew
  - preprocess_for_ocr(image, apply_deskew=None, apply_denoise=None, auto_orient=True) -> np.ndarray

Exposed debug attrs for your harness:
- _last_osd_deg_hint: Optional[int]
- _last_phase1_debug: Optional[str]
- _last_choose_scores: Dict[str, Any]
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging
import cv2
import numpy as np
from PIL import Image  # NEW: for polarity tie-break visual sampling

from utils.image.preprocessing import binarize, denoise, debug_io, geometry, ocr_osd, orientation
from utils.image.shared_utils import _to_gray_u8

# NEW: CNN orientation wrapper (ConvNeXt default + optional ensemble)
try:
    from utils.image.preprocessing.orientation_cnn import (
        init_orientation_model as _cnn_init,
        predict_degrees as _cnn_predict,
    )
    _HAS_CNN = True
except Exception:
    _HAS_CNN = False

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        from config import get_config
        cfg = dict(getattr(get_config(), "IMAGE_PROCESSING", {}))
        if config_override:
            cfg.update(config_override)
        self.config: Dict[str, Any] = cfg

        # Debug sink config
        debug_dir = cfg.get("DEBUG_DIR")
        debug_prefix = cfg.get("DEBUG_FILE_PREFIX", "")
        debug_limit = cfg.get("DEBUG_FILE_LIMIT")
        self.debug = debug_io.DebugSink(
            dirpath=debug_dir,
            prefix=debug_prefix,
            limit=debug_limit
        )

        self.tesseract_bin = cfg.get("tesseract_bin") or ocr_osd.find_tesseract_bin()
        self._last_osd_deg_hint: Optional[int] = None
        self._last_phase1_debug: Optional[str] = None
        self._last_choose_scores: Dict[str, Any] = {}

        # ---------------- CNN orientation init (lazy-safe) ----------------
        self._cnn_ready = False
        self._cnn_margin = float(cfg.get("ORIENT_ENS_MARGIN", 0.55))
        cnx = cfg.get("ORIENT_CKPT_CONVNEXT")
        efn = cfg.get("ORIENT_CKPT_EFFNET", None)
        if _HAS_CNN and cnx:
            try:
                _cnn_init(cnx, efn)
                self._cnn_ready = True
            except Exception as e:
                logger.warning("CNN orientation init failed: %s", e)
                self._cnn_ready = False

    # -----------------------------------------------------------------------
    # Phase-1: Orientation + small-angle deskew
    # -----------------------------------------------------------------------
    def deskew_image(self, image: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
        """
        Auto-orient (0/90/180/270) and correct small skew.
        Preserves input shape/channels (operates on color when provided).
        """
        try:
            disable_heur = bool(self.config.get("DISABLE_HEURISTIC_FALLBACK", False))

            # ---- config knobs ----
            blur_thresh = self.config.get('BLUR_PREBOOST_THRESHOLD', 180.0)
            preboost_strength = self.config.get('PREBOOST_UNSHARP_STRENGTH', 1.8)
            preboost_blur_weight = self.config.get('PREBOOST_UNSHARP_BLUR_WEIGHT', -0.8)

            sniff_langs = self.config.get('OCR_SNIFF_LANGUAGES', 'eng+jpn')
            hough_thresh = self.config.get('DESKEW_HOUGH_THRESHOLD', 100)
            hough_min_len = self.config.get('DESKEW_HOUGH_MIN_LINE_LENGTH', 100)
            hough_max_gap = self.config.get('DESKEW_HOUGH_MAX_LINE_GAP', 10)

            # Heuristic chooser weights (used only on fallback)
            proj_weight = self.config.get('ORIENTATION_PROJ_WEIGHT', 1.0)
            horiz_weight = self.config.get('ORIENTATION_HORIZ_WEIGHT', 0.8)

            # Polarity tie-break knobs (portrait 0 vs 180 only)
            tie_margin  = float(self.config.get('POLARITY_TIE_MARGIN', 0.10))
            tie_frac    = float(self.config.get('POLARITY_TIE_FRAC',   0.18))
            tie_thresh  = float(self.config.get('POLARITY_TIE_THRESH', 0.08))

            work = image.copy()
            gray_raw = _to_gray_u8(work)
            blur0 = denoise.blur_var(gray_raw)

            # light preboost for OSD sniff & line finding
            boosted = denoise.preboost_blurry(
                gray_raw,
                blur_threshold=blur_thresh,
                strength=preboost_strength,
                blur_weight=preboost_blur_weight
            )

            # Optional OSD hint (used only for debug / heuristic fallback)
            osd_deg, osd_conf = ocr_osd.osd_rotate_deg(boosted, self.tesseract_bin)
            self._last_osd_deg_hint = osd_deg if ((osd_conf or 0) >= 3.0) else None

            chosen: Optional[int] = None
            path = ""

            # Helper: portrait polarity tie-break using top/bottom ink density
            def _bottom_heavier(pil_img: Image.Image, frac=tie_frac, thresh=tie_thresh) -> int | None:
                w, h = pil_img.size
                bh = max(8, int(frac * h))
                top    = np.asarray(pil_img.crop((0, 0, w, bh)).convert("L"))
                bottom = np.asarray(pil_img.crop((0, h - bh, w, h)).convert("L"))
                # darker == ink; normalize 0..1
                ink_top = 1.0 - top.mean() / 255.0
                ink_bot = 1.0 - bottom.mean() / 255.0
                diff = ink_bot - ink_top
                if diff > thresh:   # bottom clearly heavier -> 180
                    return 180
                if diff < -thresh:  # top clearly heavier -> 0
                    return 0
                return None

            # ---------------- CNN orientation first ----------------
            used_model = None
            top_p = None
            cnn_scores = {}
            if self._cnn_ready:
                try:
                    deg, scores, prob, model_used = _cnn_predict(
                        work, use_ensemble_if_low_margin=True, margin=self._cnn_margin
                    )
                    chosen = int(deg)
                    top_p = float(prob)
                    cnn_scores = scores or {}
                    used_model = model_used
                    if chosen in (90, 180, 270):
                        work = geometry.rotate_deg(work, chosen)
                    path = f"CNN[{used_model} p={top_p:.2f}]"

                    # --- NEW: portrait polarity tie-break for 0 vs 180 when nearly tied ---
                    if chosen in (0, 180) and cnn_scores:
                        p0   = float(cnn_scores.get(0,   0.0))
                        p180 = float(cnn_scores.get(180, 0.0))
                        if abs(p0 - p180) < tie_margin:
                            # Build a PIL image for sampling (RGB)
                            if work.ndim == 2:
                                pil = Image.fromarray(work, mode="L").convert("RGB")
                            else:
                                pil = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
                            guess = _bottom_heavier(pil)
                            if guess is not None and guess != chosen:
                                work   = geometry.rotate_deg(work, 180)
                                chosen = guess
                                path  += "; polarity-tie"
                except Exception as e:
                    logger.warning("CNN orientation failed, falling back: %s", e)
                    chosen = None

            # ---------------- Heuristic + OSD fallback (honors DISABLE_HEURISTIC_FALLBACK) ----------------
            if chosen is None:
                if osd_deg in (90, 180, 270) and (osd_conf is None or osd_conf >= 3.0):
                    work = geometry.rotate_deg(work, osd_deg)
                    chosen, path = osd_deg, f"OSD(conf={osd_conf:.2f})"
                elif not disable_heur:
                    alt_deg, scores = orientation.choose_coarse_orientation(
                        work,
                        osd_hint=self._last_osd_deg_hint,
                        debug=self.debug,
                        proj_weight=proj_weight,
                        horiz_weight=horiz_weight
                    )
                    self._last_choose_scores = scores if isinstance(scores, dict) else {}
                    combo = self._last_choose_scores.get("combo", {}) if isinstance(self._last_choose_scores, dict) else {}
                    sorted_vals = sorted(combo.values(), reverse=True) if combo else []
                    margin = (sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else (
                        sorted_vals[0] if sorted_vals else 0.0)

                    # small +/- 2° jitter retry if margin is tiny
                    if margin < 0.12:
                        h, w = work.shape[:2]
                        for tweak in (-2.0, 2.0):
                            M = cv2.getRotationMatrix2D((w // 2, h // 2), tweak, 1.0)
                            tmp = cv2.warpAffine(work, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                            d2, s2 = orientation.choose_coarse_orientation(
                                tmp,
                                osd_hint=self._last_osd_deg_hint,
                                debug=None,
                                proj_weight=proj_weight,
                                horiz_weight=horiz_weight
                            )
                            c2 = s2.get("combo", {}) if isinstance(s2, dict) else {}
                            if c2.get(d2, -1e9) > combo.get(alt_deg, -1e9):
                                work, alt_deg, self._last_choose_scores, combo = tmp, d2, s2, c2
                                break

                    if alt_deg in (90, 180, 270):
                        work = geometry.rotate_deg(work, alt_deg)
                    chosen = alt_deg
                    path = "Heur"
                else:
                    # Policy: no heuristic; keep as-is (or rely solely on OSD)
                    chosen = 0
                    path = "None"

            # ---------------- sanity check: allow +90 if clearly better ----------------
            gray_chk = _to_gray_u8(work)
            alt = geometry.sanity_plus_90_if_better(gray_chk)
            if alt is not gray_chk:
                work = geometry.rotate_deg(work, 90)
                chosen = (chosen + 90) % 360 if chosen is not None else 90
                path = (path + "; sanity:+90") if path else "sanity:+90"

            # ---------------- small-angle deskew ----------------
            deskewed, angle_meta = geometry.deskew_small_angle(
                work,
                max_angle=max_angle,
                threshold=hough_thresh,
                min_line_length=hough_min_len,
                max_line_gap=hough_max_gap
            )

            med = angle_meta.get("median", 0.0)
            reason = angle_meta.get("reason", "")

            # record last scores for harness: prefer CNN metrics if present
            if cnn_scores:
                self._last_choose_scores = {"cnn_scores": cnn_scores, "top1_prob": top_p, "model": used_model}

            note = (
                f"chosen={chosen} via {path}; small={med:.2f}{reason}; "
                f"blur={blur0:.1f}; osd=({osd_deg},{osd_conf})"
            )
            self._last_phase1_debug = note
            return deskewed

        except Exception as e:
            self._last_phase1_debug = f"error:{e}"
            logger.exception("deskew_image failed")
            return image

    # -----------------------------------------------------------------------
    # Back-compat: private chooser used by your harness (heuristic only)
    # -----------------------------------------------------------------------
    def _choose_coarse_orientation(self, img: np.ndarray):
        """
        Compatibility wrapper so existing test harness can call
        proc._choose_coarse_orientation(img) and get (deg, scores).
        Uses the legacy heuristic chooser only (no CNN) by design.
        """
        proj_weight = self.config.get('ORIENTATION_PROJ_WEIGHT', 1.0)
        horiz_weight = self.config.get('ORIENTATION_HORIZ_WEIGHT', 0.8)
        return orientation.choose_coarse_orientation(
            img,
            osd_hint=self._last_osd_deg_hint,
            debug=self.debug,
            proj_weight=proj_weight,
            horiz_weight=horiz_weight
        )

    # -----------------------------------------------------------------------
    # Full preprocessing pipeline (for OCR-ready binary output)
    # -----------------------------------------------------------------------
    def preprocess_for_ocr(
            self,
            image: np.ndarray,
            apply_deskew: Optional[bool] = None,
            apply_denoise: Optional[bool] = None,
            auto_orient: bool = True,
    ) -> np.ndarray:
        """
        Orientation/deskew (optional) + perspective correction + denoise + binarize + unsharp.
        Preserves image regions during binarization.
        Returns a binary uint8 image optimized for OCR.
        """
        try:
            # Read binarize config values
            sauvola_window = self.config.get('SAUVOLA_WINDOW', 25)
            sauvola_k = self.config.get('SAUVOLA_K', 0.2)
            unsharp_strength = self.config.get('UNSHARP_STRENGTH', 1.5)
            unsharp_sigma = self.config.get('UNSHARP_SIGMA', 1.0)

            # Start from original; do ops in gray for stability
            gray = _to_gray_u8(image)

            # Gentle upscale for typewriter dots
            scale = float(self.config.get("resize_factor", 1.2))
            if abs(scale - 1.0) > 1e-3:
                gray = geometry.resize(gray, scale)

            # Phase-1 (optional)
            should_deskew = apply_deskew if apply_deskew is not None else self.config.get("deskew", True)
            if auto_orient and should_deskew:
                gray = _to_gray_u8(self.deskew_image(gray))

            # Perspective fix
            gray = geometry.apply_perspective_correction(gray, debug=self.debug)

            # Denoise (light by default)
            should_denoise = apply_denoise if apply_denoise is not None else self.config.get("denoise", True)
            if should_denoise:
                gray = denoise.nl_means(gray, strength="light")

            # Preserve large image regions (diagrams/photos)
            min_img_area = self.config.get('LAYOUT_DETECTION', {}).get('image_block_min_area', 5000)
            image_regions = geometry.find_image_regions(gray, min_area=min_img_area)

            # Binarize the page
            binary = binarize.sauvola(gray, window=sauvola_window, k=sauvola_k)

            # Paste original grayscale back over preserved regions
            for x, y, w, h in image_regions:
                image_crop = gray[y:y + h, x:x + w]
                binary[y:y + h, x:x + w] = image_crop

            # Unsharp after pasting to avoid haloing images
            binary = binarize.unsharp(binary, strength=unsharp_strength, sigma=unsharp_sigma)

            return binarize.normalize(binary)

        except Exception as e:
            logger.exception("preprocess_for_ocr failed")
            return image


# --- exports ---------------------------------------------------------------
__all__ = [
    "ImageProcessor",
    "preprocess_for_captions_np",
    "preprocess_for_fullpage_np",
    "preprocess_for_japanese_np",
]

# --- legacy helpers kept for backward compatibility ------------------------
def preprocess_for_captions_np(np_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY) if np_img.ndim == 3 and np_img.shape[2] == 3 else np_img.copy()
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

def preprocess_for_fullpage_np(np_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY) if np_img.ndim == 3 and np_img.shape[2] == 3 else np_img.copy()
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

def preprocess_for_japanese_np(np_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY) if np_img.ndim == 3 and np_img.shape[2] == 3 else np_img.copy()
    gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 12)
