# preprocessing/orientation_cnn.py
from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageOps
import torch

# Reuse your arch-aware single-head loader
from orientation_model.src.predict_model import load_model as load_single, predict_image as predict_single

# Optional ensemble (ConvNeXt + EffNet) if you keep the file
try:
    from orientation_model.src.predict_ensemble import predict_image_ensemble
except Exception:
    predict_image_ensemble = None  # ensemble optional

_MODEL = None
_DEVICE = None
_TFM = None
_SIZE = None
_IDX_FIXED = None
_CKPT_CNX = None
_CKPT_EFN = None

def init_orientation_model(ckpt_convnext: str,
                           ckpt_effnet: Optional[str] = None) -> None:
    """
    Initialize ConvNeXt-Tiny (best single). Optionally set EffNetV2-S for conditional ensemble.
    """
    global _MODEL, _DEVICE, _TFM, _SIZE, _IDX_FIXED, _CKPT_CNX, _CKPT_EFN
    if _MODEL is None:
        _MODEL, _DEVICE, _TFM, _SIZE, _IDX_FIXED = load_single(ckpt_convnext)
        _CKPT_CNX = ckpt_convnext
        _CKPT_EFN = ckpt_effnet  # may be None

def _np_to_pil(np_img: np.ndarray) -> Image.Image:
    # EXIF-safe: normalize orientation only for file inputs; for arrays assume raw pixels
    if np_img.ndim == 2:
        pil = Image.fromarray(np_img.astype(np.uint8), mode="L").convert("RGB")
    elif np_img.ndim == 3 and np_img.shape[2] == 3:
        pil = Image.fromarray(np_img.astype(np.uint8), mode="RGB")
    else:
        # BGRA or other: convert best-effort
        pil = Image.fromarray(np_img[..., :3].astype(np.uint8), mode="RGB")
    return pil

def predict_degrees(np_img: np.ndarray,
                    use_ensemble_if_low_margin: bool = True,
                    margin: float = 0.55) -> Tuple[int, Dict[int, float], float, str]:
    """
    Returns (deg, scores_by_deg, top1_prob, model_used)
    model_used in {"convnext", "ensemble"}
    """
    assert _MODEL is not None, "init_orientation_model(...) must be called first"
    pil = _np_to_pil(np_img)

    # Single (ConvNeXt) first
    deg, scores = predict_single(pil, _MODEL, _DEVICE, _TFM, _SIZE, _IDX_FIXED, bands=True)
    top = max(scores, key=scores.get); top_p = float(scores[top])

    if use_ensemble_if_low_margin and _CKPT_EFN and predict_image_ensemble and top_p < margin:
        # Conditional ensemble
        deg2, scores2 = predict_image_ensemble(pil, _CKPT_CNX, _CKPT_EFN, use_bands=True)
        top2 = max(scores2, key=scores2.get); top2_p = float(scores2[top2])
        return deg2, scores2, top2_p, "ensemble"

    return deg, scores, top_p, "convnext"
