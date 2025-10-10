# src/predict_ensemble.py
from __future__ import annotations
import argparse
from typing import Tuple, Dict

import torch
from PIL import Image

from src.predict_model import load_model as load_single  # arch-aware loader


@torch.inference_mode()
def _logits_of(img: Image.Image, model, device, tfm) -> torch.Tensor:
    x = tfm(img).unsqueeze(0).to(device)
    return model(x).float()  # [1,4]


def _bands(img: Image.Image, frac: float = 0.20):
    w, h = img.size
    bw = max(8, int(frac * w))
    bh = max(8, int(frac * h))
    left   = img.crop((0,     0,     bw,   h))
    right  = img.crop((w-bw,  0,     w,    h))
    top    = img.crop((0,     0,     w,    bh))
    bottom = img.crop((0,     h-bh,  w,    h))
    return left, right, top, bottom


@torch.inference_mode()
def predict_image_ensemble(
    img_path: str,
    ckpt_a: str,
    ckpt_b: str,
    use_bands: bool = True,
) -> Tuple[int, Dict[int, float]]:
    """
    Ensemble = average of logits from two models (center [+ simple bands]).
    Assumes fixed class order: idx 0..3 -> [0,90,180,270].
    """
    # Load both models (each returns: model, device, tfm, img_size, mapping)
    model_a, device, tfm_a, size_a, _ = load_single(ckpt_a)
    model_b, device, tfm_b, size_b, _ = load_single(ckpt_b)

    img = Image.open(img_path).convert("RGB")

    # Center logits
    La = _logits_of(img, model_a, device, tfm_a)
    Lb = _logits_of(img, model_b, device, tfm_b)

    if use_bands:
        for v in _bands(img, frac=0.20):
            La += _logits_of(v, model_a, device, tfm_a)
            Lb += _logits_of(v, model_b, device, tfm_b)
        La /= 5.0
        Lb /= 5.0

    L = (La + Lb) / 2.0
    probs = torch.softmax(L, dim=1).squeeze(0).cpu().numpy()

    INV = {0: 0, 1: 90, 2: 180, 3: 270}  # fixed order
    idx = int(probs.argmax())
    deg = INV[idx]
    scores = {INV[i]: float(p) for i, p in enumerate(probs.tolist())}
    return deg, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="e.g., checkpoints/orient_convnext_tiny.pth")
    ap.add_argument("--ckpt_b", required=True, help="e.g., checkpoints/orient_effnetv2s.pth")
    ap.add_argument("--img",    required=True)
    ap.add_argument("--no_bands", action="store_true")
    args = ap.parse_args()

    deg, scores = predict_image_ensemble(
        args.img, args.ckpt_a, args.ckpt_b, use_bands=not args.no_bands
    )
    print("pred:", deg, "scores:", scores)
