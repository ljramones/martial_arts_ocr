# src/eval_ensemble.py
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from src.predict_model import load_model as load_single  # arch-aware


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
def predict_ensemble_logits(img: Image.Image, m1, dev, tfm1, m2, tfm2, bands=True) -> torch.Tensor:
    L1 = _logits_of(img, m1, dev, tfm1)
    L2 = _logits_of(img, m2, dev, tfm2)
    if bands:
        for v in _bands(img, frac=0.20):
            L1 += _logits_of(v, m1, dev, tfm1)
            L2 += _logits_of(v, m2, dev, tfm2)
        L1 /= 5.0; L2 /= 5.0
    return (L1 + L2) / 2.0  # [1,4]

def main(ckpt_a: str, ckpt_b: str, test_dir: str, bands: bool, verbose: int):
    # Load both models
    m1, dev, tfm1, size1, _ = load_single(ckpt_a)
    m2, dev, tfm2, size2, _ = load_single(ckpt_b)

    INV = {0:0, 1:90, 2:180, 3:270}
    root = Path(test_dir)
    deg_dirs = [p for p in root.iterdir() if p.is_dir()]
    deg_dirs.sort(key=lambda p: int(p.name))

    total = correct = 0
    per = {0:[0,0], 90:[0,0], 180:[0,0], 270:[0,0]}
    conf = defaultdict(lambda: defaultdict(int))
    printed = {0:0, 90:0, 180:0, 270:0}

    for d in deg_dirs:
        true_deg = int(d.name)
        for img_p in d.iterdir():
            if not img_p.is_file(): continue
            img = Image.open(img_p).convert("RGB")
            L = predict_ensemble_logits(img, m1, dev, tfm1, m2, tfm2, bands=bands)
            probs = torch.softmax(L, dim=1).squeeze(0).cpu().numpy()
            idx = int(probs.argmax()); pred_deg = INV[idx]

            total += 1
            per[true_deg][1] += 1
            conf[true_deg][pred_deg] += 1
            if pred_deg == true_deg:
                correct += 1
                per[true_deg][0] += 1

            if verbose and printed[true_deg] < verbose:
                top2 = sorted(
                    ((INV[i], float(p)) for i,p in enumerate(probs.tolist())),
                    key=lambda kv: kv[1], reverse=True
                )[:2]
                print(f"[{true_deg}] {img_p.name} -> pred {pred_deg} top2={top2}")
                printed[true_deg] += 1

    acc = correct / max(1, total)
    print(f"\nTest accuracy: {correct}/{total} = {acc*100:.2f}%")
    for k in (0, 90, 180, 270):
        c, n = per[k]
        print(f"{k:>4}: {c}/{n} ({(c/max(1,n))*100:.1f}%)")

    print("\nConfusion (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{c:>5}" for c in (0,90,180,270))
    print(header)
    for t in (0,90,180,270):
        row = [conf[t][p] for p in (0,90,180,270)]
        print(f"{t:>5}: " + "  ".join(f"{n:>5}" for n in row))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="e.g., checkpoints/orient_convnext_tiny.pth")
    ap.add_argument("--ckpt_b", required=True, help="e.g., checkpoints/orient_effnetv2s.pth")
    ap.add_argument("--test",  default="data/test")
    ap.add_argument("--no_bands", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()
    main(args.ckpt_a, args.ckpt_b, args.test, bands=not args.no_bands, verbose=args.verbose)
