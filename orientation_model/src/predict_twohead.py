# src/predict_twohead.py  — EXIF-safe, two-stage voting with strong center anchor
from __future__ import annotations
import argparse
from typing import Iterable, List, Tuple

import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms as T

from src.model_twohead import TwoHeadMobileNet

class To3Channels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.size(0) == 3 else x.expand(3, -1, -1)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_tfm(size: int) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        To3Channels(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def _pil_bottom(img: Image.Image, frac: float = 0.22) -> Image.Image:
    w, h = img.size; bh = max(8, int(frac * h)); return img.crop((0, h - bh, w, h))
def _pil_top(img: Image.Image, frac: float = 0.20) -> Image.Image:
    w, h = img.size; th = max(8, int(frac * h)); return img.crop((0, 0, w, th))
def _pil_right(img: Image.Image, frac: float = 0.24) -> Image.Image:
    w, h = img.size; bw = max(8, int(frac * w)); return img.crop((w - bw, 0, w, h))
def _pil_left(img: Image.Image, frac: float = 0.24) -> Image.Image:
    w, h = img.size; bw = max(8, int(frac * w)); return img.crop((0, 0, bw, h))
def _hflip(p: Image.Image) -> Image.Image:
    return p.transpose(Image.FLIP_LEFT_RIGHT)
def _vflip(p: Image.Image) -> Image.Image:
    return p.transpose(Image.FLIP_TOP_BOTTOM)

def load_model(ckpt_path: str, device: torch.device | None = None):
    if device is None:
        device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    img_size = int(ckpt.get("img_size", 384))
    model = TwoHeadMobileNet(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, device, img_size

@torch.inference_mode()
def predict_image(
    img_path: str,
    model: TwoHeadMobileNet,
    device,
    scales: Iterable[int],
    bands: bool = True,
    tta: bool = True,
    portrait_tb_weight: float = 1.2,   # portrait: top/bottom emphasis
    portrait_lr_weight: float = 0.3,   # portrait: left/right very light
    landscape_lr_weight: float = 1.2,  # landscape: left/right emphasis
    landscape_tb_weight: float = 0.3,  # landscape: top/bottom very light
    center_weight: float = 4.0,        # STRONG center anchor
) -> Tuple[int, dict[int, float]]:
    # EXIF-safe read so pixels match what you saw when labeling
    img0 = Image.open(img_path)
    img  = ImageOps.exif_transpose(img0).convert("RGB")

    scales = list(dict.fromkeys(int(s) for s in scales if s and s > 0))

    def fwd(pil: Image.Image, tfm: T.Compose):
        x = tfm(pil).unsqueeze(0).to(device)
        fam, pol = model(x)  # logits [1,2],[1,2]
        return fam, pol

    def four_lp(fam_logits: torch.Tensor, pol_logits: torch.Tensor) -> torch.Tensor:
        return TwoHeadMobileNet.combine_to_fourway(fam_logits, pol_logits)  # [1,4] log-probs

    # ---------- Stage A: family from center-only four-way ----------
    acc_center_lp = None; cnt = 0
    for sz in scales:
        tfm = _make_tfm(sz)
        views = [img]
        if tta:
            views += [_hflip(img), _vflip(img)]
        for v in views:
            if min(v.size) < 16: continue
            fam, pol = fwd(v, tfm)
            lp = four_lp(fam, pol)
            acc_center_lp = lp if acc_center_lp is None else acc_center_lp + lp
            cnt += 1
    if acc_center_lp is None:
        tfm = _make_tfm(scales[0] if scales else 384)
        fam, pol = fwd(img, tfm)
        acc_center_lp = four_lp(fam, pol); cnt = 1

    center_probs = torch.softmax(acc_center_lp / max(1, cnt), dim=1).squeeze(0)
    P0, P90, P180, P270 = center_probs.tolist()
    family_idx = 0 if (P0 + P180) >= (P90 + P270) else 1  # 0=portrait, 1=landscape

    # ---------- Stage B: polarity within family (center + family-aware bands) ----------
    acc_lp = None; wsum = 0.0
    for sz in scales:
        tfm = _make_tfm(sz)
        views: List[Tuple[Image.Image, float]] = [(img, center_weight)]
        if bands:
            if family_idx == 0:  # portrait
                views += [
                    (_pil_top(img),    portrait_tb_weight),
                    (_pil_bottom(img), portrait_tb_weight),
                    (_pil_left(img),   portrait_lr_weight),
                    (_pil_right(img),  portrait_lr_weight),
                ]
            else:  # landscape
                views += [
                    (_pil_left(img),   landscape_lr_weight),
                    (_pil_right(img),  landscape_lr_weight),
                    (_pil_top(img),    landscape_tb_weight),
                    (_pil_bottom(img), landscape_tb_weight),
                ]
        if tta:
            aug = []
            for v, w in views:
                aug += [(v, w), (_hflip(v), w), (_vflip(v), w)]
            views = aug

        for v, w in views:
            if min(v.size) < 16: continue
            fam, pol = fwd(v, tfm)
            lp4 = four_lp(fam, pol)

            # mask other family: portrait keep [0,2], landscape keep [1,3]
            mask = torch.zeros(4, device=lp4.device)
            if family_idx == 0:
                mask[1] = mask[3] = -1e9
            else:
                mask[0] = mask[2] = -1e9
            masked = lp4 + mask.unsqueeze(0)
            if torch.isnan(masked).any() or torch.isinf(masked).any(): continue

            acc_lp = masked * w if acc_lp is None else acc_lp + masked * w
            wsum += w

    if acc_lp is None:
        tfm = _make_tfm(scales[0] if scales else 384)
        fam, pol = fwd(img, tfm)
        acc_lp = four_lp(fam, pol)

    acc_lp = acc_lp / max(1e-6, wsum if wsum > 0 else 1.0)
    probs = torch.softmax(acc_lp, dim=1).squeeze(0).cpu().numpy()

    idx_to_deg_4 = {0: 0, 1: 90, 2: 180, 3: 270}
    label = int(probs.argmax())
    deg = idx_to_deg_4[label]
    score_map = {idx_to_deg_4[i]: float(p) for i, p in enumerate(probs.tolist())}
    return deg, score_map

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--scales", default="")
    ap.add_argument("--no_bands", action="store_true")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--center_weight", type=float, default=4.0)
    args = ap.parse_args()

    model, device, img_size = load_model(args.ckpt)
    if args.scales.strip():
        scales: List[int] = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    else:
        scales = [img_size, max(img_size, 448)]

    deg, scores = predict_image(
        args.img, model, device, scales,
        bands=not args.no_bands, tta=not args.no_tta,
        center_weight=args.center_weight
    )
    print("pred:", deg, "scores:", scores)
