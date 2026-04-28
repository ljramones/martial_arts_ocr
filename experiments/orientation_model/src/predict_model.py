# src/predict_model.py
from __future__ import annotations
import argparse
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms as T

# -----------------------------
# Utility layers / device
# -----------------------------
class To3Channels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure 3 x H x W
        return x if x.size(0) == 3 else x.expand(3, -1, -1)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model factory (arch-aware)
# -----------------------------
def _build_model_for_arch(arch: str, num_classes: int = 4) -> nn.Module:
    """
    Build the correct backbone for the checkpoint.
    Supported arch values in ckpt['arch']:
      - 'mobilenet_v3_small'  (default)
      - 'convnext_tiny'
      - 'effnetv2_s'  (a.k.a. efficientnet_v2_s)
    """
    a = (arch or "mobilenet_v3_small").lower()
    if a in ("convnext_tiny", "convnext-tiny", "convnextt"):
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if a in ("effnetv2_s", "efficientnet_v2_s", "efficientnetv2s", "effnetv2s"):
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    # default: MobileNetV3-Small
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

# -----------------------------
# Public API expected by eval_test.py
# -----------------------------
def load_model(ckpt_path: str, device: torch.device | None = None):
    """
    Returns (model, device, tfm, img_size, idx_to_deg)
    NOTE: We always decode indices 0..3 as degrees [0,90,180,270] (fixed label order used during training).
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)

    img_size = int(ckpt.get("img_size", 384))
    arch = ckpt.get("arch", "mobilenet_v3_small")

    model = _build_model_for_arch(arch, num_classes=4)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    tfm = T.Compose([
        T.ToTensor(),
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        To3Channels(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Return fixed mapping to satisfy eval_test unpacking (even though we don't use it to decode)
    idx_to_deg_fixed = {0: 0, 1: 90, 2: 180, 3: 270}
    return model, device, tfm, img_size, idx_to_deg_fixed

@torch.inference_mode()
def predict_image(
    img_path_or_pil: str,
    model: nn.Module,
    device: torch.device,
    tfm: T.Compose,
    size: int,
    idx_to_deg: dict[int, int],   # kept for eval_test signature compatibility; not used for decoding
    bands: bool = True,
):
    """
    Single-head prediction with simple center + bands voting.
    Decoding uses FIXED degree order: idx 0..3 -> [0, 90, 180, 270].
    """
    INV = {0: 0, 1: 90, 2: 180, 3: 270}  # fixed label order used during training

    # NEW: accept PIL.Image or path
    if isinstance(img_path_or_pil, Image.Image):
        img = img_path_or_pil.convert("RGB")
    else:
        img = Image.open(img_path_or_pil).convert("RGB")
    def logits_of(pil_img: Image.Image):
        x = tfm(pil_img).unsqueeze(0).to(device)
        return model(x).float()  # [1,4]

    # Center view
    L = logits_of(img)

    if bands:
        # Simple border strips (20%) to pick up footer/margins; averaged with center
        w, h = img.size
        bw = max(8, int(0.20 * w))
        bh = max(8, int(0.20 * h))
        left   = img.crop((0,     0,     bw,   h))
        right  = img.crop((w-bw,  0,     w,    h))
        top    = img.crop((0,     0,     w,    bh))
        bottom = img.crop((0,     h-bh,  w,    h))
        for v in (left, right, top, bottom):
            L += logits_of(v)
        L /= 5.0

    probs = torch.softmax(L, dim=1).squeeze(0).cpu().numpy()
    idx = int(probs.argmax())
    deg = INV[idx]
    scores = {INV[i]: float(p) for i, p in enumerate(probs.tolist())}
    return deg, scores

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--no_bands", action="store_true")
    args = ap.parse_args()

    model, device, tfm, sz, _ = load_model(args.ckpt)
    deg, scores = predict_image(args.img, model, device, tfm, sz, idx_to_deg={0:0,1:90,2:180,3:270}, bands=not args.no_bands)
    print("pred:", deg, "scores:", scores)
