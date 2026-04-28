# src/train_convnext_tiny.py
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T, models

# ---------- Utils ----------
class To3Channels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.size(0) == 3 else x.expand(3, -1, -1)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(img_size: int = 384, train: bool = True) -> T.Compose:
    ops = [T.ToTensor()]
    if train:
        # label-safe, orientation-safe augs
        ops += [
            T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.2, 0.8))], p=0.20),
            T.RandomApply([T.RandomAffine(degrees=0, shear=(-4, 4))], p=0.35),
            T.RandomApply([T.Pad(padding=12, fill=255, padding_mode="constant")], p=0.35),
            T.RandomApply([T.RandomResizedCrop(img_size, scale=(0.9, 1.0),
                                               ratio=(0.95, 1.05), antialias=True)], p=0.35),
        ]
    ops += [
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        To3Channels(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
    return T.Compose(ops)

def build_loaders(data_root: str, img_size=384, batch_size=16, num_workers=0):
    root = Path(data_root)
    tf_train = build_transforms(img_size, train=True)
    tf_val   = build_transforms(img_size, train=False)

    ds_train = datasets.ImageFolder(root / "train", transform=tf_train)
    ds_val   = datasets.ImageFolder(root / "validation", transform=tf_val)

    device = get_device()
    is_mps = device.type == "mps"
    nw = 0 if is_mps else num_workers
    pin = False if is_mps else True

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    idx_to_deg_train = {idx: int(name) for name, idx in ds_train.class_to_idx.items()}
    idx_to_deg_val   = {idx: int(name) for name, idx in ds_val.class_to_idx.items()}
    print("Class mapping (train idx->deg):", idx_to_deg_train)
    print("Class mapping (valid idx->deg):", idx_to_deg_val)
    return ds_train, ds_val, dl_train, dl_val, device, idx_to_deg_train, idx_to_deg_val

# ---------- Mixup ----------
class Mixup:
    def __init__(self, alpha: float = 0.1, p: float = 0.5):
        self.alpha, self.p = alpha, p
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if torch.rand(1).item() > self.p:  # no-op
            return x, y, 1.0, None
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        xmix = lam * x + (1 - lam) * x[idx]
        return xmix, (y, y[idx]), lam, idx

def ce_with_mixup(logits, y, lam, idx, label_smoothing=0.1, num_classes=4):
    # label smoothing CE (manual)
    logp = torch.log_softmax(logits, dim=1)
    if isinstance(y, tuple):
        y1, y2 = y
        y1oh = torch.zeros_like(logp).scatter_(1, y1.unsqueeze(1), 1.0)
        y2oh = torch.zeros_like(logp).scatter_(1, y2.unsqueeze(1), 1.0)
        yoh  = lam * y1oh + (1-lam) * y2oh
    else:
        yoh = torch.zeros_like(logp).scatter_(1, y.unsqueeze(1), 1.0)
    yoh = yoh * (1 - label_smoothing) + label_smoothing / num_classes
    return -(yoh * logp).sum(dim=1).mean()

# ---------- Training ----------
def build_model(num_classes=4, pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    m = models.convnext_tiny(weights=weights)
    in_feat = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_feat, num_classes)
    return m

def train(
    data_dir: str,
    out_ckpt: str = "checkpoints/orient_convnext_tiny.pth",
    img_size: int = 384,
    epochs: int = 40,
    batch_size: int = 16,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    weight_decay: float = 1e-4,
    mixup_alpha: float = 0.1,
    mixup_p: float = 0.5,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 2,
):
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    _, _, dl_train, dl_val, device, idx2deg_train, idx2deg_val = build_loaders(data_dir, img_size, batch_size)

    torch.set_float32_matmul_precision("high")

    model = build_model(pretrained=True).to(device)

    # Head-only warmup
    for p in model.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, warmup_epochs))
    mixup = Mixup(alpha=mixup_alpha, p=mixup_p)

    best_acc, best_state = 0.0, None

    def run_epoch(train_mode=True):
        model.train(mode=train_mode)
        loader = dl_train if train_mode else dl_val
        idx2deg = idx2deg_train if train_mode else idx2deg_val

        total = correct = 0
        loss_sum = 0.0
        for xb, y_idx in loader:
            xb, y_idx = xb.to(device), y_idx.to(device)
            # remap to [0,1,2,3] indices consistent with degree order
            deg = torch.tensor([idx2deg[int(i)] for i in y_idx.tolist()], device=device)
            deg_to_idx = {0:0, 90:1, 180:2, 270:3}
            y = torch.tensor([deg_to_idx[int(d)] for d in deg.tolist()], device=device)

            if train_mode:
                opt.zero_grad(set_to_none=True)
                x_mix, y_mix, lam, idx_perm = mixup(xb, y)
                logits = model(x_mix)
                loss = ce_with_mixup(logits, y_mix, lam, idx_perm, label_smoothing=label_smoothing)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    logits = model(xb)
                    loss = ce_with_mixup(logits, y, 1.0, None, label_smoothing=label_smoothing)

            pred = logits.argmax(1)
            correct += int((pred == (y if not isinstance(y, tuple) else y[0])).sum().item())
            total += xb.size(0)
            loss_sum += float(loss.item()) * xb.size(0)

        return loss_sum / max(1,total), correct / max(1,total)

    # Warmup
    for ep in range(1, warmup_epochs+1):
        tr_loss, tr_acc = run_epoch(True)
        va_loss, va_acc = run_epoch(False)
        sched.step()
        print(f"[Head] Ep {ep:02d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.2f}% | val_acc {va_acc*100:.2f}%")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {"model": model.state_dict(), "img_size": img_size,
                          "idx_to_deg": idx2deg_train, "arch": "convnext_tiny"}
            torch.save(best_state, out_ckpt)
            print(f"  ↳ saved {out_ckpt} ({best_acc*100:.2f}%)")

    # Unfreeze all
    for p in model.features.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr_full, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))

    for ep in range(warmup_epochs+1, epochs+1):
        tr_loss, tr_acc = run_epoch(True)
        va_loss, va_acc = run_epoch(False)
        sched.step()
        print(f"[Full] Ep {ep:02d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.2f}% | val_acc {va_acc*100:.2f}%")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {"model": model.state_dict(), "img_size": img_size,
                          "idx_to_deg": idx2deg_train, "arch": "convnext_tiny"}
            torch.save(best_state, out_ckpt)
            print(f"  ↳ saved {out_ckpt} ({best_acc*100:.2f}%)")

    print(f"Best val_acc: {best_acc*100:.2f}% (ckpt: {out_ckpt})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out",  default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "orient_convnext_tiny.pth"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=1e-4)
    ap.add_argument("--mixup_alpha", type=float, default=0.1)
    args = ap.parse_args()
    train(args.data, out_ckpt=args.out, img_size=args.img_size, epochs=args.epochs,
          batch_size=args.batch, lr_head=args.lr_head, lr_full=args.lr_full,
          mixup_alpha=args.mixup_alpha)
