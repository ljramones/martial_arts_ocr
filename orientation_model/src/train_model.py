
# src/train_model.py
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

# --- Utilities ---

class To3Channels(torch.nn.Module):
    """Ensure tensor is 3xHxW without using a pickling-unsafe lambda."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.size(0) == 3 else x.expand(3, -1, -1)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(img_size: int = 384, train: bool = True) -> T.Compose:
    ops = [T.ToTensor()]
    if train:
        ops += [
            T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.2, 0.8))], p=0.25),
            T.RandomApply(
                [T.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), antialias=True)],
                p=0.35,
            ),
        ]
    ops += [
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        To3Channels(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return T.Compose(ops)

def build_loaders(data_root: str, img_size=384, batch_size=16, num_workers=0):
    """
    Expects directory layout:
      data_root/
        train/{0,90,180,270}/
        validation/{0,90,180,270}/
    """
    root = Path(data_root)
    tf_train = build_transforms(img_size, train=True)
    tf_val   = build_transforms(img_size, train=False)

    ds_train = datasets.ImageFolder(root / "train", transform=tf_train)
    ds_val   = datasets.ImageFolder(root / "validation", transform=tf_val)

    device = get_device()
    is_mps = (device.type == "mps")
    # macOS/MPS: use workers=0 and pin_memory=False for stability
    nw = 0 if is_mps else num_workers
    pin = False if is_mps else True

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=nw, pin_memory=pin)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=nw, pin_memory=pin)
    return ds_train, ds_val, dl_train, dl_val, device

def build_model(num_classes=4) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_feat = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_feat, num_classes)
    return m

# --- Training ---

def train(
    data_dir: str,
    out_ckpt: str = "checkpoints/orient_mnv3s.pth",
    img_size: int = 384,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    ckpt_path = Path(out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds_train, ds_val, dl_train, dl_val, device = build_loaders(
        data_dir, img_size=img_size, batch_size=batch_size
    )

    # Real index->degree mapping from folder names (ImageFolder sorts alphabetically)
    # e.g., {'0':0, '180':1, '270':2, '90':3} depending on filesystem ordering
    idx_to_deg = {idx: int(name) for name, idx in ds_train.class_to_idx.items()}
    print("Class mapping (idx->deg):", idx_to_deg)

    model = build_model().to(device)

    # Warmup: train classifier head only
    for p in model.features.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(3, epochs // 3))
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None

    def run_epoch(train_mode=True):
        model.train(mode=train_mode)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train_mode):
            for xb, yb in (dl_train if train_mode else dl_val):
                xb, yb = xb.to(device), yb.to(device)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                if train_mode:
                    loss.backward()
                    opt.step()
                loss_sum += float(loss.item()) * xb.size(0)
                pred = logits.argmax(1)
                correct += int((pred == yb).sum().item())
                total += xb.size(0)
        return loss_sum / max(1, total), correct / max(1, total)

    # Head-only epochs
    warmup_epochs = min(3, epochs)
    for ep in range(1, warmup_epochs + 1):
        tr_loss, tr_acc = run_epoch(train_mode=True)
        va_loss, va_acc = run_epoch(train_mode=False)
        sched.step()
        print(f"[Head] Ep {ep:02d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.2f}% | val_acc {va_acc*100:.2f}%")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                "model": model.state_dict(),
                "img_size": img_size,
                "idx_to_deg": idx_to_deg,   # save true mapping
            }
            torch.save(best_state, ckpt_path)
            print(f"  ↳ saved {ckpt_path} ({best_acc*100:.2f}%)")

    # Unfreeze entire model
    for p in model.features.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))

    for ep in range(warmup_epochs + 1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_mode=True)
        va_loss, va_acc = run_epoch(train_mode=False)
        sched.step()
        print(f"[Full] Ep {ep:02d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc*100:.2f}% | val_acc {va_acc*100:.2f}%")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                "model": model.state_dict(),
                "img_size": img_size,
                "idx_to_deg": idx_to_deg,   # save true mapping
            }
            torch.save(best_state, ckpt_path)
            print(f"  ↳ saved {ckpt_path} ({best_acc*100:.2f}%)")

    print(f"Best val_acc: {best_acc*100:.2f}%  (ckpt: {ckpt_path})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out",  default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "orient_mnv3s.pth"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args.data, out_ckpt=args.out, img_size=args.img_size, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
