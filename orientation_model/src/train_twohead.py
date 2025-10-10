# src/train_twohead.py
from __future__ import annotations
import argparse
from pathlib import Path
import time
import math
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

# Robust import when run as a module or directly
try:
    from src.model_twohead import TwoHeadMobileNet  # python -m src.train_twohead
except ModuleNotFoundError:
    from model_twohead import TwoHeadMobileNet      # python train_twohead.py

# ---------- utils ----------

class To3Channels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.size(0) == 3 else x.expand(3, -1, -1)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms(img_size: int = 384, train: bool = True) -> T.Compose:
    ops = [T.ToTensor()]
    if train:
        # Orientation-safe augs that help side/footers → better 90/270
        ops += [
            T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.2, 0.8))], p=0.25),
            T.RandomApply([T.RandomAffine(degrees=0, shear=(-4, 4))], p=0.35),
            T.RandomApply([T.Pad(padding=12, fill=255, padding_mode="constant")], p=0.40),
            T.RandomApply([T.RandomResizedCrop(img_size, scale=(0.9, 1.0),
                                               ratio=(0.95, 1.05), antialias=True)], p=0.35),
        ]
    ops += [
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        To3Channels(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return T.Compose(ops)

def build_loaders(data_root: str, img_size=384, batch_size=16, num_workers=0):
    root = Path(data_root)
    tf_train = build_transforms(img_size, train=True)
    tf_val   = build_transforms(img_size, train=False)

    ds_train = datasets.ImageFolder(root / "train", transform=tf_train)
    ds_val   = datasets.ImageFolder(root / "validation", transform=tf_val)

    device = get_device()
    is_mps = (device.type == "mps")
    # macOS/MPS: workers=0 and pin_memory=False for stability
    nw = 0 if is_mps else num_workers
    pin = False if is_mps else True

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pin)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    # True mapping from ImageFolder (index -> degree)
    idx_to_deg = {idx: int(name) for name, idx in ds_train.class_to_idx.items()}
    print("Class mapping (idx->deg):", idx_to_deg)
    return ds_train, ds_val, dl_train, dl_val, device, idx_to_deg

# ---------- label helpers ----------

def deg_to_family_polarity(deg: int):
    """
    Returns (family, polarity):
      family:   0 portrait (0/180), 1 landscape (90/270)
      polarity: 0 up (0/90),        1 down     (180/270)
    """
    family = 0 if deg in (0, 180) else 1
    polarity = 0 if deg in (0, 90) else 1
    return family, polarity

# ---------- training ----------

def train(
    data_dir: str,
    out_ckpt: str = "checkpoints/orient_twohead_mnv3s.pth",
    img_size: int = 384,
    epochs: int = 12,
    batch_size: int = 16,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    weight_decay: float = 1e-4,
    polarity_bias: float = 1.0,         # e.g., 1.15 if many 0↔180 or 90↔270 flips
    warmup_epochs: int = 2,             # head-only epochs (1–3 is typical)
    early_stop_patience: int = 0,       # 0 = disabled
    save_last: bool = True,             # also save a final snapshot
    seed: int = 42,
):
    set_seeds(seed)
    t0 = time.time()

    ckpt_path = Path(out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, dl_train, dl_val, device, idx_to_deg = build_loaders(
        data_dir, img_size=img_size, batch_size=batch_size
    )

    model = TwoHeadMobileNet(pretrained=True).to(device)

    # Warmup: train heads only
    for p in model.features.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(
        [{"params": model.head_family.parameters()},
         {"params": model.head_polarity.parameters()}],
        lr=lr_head, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, warmup_epochs))
    ce = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    epochs_no_improve = 0

    def run_epoch(train_mode=True):
        model.train(mode=train_mode)
        total, correct4, loss_sum = 0, 0, 0.0
        loader = dl_train if train_mode else dl_val

        for xb, y_idx in loader:
            xb = xb.to(device)
            y_idx = y_idx.to(device)

            # map y_idx -> degree -> (family, polarity)
            y_deg = torch.tensor([idx_to_deg[int(i)] for i in y_idx.tolist()], device=device)
            y_family = torch.tensor([deg_to_family_polarity(int(d))[0] for d in y_deg.tolist()], device=device)
            y_polar  = torch.tensor([deg_to_family_polarity(int(d))[1] for d in y_deg.tolist()], device=device)

            if train_mode:
                opt.zero_grad(set_to_none=True)

            fam_logits, pol_logits = model(xb)

            # optional bias on polarity (helps within-family flips)
            loss_fam = ce(fam_logits, y_family)
            loss_pol = ce(pol_logits, y_polar) * polarity_bias
            loss = loss_fam + loss_pol

            if train_mode:
                loss.backward()
                opt.step()

            # compute 4-way acc for logging using LOG-PROB combiner
            four_lp = TwoHeadMobileNet.combine_to_fourway(fam_logits, pol_logits)  # [B,4] log-probs
            pred4 = four_lp.argmax(1)

            # labels in [0,1,2,3] aligned to [0,90,180,270]
            deg_to_idx = {0: 0, 90: 1, 180: 2, 270: 3}
            y4 = torch.tensor([deg_to_idx[int(d)] for d in y_deg.tolist()], device=device)

            bs = xb.size(0)
            loss_sum += float(loss.item()) * bs
            correct4 += int((pred4 == y4).sum().item())
            total    += bs

        return loss_sum / max(1,total), correct4 / max(1,total)

    # ---------- head-only warmup ----------
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
                "idx_to_deg": idx_to_deg,
                "arch": "twohead",
            }
            torch.save(best_state, ckpt_path)
            print(f"  ↳ saved {ckpt_path} ({best_acc*100:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stop_patience and epochs_no_improve >= early_stop_patience:
                print("Early stop triggered during warmup.")
                done = time.time() - t0
                print(f"Best val_acc: {best_acc*100:.2f}%  (ckpt: {ckpt_path})  time={done:.1f}s")
                if save_last:
                    torch.save({
                        "model": model.state_dict(),
                        "img_size": img_size,
                        "idx_to_deg": idx_to_deg,
                        "arch": "twohead",
                    }, ckpt_path.with_suffix(".last.pth"))
                return

    # ---------- unfreeze all & full training ----------
    for p in model.features.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr_full, weight_decay=weight_decay)
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
                "idx_to_deg": idx_to_deg,
                "arch": "twohead",
            }
            torch.save(best_state, ckpt_path)
            print(f"  ↳ saved {ckpt_path} ({best_acc*100:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stop_patience and epochs_no_improve >= early_stop_patience:
                print("Early stop triggered.")
                break

    done = time.time() - t0
    print(f"Best val_acc: {best_acc*100:.2f}%  (ckpt: {ckpt_path})  time={done:.1f}s")

    if save_last:
        torch.save({
            "model": model.state_dict(),
            "img_size": img_size,
            "idx_to_deg": idx_to_deg,
            "arch": "twohead",
        }, ckpt_path.with_suffix(".last.pth"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out",  default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "orient_twohead_mnv3s.pth"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=1e-4)
    ap.add_argument("--polarity_bias", type=float, default=1.0)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--early_stop_patience", type=int, default=0)
    ap.add_argument("--save_last", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(
        args.data, out_ckpt=args.out, img_size=args.img_size, epochs=args.epochs,
        batch_size=args.batch, lr_head=args.lr_head, lr_full=args.lr_full,
        weight_decay=1e-4, polarity_bias=args.polarity_bias,
        warmup_epochs=args.warmup_epochs, early_stop_patience=args.early_stop_patience,
        save_last=args.save_last, seed=args.seed
    )
