#!/usr/bin/env python3
# prepare_dataset.py — EXIF-safe, native-size, JPEG-by-default

from __future__ import annotations
import argparse, random, shutil
from pathlib import Path
from typing import Dict, Tuple, List

from PIL import Image, ImageOps
import numpy as np
import cv2
from tqdm import tqdm

SPLIT_NAMES = ("train", "validation", "test")
CLASS_DIRS  = ("0", "90", "180", "270")

def parse_master_key(key_path: Path) -> Dict[str, int]:
    key: Dict[str, int] = {}
    with key_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2: continue
            fname, angle_s = parts[0], parts[1]
            try:
                ang = int(angle_s) % 360
            except ValueError:
                print(f"Warning: bad angle for {fname!r}; skipping"); continue
            if ang % 90 != 0:
                print(f"Warning: angle not multiple of 90 for {fname!r}; skipping"); continue
            key[fname] = ang
    return key

def clean_and_make_dirs(out_root: Path) -> None:
    if out_root.exists():
        print(f"Cleaning old data from {out_root} …")
        shutil.rmtree(out_root)
    print("Creating dataset directories …")
    for split in SPLIT_NAMES:
        for c in CLASS_DIRS:
            (out_root / split / c).mkdir(parents=True, exist_ok=True)

def pil_rotate_90cw(img: Image.Image, angle_cw: int) -> Image.Image:
    a = angle_cw % 360
    if a == 0:   return img
    if a == 90:  return img.transpose(Image.ROTATE_270)  # 90 CW == 270 CCW
    if a == 180: return img.transpose(Image.ROTATE_180)
    if a == 270: return img.transpose(Image.ROTATE_90)
    return img

def to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:     # L
        return arr
    if arr.shape[2] == 4: # RGBA -> BGRA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def write_image(path: Path, arr: np.ndarray, ext: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        # cv2: encode JPEG (quality 95, 4:4:4)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95,
                  int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), 444]
        return bool(cv2.imwrite(str(path), arr, params))
    elif ext == ".png":
        return bool(cv2.imwrite(str(path), arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 6]))
    elif ext in (".tif", ".tiff"):
        # cv2 TIFF writing is limited; PNG/JPEG are safer. If you need TIFF+LZW, use Pillow instead.
        return bool(cv2.imwrite(str(path), arr))
    else:
        # default: try via cv2
        return bool(cv2.imwrite(str(path), arr))

def summarize_counts(out_root: Path) -> None:
    print("\nSanity check (files per split/class):")
    for split in SPLIT_NAMES:
        print(f"  {split}/")
        for c in CLASS_DIRS:
            n = len(list((out_root / split / c).glob("*")))
            print(f"    {c:>3}: {n}")

def build_dataset(
    source_images: Path,
    master_key: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float],
    seed: int,
    grayscale: bool,
    ext: str,
) -> None:
    # Resolve + checks
    source_images = source_images.resolve()
    master_key    = master_key.resolve()
    output_dir    = output_dir.resolve()
    if not master_key.exists():
        raise SystemExit(f"Error: --master_key not found at {master_key}")
    if not source_images.exists():
        raise SystemExit(f"Error: --source_images not found at {source_images}")

    key = parse_master_key(master_key)
    files = sorted(key.keys())
    rng = random.Random(seed); rng.shuffle(files)

    n = len(files)
    t, v, u = split_ratios
    n_train = int(n * t)
    n_val   = int(n * v)
    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test.")

    clean_and_make_dirs(output_dir)
    total = 0

    def process_list(split_name: str, fnames: List[str]) -> int:
        written = 0
        for fname in tqdm(fnames, ncols=80, desc=f"  {split_name.capitalize()}"):
            src = source_images / fname
            if not src.exists():
                print(f"  ! missing: {src}"); continue

            try:
                # Load & EXIF-normalize so pixels match what you saw
                img0 = Image.open(src)
                img0 = ImageOps.exif_transpose(img0)
            except Exception:
                print(f"  ! unreadable: {src}"); continue

            if grayscale and img0.mode != "L":
                img0 = img0.convert("L")

            # Rotate to canonical upright by NEGATING the labeled angle (CW)
            corrective_cw = (-int(key.get(fname, 0))) % 360
            upright = pil_rotate_90cw(img0, corrective_cw)

            base = Path(fname).stem
            for lbl in (0,90,180,270):
                sample = pil_rotate_90cw(upright, lbl)
                arr = to_cv(sample)  # cv2 ndarray (BGR/GRAY/BGRA)
                out_path = output_dir / split_name / str(lbl) / f"{base}_rot{lbl}{ext}"
                if not write_image(out_path, arr, ext):
                    print(f"  ! failed to write: {out_path}")
                else:
                    written += 1
        return written

    total += process_list("train", train_files)
    total += process_list("validation", val_files)
    total += process_list("test", test_files)

    print(f"\n✅ Done. Wrote {total} files to {output_dir}")
    summarize_counts(output_dir)

def parse_ratios(s: str) -> Tuple[float,float,float]:
    t, v, u = [float(x.strip()) for x in s.split(",")]
    if abs((t+v+u) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    return t, v, u

def main():
    ap = argparse.ArgumentParser(description="Build EXIF-safe 0/90/180/270 dataset (native size, JPEG by default).")
    ap.add_argument("--source_images", type=Path, required=True)
    ap.add_argument("--master_key",   type=Path, required=True)
    ap.add_argument("--output_dir",   type=Path, default=Path("./data"))
    ap.add_argument("--split",        type=str,  default="0.7,0.15,0.15")
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--grayscale",    action="store_true")
    ap.add_argument("--ext",          type=str,  default=".jpg",
                    help="Output extension: .jpg/.jpeg/.png/.tif/.tiff (default: .jpg)")
    args = ap.parse_args()

    ext = args.ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in (".jpg",".jpeg",".png",".tif",".tiff"):
        raise SystemExit("Error: --ext must be one of .jpg/.jpeg/.png/.tif/.tiff")

    ratios = parse_ratios(args.split)
    build_dataset(
        source_images=args.source_images,
        master_key=args.master_key,
        output_dir=args.output_dir,
        split_ratios=ratios,
        seed=args.seed,
        grayscale=args.grayscale,
        ext=ext,
    )

if __name__ == "__main__":
    main()

