# make_micro_crops.py
from __future__ import annotations
from pathlib import Path
import cv2

# --- config ------------------------------------------------------------
# Split to read from (use "train" if the example page is in train)
SPLIT = "val"   # or "train"

# List of manual crops: (filename, x1, y1, x2, y2) in *page pixel coords*
# Add a few high-value panels here (examples shown):
CROPS = [
    ("IMG_3397.jpg",  60,  80,  430,  460),   # small shuriken grid (example)
    ("IMG_3397.jpg", 470,  80,  760,  420),   # right-side panel (example)
    ("IMG_3352.jpg",  90, 120,  900,  650),   # top kusarigama fig (example)
    ("IMG_3352.jpg",  90, 710,  900, 1250),   # human figure panel (example)
    ("IMG_3292.jpg",  80,  60, 1040,  560),   # big top panel (example)
]

# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
IN_DIR  = ROOT / "dataset" / "images" / SPLIT
OUTI    = ROOT / "dataset" / "images" / "train_tiles"
OUTL    = ROOT / "dataset" / "labels" / "train_tiles"
OUTI.mkdir(parents=True, exist_ok=True)
OUTL.mkdir(parents=True, exist_ok=True)

def write_full_box_label(txt_path: Path, w: int, h: int):
    # Full-image box (class 0): xc=0.5,yc=0.5,w=1,h=1 (normalized)
    txt_path.write_text("0 0.5 0.5 1.0 1.0\n")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    made = 0
    for name, x1, y1, x2, y2 in CROPS:
        src = IN_DIR / name
        if not src.exists():
            print(f"[WARN] Missing: {src}")
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"[WARN] imread failed: {src}")
            continue

        H, W = img.shape[:2]
        # clamp coords to image
        x1c, y1c = clamp(x1, 0, W-1), clamp(y1, 0, H-1)
        x2c, y2c = clamp(x2, 0, W),   clamp(y2, 0, H)
        if x2c <= x1c or y2c <= y1c:
            print(f"[WARN] Bad crop for {name}: ({x1},{y1},{x2},{y2})")
            continue

        crop = img[y1c:y2c, x1c:x2c]
        out_name = f"{Path(name).stem}_micro_{x1c}_{y1c}_{x2c}_{y2c}.jpg"
        out_img  = OUTI / out_name
        out_lbl  = OUTL / out_name.replace(".jpg", ".txt")

        ok = cv2.imwrite(str(out_img), crop)
        if not ok:
            print(f"[WARN] write failed: {out_img}")
            continue

        h, w = crop.shape[:2]
        if w < 5 or h < 5:
            print(f"[WARN] tiny crop skipped: {out_img}")
            out_img.unlink(missing_ok=True)
            continue

        write_full_box_label(out_lbl, w, h)
        made += 1
        print(f"[OK] {out_img}  ({w}x{h})")

    print(f"Created {made} micro-crops in {OUTI}")

if __name__ == "__main__":
    main()
