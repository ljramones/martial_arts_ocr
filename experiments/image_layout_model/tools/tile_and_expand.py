# tools/tile_and_expand.py
from __future__ import annotations
from pathlib import Path
import random, shutil
import cv2

ROOT = Path("../dataset")
IMG_DIR = ROOT / "images" / "train"
LBL_DIR = ROOT / "labels" / "train"
OUT_IMG = ROOT / "images" / "train_tiles"
OUT_LBL = ROOT / "labels" / "train_tiles"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

def read_labels(txt: Path, W: int, H: int):
    # YOLO: class xc yc w h (normalized)
    boxes = []
    for ln in txt.read_text().splitlines():
        if not ln.strip():
            continue
        cls, xc, yc, w, h = map(float, ln.split())
        x1 = (xc - w/2) * W
        y1 = (yc - h/2) * H
        x2 = (xc + w/2) * W
        y2 = (yc + h/2) * H
        boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

def write_labels(txt: Path, boxes, W: int, H: int):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    if lines:
        txt.write_text("\n".join(lines) + "\n")

def intersect(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def main(n_crops_per_box=3, pad_range=(0.08, 0.20)):
    ims = sorted(IMG_DIR.glob("*.jpg"))
    total_out = 0
    for imgp in ims:
        lblp = (LBL_DIR / (imgp.stem + ".txt"))
        if not lblp.exists():
            continue
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = read_labels(lblp, W, H)  # list of (cls,x1,y1,x2,y2)

        for idx, (cls, x1, y1, x2, y2) in enumerate(boxes):
            bw, bh = (x2 - x1), (y2 - y1)
            if bw < 10 or bh < 10:
                continue
            for k in range(n_crops_per_box):
                pad = random.uniform(*pad_range)
                dx = int(pad * W); dy = int(pad * H)
                cx1 = max(0, int(x1 - dx)); cy1 = max(0, int(y1 - dy))
                cx2 = min(W, int(x2 + dx)); cy2 = min(H, int(y2 + dy))
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # remap all boxes that fall into this crop
                remap = []
                crop_rect = (cx1, cy1, cx2, cy2)
                for (c2, bx1, by1, bx2, by2) in boxes:
                    inter = intersect((bx1, by1, bx2, by2), crop_rect)
                    if inter:
                        ix1, iy1, ix2, iy2 = inter
                        # shift to crop coords
                        rx1 = ix1 - cx1; ry1 = iy1 - cy1
                        rx2 = ix2 - cx1; ry2 = iy2 - cy1
                        remap.append((c2, rx1, ry1, rx2, ry2))

                # write out
                out_name = f"{imgp.stem}_t{idx}_{k}.jpg"
                out_img = OUT_IMG / out_name
                out_lbl = OUT_LBL / (out_img.stem + ".txt")
                cv2.imwrite(str(out_img), crop)
                ch, cw = crop.shape[:2]
                write_labels(out_lbl, remap, cw, ch)
                total_out += 1
    print(f"Created {total_out} tile images in {OUT_IMG}")

if __name__ == "__main__":
    main()
