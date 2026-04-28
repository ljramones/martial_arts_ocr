from pathlib import Path
import cv2, numpy as np

IN = Path("dataset/images/train_tiles")
OUT = Path("dataset/images/train_tiles")
LBL = Path("dataset/labels/train_tiles")

for imgp in sorted(IN.glob("*_micro_*.jpg")):
    im = cv2.imread(str(imgp))
    if im is None: continue

    # 1) light gaussian blur
    g = cv2.GaussianBlur(im, (3,3), 0.8)
    g_name = imgp.stem + "_g.jpg"
    cv2.imwrite(str(OUT / g_name), g)
    (LBL / (g_name.replace(".jpg",".txt"))).write_text("0 0.5 0.5 1 1\n")

    # 2) contrast up (alpha) + small brightness (beta)
    c = cv2.convertScaleAbs(im, alpha=1.15, beta=10)
    c_name = imgp.stem + "_c.jpg"
    cv2.imwrite(str(OUT / c_name), c)
    (LBL / (c_name.replace(".jpg",".txt"))).write_text("0 0.5 0.5 1 1\n")
print("Blur/contrast variants added.")
