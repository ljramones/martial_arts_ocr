#!/usr/bin/env python3
"""
Phase 1 Orientation + Deskew Test Harness
Processes every image in ./all_DFD_Notes_Master_File and writes side-by-side comparisons
to ./all_DFD_Notes_Master_File_results, with annotated captions and a CSV summary.
"""

import os
import csv
import cv2
import numpy as np
import logging
from utils.image_preprocessing import ImageProcessor  # adjust import if needed
from utils.image_io import load_image

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

INPUT_DIR = "all_DFD_Notes_Master_File"
OUTPUT_DIR = "all_DFD_Notes_Master_File_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

proc = ImageProcessor()

def annotate(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a readable caption on the image."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    org = (24, 40)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def coarse_orientation_deg(img: np.ndarray) -> int:
    """
    Very lightweight coarse orientation (0/90/180/270) using Hough lines.
    Display-only; actual correction happens in ImageProcessor.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    e = cv2.Canny(g, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(e, 1, np.pi/180, 200)
    if lines is None or len(lines) == 0:
        return 0
    thetas = (lines[:, 0, 1] * 180.0 / np.pi) - 90.0
    med = float(np.median(thetas))
    candidates = [-90, 0, 90, 180]
    best = min(candidates, key=lambda d: abs((med - d + 180) % 360 - 180))
    if best == -90: return 270
    if best == 180: return 180
    return best

summary_rows = [["filename", "orig_deg", "after_deg",
                 "width_before", "height_before", "width_after", "height_after",
                 "phase1_note"]]

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        continue

    path = os.path.join(INPUT_DIR, fname)
    img = load_image(path)
    if img is None:
        print(f"Skipping {fname} (unreadable)")
        continue

    print(f"Processing {fname} ...")

    # Before orientation estimate (display-only)
    orig_deg = coarse_orientation_deg(img)

    # Phase 1 processing (OSD + fallback + deskew inside ImageProcessor.deskew_image)
    result = proc.deskew_image(img)
    note = getattr(proc, "_last_phase1_debug", "")

    # After orientation estimate (display-only)
    after_deg = coarse_orientation_deg(result)

    # Side-by-side with captions
    left  = annotate(img,    f"Original  ({orig_deg}°)")
    right = annotate(result, f"Phase 1   ({after_deg}°)")

    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    h = max(h1, h2)
    canvas = np.ones((h, w1 + w2 + 20, 3), dtype=np.uint8) * 255
    canvas[0:h1, 0:w1] = left
    canvas[0:h2, w1 + 20:w1 + 20 + w2] = right

    out_path = os.path.join(OUTPUT_DIR, f"phase1_{fname}")
    cv2.imwrite(out_path, canvas)
    print(f" → saved {out_path}")
    if note:
        print(f"   debug: {note}")

    summary_rows.append([fname, orig_deg, after_deg, w1, h1, w2, h2, note])

# Write CSV summary
with open(os.path.join(OUTPUT_DIR, "phase1_summary.csv"), "w", newline="") as f:
    csv.writer(f).writerows(summary_rows)

print(f"\n✅ Phase 1 test complete. Results in ./{OUTPUT_DIR}")
print(f"   - Side-by-side images: {OUTPUT_DIR}/phase1_<filename>.jpg")
print(f"   - Summary CSV:         {OUTPUT_DIR}/phase1_summary.csv")
