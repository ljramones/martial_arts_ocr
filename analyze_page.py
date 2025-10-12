#!/usr/bin/env python3
"""
analyze_page.py — quick CLI harness for layout + preprocessing verification.

Usage:
  python analyze_page.py --input ./all_DFD_Notes_Master_File --out ./debug_output

This script:
  • Loads each image (jpg/png/tif)
  • Runs ImageProcessor.deskew_image() → auto-orient + deskew + Phase-2 textmask
  • Feeds result to LayoutAnalyzer → detects text + image regions
  • Saves annotated overlays for manual inspection
  • Prints per-page + summary stats
"""

from __future__ import annotations

# --- ensure repo root is in sys.path (robust up-walk) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for parent in (ROOT, *ROOT.parents):
    if (parent / "utils" / "image" / "preprocessing" / "facade.py").exists():
        sys.path.insert(0, str(parent))
        break

import argparse
import cv2
import numpy as np
import time
import logging
from glob import glob

# ✅ correct imports
from utils.image.preprocessing.facade import ImageProcessor
from utils.image.layout.analyzer import LayoutAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analyze_page")


# --- draw helper -------------------------------------------------------------
def draw_regions(image: np.ndarray, regions: list, color: tuple) -> np.ndarray:
    """
    Draw rectangles for a list of region dicts:
      {x, y, width, height, region_type?}
    """
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("width", 0), r.get("height", 0)
        label = r.get("region_type", "text")
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to an image, a directory, or a glob pattern (e.g. ./pages/**/*.jpg)")
    ap.add_argument("--out", default="./debug_output",
                    help="Directory for annotated outputs (default: ./debug_output)")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for # of images")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"],
                    help="Accepted extensions")

    # Tuning & switches
    ap.add_argument("--no_mask_images", action="store_true",
                    help="Disable non-text mask for image detectors (still run preprocessing)")
    ap.add_argument("--detectors", nargs="+", default=None,
                    help="Which detectors to run: figure contours variance uniform")
    ap.add_argument("--relax", action="store_true",
                    help="Use relaxed thresholds for early tuning")
    ap.add_argument("--aggressive", action="store_true",
                    help="Run all detectors + looser thresholds for image/diagram recall")
    ap.add_argument("--gray", action="store_true",
                    help="Also save grayscale intermediate image for debugging")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose logging (DEBUG)")

    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    inp = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- robust file discovery ---
    files = []
    if inp.exists():
        if inp.is_file():
            files = [inp]
        else:
            # recursive scan for accepted extensions
            exts = {e.lower() for e in args.exts}
            files = sorted(p for p in inp.rglob("*") if p.suffix.lower() in exts)
    else:
        # allow glob patterns like "images/**/*.jpg"
        exts = {e.lower() for e in args.exts}
        paths = [Path(p) for p in glob(args.input, recursive=True)]
        files = sorted(p for p in paths if p.suffix.lower() in exts and p.exists())

    if args.limit and len(files) > args.limit:
        files = files[: args.limit]

    if not files:
        logger.error(f"No valid input images found for: {args.input}")
        sys.exit(1)

    proc = ImageProcessor()

    # Build config overrides for the analyzer
    overrides = {}
    if args.detectors:
        overrides["enabled_detectors"] = [d.lower() for d in args.detectors]

    if args.relax:
        # Loosen a few thresholds for easier initial detection
        overrides.update({
            "figure_min_area": 8000,             # was 10000
            "contour_min_area": 8000,            # was 15000
            "halo_min_white": 0.80,              # was 0.85
            "contour_edge_density_max": 0.20,    # was 0.15
            "figure_left_bias_xmax": 0.80,       # was 0.70
        })

    if args.aggressive:
        # Aggressive mode: run all detectors + keep contours even if figures found
        overrides.update({
            "enabled_detectors": ["figure", "contours", "variance", "uniform"],
            "contours_always": True,               # add contours even if figures exist
            "contours_require_halo": False,        # allow diagrams without a halo
            "contours_line_mode": True,            # relaxed thin-line acceptance

            # Figure / halo
            "figure_min_area": 6000,
            "halo_min_white": 0.75,                # relax halo gate a bit

            # Contours (diagrams)
            "contour_min_area": 6000,
            "contour_edge_density_min": 0.02,
            "contour_edge_density_max": 0.25,
            "contour_topk": 5,

            # Variance (photos)
            "variance_window": 64,
            "variance_std_min": 12.0,
            "variance_stride_rel": 0.5,            # overlap to boost recall

            # Uniform (shaded panels)
            "uniform_min_area_ratio": 0.015,
            "uniform_max_area_ratio": 0.80,
            "uniform_std_min": 5.0,
            "uniform_std_max": 60.0,

            # NEW: gentler diagram merge + looser final NMS to keep distinct drawings
            "diagram_merge_iou": 0.00,             # require true overlap to merge
            "diagram_merge_gap": 4,                # small neighborhood to avoid cross-merge
            "final_iou_nms": 0.60,                 # keep slightly-overlapping boxes
        })

    analyzer = LayoutAnalyzer(config_override=overrides)

    summary = {"pages": 0, "total_text": 0, "total_image": 0}

    for idx, img_path in enumerate(files, 1):
        logger.info(f"[{idx}/{len(files)}] Processing {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load {img_path}")
            continue

        t0 = time.time()
        # --- preprocessing phase ---
        oriented = proc.deskew_image(img)

        # Optional grayscale dump
        if args.gray:
            gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY) if oriented.ndim == 3 else oriented
            gray_path = out_dir / f"{img_path.stem}_gray.jpg"
            cv2.imwrite(str(gray_path), gray)

        # Mask guard
        mask = None
        if not args.no_mask_images:
            try:
                mask = proc.get_last_nontext_mask()
            except Exception as e:
                logger.warning(f"Non-text mask unavailable: {e}")
                mask = None

        # --- layout analysis ---
        result = analyzer.analyze_page_layout(oriented, nontext_mask=mask)
        stats = result["statistics"]
        summary["pages"] += 1
        summary["total_text"] += stats["num_text_regions"]
        summary["total_image"] += stats["num_image_regions"]

        # --- debug overlay ---
        overlay = draw_regions(oriented, result["text_regions"], (255, 0, 0))
        overlay = draw_regions(overlay, result["image_regions"], (0, 255, 0))

        out_file = out_dir / f"{img_path.stem}_layout.jpg"
        cv2.imwrite(str(out_file), overlay)
        elapsed = (time.time() - t0) * 1000

        # Print mask keep ratio (None if no mask)
        keep = stats.get("nontext_keep_ratio")
        keep_val = -1 if keep is None else keep

        logger.info(
            f"→ Saved {out_file.name:<28} "
            f"text={stats['num_text_regions']:3d}  "
            f"image={stats['num_image_regions']:3d}  "
            f"time={elapsed:7.1f} ms  "
            f"mask_keep={keep_val:.2f}"
        )

    if not summary["pages"]:
        logger.error("No pages processed successfully.")
        sys.exit(1)

    logger.info(
        "\n--- Summary ---\n"
        f"Pages processed:   {summary['pages']:5d}\n"
        f"Total text regions:{summary['total_text']:5d}\n"
        f"Total image regions:{summary['total_image']:5d}\n"
        f"Avg text/page:     {summary['total_text']/summary['pages']:.1f}\n"
        f"Avg image/page:    {summary['total_image']/summary['pages']:.1f}\n"
    )


if __name__ == "__main__":
    main()
