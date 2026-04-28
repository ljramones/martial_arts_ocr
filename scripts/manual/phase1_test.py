#!/usr/bin/env python3
"""
Phase 1 Orientation + Deskew Test Harness (outliers)
Processes every image in an input folder and writes side-by-side comparisons,
a CSV summary, and a suspects/ folder with flagged cases.

Usage:
  python scripts/manual/phase1_test.py --input data/corpora/donn_draeger/dfd_notes_master/original --output stuff_results
"""

from __future__ import annotations
import re, csv, cv2, time, math, argparse, logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.image.preprocessing.facade import ImageProcessor
from utils.image.io.image_io import load_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("phase1_harness")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/corpora/donn_draeger/dfd_notes_master/original", help="Input directory of images")
    ap.add_argument("--output", default="stuff_results", help="Output directory for results")
    ap.add_argument("--max-side", type=int, default=1800, help="Max side for preview canvas (per panel); 0=no limit")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--sus-thresh", type=float, default=0.12, help="Score margin below which a case is suspect")
    ap.add_argument("--debug-dir", default="", help="If set, pass through to ImageProcessor to dump chooser crops")
    return ap.parse_args()

def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    org = (24, 40)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def preview_downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    return cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)

def chooser_deg_for_label(proc: ImageProcessor, img: np.ndarray) -> int:
    """Use the processor's own chooser for display (best of {0,90,180,270})."""
    try:
        deg, _scores = proc._choose_coarse_orientation(img)  # returns (deg, scores)
        return int(deg)
    except Exception:
        return 0

def parse_osd_from_note(note: str) -> tuple[str, str]:
    m = re.search(r"osd=\(([^,]*),([^)]*)\)", note.lower())
    if not m:
        return ("", "")
    return (m.group(1).strip(), m.group(2).strip())

def score_margin_from_note(note: str) -> float | None:
    # expects something like "... 0:0.123, 90:1.234, 180:0.456, 270:0.789 ..."
    vals = [float(x) for x in re.findall(r":([0-9]+\.[0-9]+)", note)]
    if len(vals) < 2:
        return None
    vals.sort(reverse=True)
    return vals[0] - vals[1]

def is_suspect(note: str, before: int, after: int, margin_thresh: float) -> bool:
    n = (note or "").lower()
    if "no_lines" in n:
        return True
    m = re.search(r"conf=([0-9.]+)", n)
    if m:
        try:
            if float(m.group(1)) < 3.0:
                return True
        except ValueError:
            pass
    margin = score_margin_from_note(note)
    if margin is not None and margin < margin_thresh:
        return True
    if before in (90, 180, 270) and after == 0:
        return True
    return False

def process_one(proc: ImageProcessor, in_path: Path, out_dir: Path, sus_dir: Path, max_side: int, sus_thresh: float):
    row = None
    try:
        img = load_image(str(in_path))
        if img is None:
            raise RuntimeError("unreadable image")
        h0, w0 = img.shape[:2]

        t0 = time.perf_counter()
        orig_deg = chooser_deg_for_label(proc, img)
        result = proc.deskew_image(img)  # phase 1
        note   = getattr(proc, "_last_phase1_debug", "")
        choose_scores = getattr(proc, "_last_choose_scores", {}) or {}
        blur_match = re.search(r"blur=([0-9.]+)", note or "")
        blur = float(blur_match.group(1)) if blur_match else math.nan
        osd_deg, osd_conf = parse_osd_from_note(note)
        after_deg = chooser_deg_for_label(proc, result)
        t1 = time.perf_counter()

        left  = annotate(preview_downscale(img,     max_side), f"Original  ({orig_deg}°)")
        right = annotate(preview_downscale(result,  max_side), f"Phase 1   ({after_deg}°)")

        h1,w1 = left.shape[:2]; h2,w2 = right.shape[:2]
        h = max(h1,h2)
        canvas = np.ones((h, w1+w2+20, 3), dtype=np.uint8) * 255
        canvas[0:h1, 0:w1] = left
        canvas[0:h2, w1+20:w1+20+w2] = right

        out_path = out_dir / f"phase1_{in_path.name}"
        cv2.imwrite(str(out_path), canvas)

        suspect = is_suspect(note, orig_deg, after_deg, sus_thresh)
        if suspect:
            sus_img = sus_dir / f"phase1_{in_path.name}"
            sus_txt = sus_dir / f"phase1_{in_path.stem}.txt"
            cv2.imwrite(str(sus_img), canvas)
            sus_txt.write_text(note or "")

        margin = score_margin_from_note(note)
        row = [
            in_path.name, orig_deg, after_deg,
            w0, h0, w2, h2,
            f"{(t1-t0):.3f}",
            f"{blur:.1f}" if not math.isnan(blur) else "",
            osd_deg, osd_conf,
            note.replace("\n", " ").strip(),
            "yes" if suspect else "no",
            f"{margin:.3f}" if margin is not None else "",
            repr(choose_scores)[:500],  # keep CSV readable
        ]
        log.debug(f"Saved {out_path} ({t1-t0:.2f}s){' [SUSPECT]' if suspect else ''}")
        return row, suspect
    except Exception as e:
        log.error(f"{in_path.name}: {e}")
        # minimal row so the CSV reflects the failure
        row = [in_path.name, "", "", "", "", "", "", "", "", "", "", f"error:{e}", "yes", "", ""]
        return row, True

def main():
    args = parse_args()
    in_dir  = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    sus_dir = out_dir / "suspects"; sus_dir.mkdir(parents=True, exist_ok=True)

    cfg_override = {}
    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        cfg_override["debug_dir"] = args.debug_dir
    proc = ImageProcessor(config_override=cfg_override)

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        log.warning(f"No images found in {in_dir}")
        return

    header = ["filename","orig_deg","after_deg",
              "width_before","height_before","width_after","height_after",
              "seconds","blur","osd_deg","osd_conf",
              "phase1_note","suspect","score_margin","chooser_scores"]

    rows = [header]
    sus_count = 0

    log.info(f"Processing {len(files)} files from {in_dir} → {out_dir} (workers={args.workers})")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_one, proc, p, out_dir, sus_dir, args.max_side, args.sus_thresh) for p in files]
        for fut in as_completed(futs):
            row, sus = fut.result()
            rows.append(row)
            if sus: sus_count += 1

    csv_path = out_dir / "phase1_summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    tot = len(files)
    log.info(f"\n✅ Phase 1 test complete. Results in ./{out_dir}")
    log.info(f"   - Side-by-side images: {out_dir}/phase1_<filename>.<ext>")
    log.info(f"   - Summary CSV:         {csv_path}")
    log.info(f"   - Suspects folder:     {sus_dir}/  ({sus_count}/{tot} flagged)")

if __name__ == "__main__":
    main()
