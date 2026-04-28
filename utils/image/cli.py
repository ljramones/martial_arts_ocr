# utils/image/cli.py
"""
CLI for image IO utilities.

Usage:
  python -m utils.image.cli info input.jpg
  python -m utils.image.cli validate input.jpg
  python -m utils.image.cli thumb input.jpg out.jpg --size 320x320 --pad
  python -m utils.image.cli resize input.jpg out.jpg --long 1664
  python -m utils.image.cli crop input.jpg out.jpg --bbox 100,200,600,500 --pad 8
  python -m utils.image.cli ocrprep input.jpg out.png --long 1664
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.image.api import (
    get_image_meta, ensure_valid_then_load, save_image, make_thumb,
)
from utils.image.ops.resize import resize_long_edge
from utils.image.ops.extract import clamp_bbox, region_to_slice
from utils.image.pipelines.ocr_prep import jp_text_prep

def _parse_size(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("size must be like 320x320")

def _parse_bbox(s: str) -> Tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = [int(p) for p in s.split(",")]
        return x1, y1, x2, y2
    except Exception:
        raise argparse.ArgumentTypeError("bbox must be like x1,y1,x2,y2")

def cmd_info(args):
    m = get_image_meta(args.input, decode_for_dtype=False)
    print(f"path: {m.path}")
    print(f"format: {m.fmt}, size: {m.width}x{m.height}, mp: {m.megapixels}")
    print(f"mode: {m.pil_mode}, color_space: {m.color_space}, alpha: {m.has_alpha}")
    print(f"dpi: {m.dpi}, exif_orientation: {m.exif_orientation}, datetime: {m.exif_datetime}")
    print(f"file_size: {m.file_size}")

def cmd_validate(args):
    from utils.image.io.validate import validate_image_file
    rpt = validate_image_file(args.input, require_extension_match=not args.loose)
    print("valid:" if rpt.is_valid else "invalid:", rpt.reason)
    if rpt.is_valid:
        print(f"format: {rpt.format}, size: {rpt.width}x{rpt.height}, mode: {rpt.mode}, dpi: {rpt.dpi}, bytes: {rpt.size_bytes}")

def cmd_thumb(args):
    img = ensure_valid_then_load(args.input, mode="bgr")
    w, h = _parse_size(args.size)
    out = make_thumb(img, (w, h), pad_to_box=args.pad)
    ok = save_image(out, args.output, quality=args.quality, input_mode="bgr")
    if not ok:
        raise SystemExit("failed to save thumbnail")

def cmd_resize(args):
    img = ensure_valid_then_load(args.input, mode="bgr")
    out = resize_long_edge(img, args.long)
    ok = save_image(out, args.output, quality=args.quality, input_mode="bgr")
    if not ok:
        raise SystemExit("failed to save resized image")

def cmd_crop(args):
    from utils.image.regions.core_image import ImageRegion
    img = ensure_valid_then_load(args.input, mode="bgr")
    x1, y1, x2, y2 = _parse_bbox(args.bbox)
    if args.pad > 0:
        x1, y1, x2, y2 = x1 - args.pad, y1 - args.pad, x2 + args.pad, y2 + args.pad
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), w, h)
    ys, xs = region_to_slice((x1, y1, x2, y2))
    crop = img[ys, xs]
    ok = save_image(crop, args.output, quality=args.quality, input_mode="bgr")
    if not ok:
        raise SystemExit("failed to save crop")

def cmd_ocrprep(args):
    img = ensure_valid_then_load(args.input, mode="bgr")
    out = jp_text_prep(img, target_long_edge=args.long)
    ok = save_image(out, args.output, quality=100 if args.png else 95, input_mode="gray")
    if not ok:
        raise SystemExit("failed to save preprocessed image")

def main():
    p = argparse.ArgumentParser(prog="utils.image.cli", description="Image IO utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info"); s.add_argument("input"); s.set_defaults(func=cmd_info)
    s = sub.add_parser("validate"); s.add_argument("input"); s.add_argument("--loose", action="store_true"); s.set_defaults(func=cmd_validate)
    s = sub.add_parser("thumb")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--size", default="320x320")
    s.add_argument("--pad", action="store_true"); s.add_argument("--quality", type=int, default=90)
    s.set_defaults(func=cmd_thumb)

    s = sub.add_parser("resize")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--long", type=int, required=True)
    s.add_argument("--quality", type=int, default=95)
    s.set_defaults(func=cmd_resize)

    s = sub.add_parser("crop")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--bbox", required=True)
    s.add_argument("--pad", type=int, default=0)
    s.add_argument("--quality", type=int, default=95)
    s.set_defaults(func=cmd_crop)

    s = sub.add_parser("ocrprep")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--long", type=int, default=0)
    s.add_argument("--png", action="store_true", help="If saving to PNG, quality is ignored.")
    s.set_defaults(func=cmd_ocrprep)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
