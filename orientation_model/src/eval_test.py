# src/eval_test.py
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

def _import_predictor(twohead: bool):
    if twohead:
        from src.predict_twohead import load_model, predict_image
        return load_model, predict_image, True
    else:
        from src.predict_model import load_model, predict_image
        return load_model, predict_image, False

def _top2(d: dict[int,float]):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:2]

def main(ckpt: str, test_dir: str, twohead: bool, bands: bool, tta: bool, scales: str, verbose: int):
    load_model, predict_image, is_twohead = _import_predictor(twohead)

    if is_twohead:
        model, device, ckpt_size = load_model(ckpt)
        # parse scales
        if scales.strip():
            scale_list = [int(s.strip()) for s in scales.split(",") if s.strip()]
        else:
            scale_list = [ckpt_size, max(ckpt_size, 448)]
    else:
        model, device, tfm, ckpt_size, idx2deg = load_model(ckpt)

    root = Path(test_dir)
    deg_dirs = [p for p in root.iterdir() if p.is_dir()]
    deg_dirs.sort(key=lambda p: int(p.name))

    total = correct = 0
    per = {0:[0,0], 90:[0,0], 180:[0,0], 270:[0,0]}
    conf = defaultdict(lambda: defaultdict(int))

    samples_printed = {0:0, 90:0, 180:0, 270:0}

    for d in deg_dirs:
        deg_true = int(d.name)
        for img_p in d.iterdir():
            if not img_p.is_file():
                continue

            if is_twohead:
                pred_deg, scores = predict_image(
                    str(img_p), model, device, scale_list,
                    bands=bands, tta=tta
                )
            else:
                pred_deg, scores = predict_image(
                    str(img_p), model, device, tfm, ckpt_size, idx2deg, bands=bands
                )

            total += 1
            per[deg_true][1] += 1
            conf[deg_true][pred_deg] += 1
            if pred_deg == deg_true:
                correct += 1
                per[deg_true][0] += 1

            if verbose and samples_printed[deg_true] < verbose:
                top2 = _top2(scores)
                print(f"[{deg_true}] {img_p.name} -> pred {pred_deg}, top2={top2}")
                samples_printed[deg_true] += 1

    acc = correct / max(1, total)
    print(f"Test accuracy: {correct}/{total} = {acc*100:.2f}%")
    for k in (0, 90, 180, 270):
        c, n = per[k]
        print(f"{k:>4}: {c}/{n} ({(c/max(1,n))*100:.1f}%)")

    print("\nConfusion (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{c:>5}" for c in (0,90,180,270))
    print(header)
    for t in (0,90,180,270):
        row = [conf[t][p] for p in (0,90,180,270)]
        print(f"{t:>5}: " + "  ".join(f"{n:>5}" for n in row))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--twohead", action="store_true")
    ap.add_argument("--no_bands", action="store_true")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--scales", default="")
    ap.add_argument("--verbose", type=int, default=0, help="Print N sample preds per class")
    args = ap.parse_args()
    main(
        args.ckpt, args.test, args.twohead,
        bands=not args.no_bands, tta=not args.no_tta, scales=args.scales,
        verbose=args.verbose
    )
