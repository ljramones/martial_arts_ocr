# Image Preprocessing — Overview

This package implements **Step 1 of the OCR pipeline** — converting raw scans or photographs into clean, upright, de-skewed, perspective-corrected, denoised, and binarized pages optimized for OCR (English + Japanese).

The design uses a stable façade (`ImageProcessor`) layered over modular subcomponents.  The façade keeps your public API unchanged while allowing each processing stage (orientation, deskew, denoise, binarize, etc.) to evolve independently.

---

## Directory layout

```
utils/
└─ image/
   ├─ facade.py              # Façade class + pipeline entry points
   ├─ orientation_cnn.py     # CNN (ConvNeXt-Tiny + EffNetV2) orientation predictor
   ├─ orientation.py         # Heuristic fallback (projection & line analysis)
   ├─ geometry.py            # rotate / trim / crop / deskew / perspective / resize
   ├─ binarize.py            # Sauvola + adaptive thresholds + unsharp / normalize
   ├─ denoise.py             # blur metric / preboost / NL-means
   ├─ ocr_osd.py             # Tesseract OSD hint + portrait polarity check
   ├─ debug_io.py            # debug sink for dumps (images + text)
   └─ Overview.md            # this document
```

---

## Phase overview

1. **Orientation (0 / 90 / 180 / 270)**
   • CNN-based classifier (ConvNeXt-Tiny primary, optional ensemble with EffNetV2-S).
   • Fallback: Tesseract OSD hint + heuristic projection/line scorer.
   • Typical runtime: 2–5 ms per page on Apple M-series.

2. **Polarity tie-break (0 vs 180)**
   • Added in 2025-10: compares top/bottom ink density on portrait pages when CNN’s 0°/180° logits are nearly equal.
   • Eliminated final 1 % of orientation errors (now 100 % accuracy on master corpus).

3. **Small-angle de-skew (± 30°)**
   • Median Hough angle of text lines → rotation; expands canvas.

4. **Perspective correction**
   • Detects quadrilateral page outlines → homography warp to rectangular shape.

5. **Denoise**
   • Light NL-Means; adaptive pre-boost for blurry pages to help OSD/chooser.

6. **Binarization**
   • Sauvola threshold + CLAHE contrast + optional unsharp; reinserts grayscale regions for photos/diagrams.

---

## Public API

```python
from utils.image.facade import ImageProcessor

proc = ImageProcessor({"DEBUG_DIR": "data/notebook_outputs/debug_output"})
upright = proc.deskew_image(img)     # phase 1: orientation + small-angle deskew
binary  = proc.preprocess_for_ocr(img)
```

### Debug attributes

| Attribute             | Purpose                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| `_last_phase1_debug`  | summary string: “chosen=90 via CNN[p=0.99]; small=-0.3; blur=162.1; osd=(None,None)” |
| `_last_osd_deg_hint`  | int or None from Tesseract OSD                                                       |
| `_last_choose_scores` | dict → `cnn_scores`, `top1_prob`, `model` etc.                                       |

### Legacy helpers (still exported)

* `preprocess_for_captions_np(np_img)`
* `preprocess_for_fullpage_np(np_img)`
* `preprocess_for_japanese_np(np_img)`

---

## Key configuration knobs (`config.py`)

```python
IMAGE_PROCESSING = {
    # CNN orientation
    'ORIENT_CKPT_CONVNEXT': 'orientation_model/checkpoints/orient_convnext_tiny.pth',
    'ORIENT_CKPT_EFFNET':   'orientation_model/checkpoints/orient_effnetv2s.pth',
    'ORIENT_ENS_MARGIN':    0.55,

    # Disable legacy heuristic if desired
    'DISABLE_HEURISTIC_FALLBACK': True,

    # Polarity tie-break (0 ↔ 180)
    'POLARITY_TIE_MARGIN': 0.10,
    'POLARITY_TIE_FRAC':   0.18,
    'POLARITY_TIE_THRESH': 0.08,

    # Other processing
    'resize_factor': 1.2,
    'deskew': True,
    'denoise': True,
    'SAUVOLA_WINDOW': 25,
    'SAUVOLA_K': 0.2,
}
```

---

## Evaluation results (October 2025)

| Feature         | Accuracy            | Notes                             |
| --------------- | ------------------- | --------------------------------- |
| **Orientation** | **100 % (112/112)** | ConvNeXt + tie-break; zero errors |
| Blur detection  | 91 %                | variance-of-Laplacian             |
| Border trimming | 92 %                | geometry crop correctness         |

All previous heuristic and OSD-only methods topped out ≈ 57 %.
The new CNN pipeline + polarity tie-break raised accuracy from 50 % → 99–100 %.

---

## Developer notes

* **orientation_cnn.py** – wraps ConvNeXt/EffNet models from `orientation_model`; supports ensemble and NumPy → PIL conversion.
* **facade.py** – orchestrates all modules; logs detailed debug notes.
* **orientation.py** – retained for fallback/legacy test harness.
* **evaluate_image_preprocessing.py** – standalone harness reading `master_key.txt`; outputs accuracy report + failure dumps.

---

## Maintenance & extension

| Area         | How to extend                                                     |
| ------------ | ----------------------------------------------------------------- |
| Orientation  | plug new checkpoints via config; tie-break auto-applies.          |
| Denoise      | add new filters under `denoise.py` and toggle via config.         |
| Thresholding | experiment with larger Sauvola windows (35–45) for 600 dpi scans. |
| Debugging    | enable `DEBUG_DIR` to inspect intermediate images.                |
| Languages    | edit `ocr_osd.py` (default `eng+jpn`) for alternate OSD hints.    |

---

## Quick test harness

```bash
python -m evaluate_image_preprocessing \
  --images ./data/corpora/donn_draeger/dfd_notes_master/original \
  --key ./master_key.txt
```

**Output sample**

```
--- Evaluation Report ---
Orientation: 100.00 % (112/112)
Blur Detection: 91.07 %
Border Trimming: 91.96 %
```

---

### Summary

* CNN orientation = ConvNeXt-Tiny (+ EffNet ensemble)
* Added portrait polarity tie-break → perfect accuracy on test set
* Heuristic retained for safety/debug
* Fully EXIF-safe JPEG pipeline
* Ready for integration with OCR/NGPA stages

*(Last updated 2025-10-10 — ConvNeXt CNN orientation + polarity tie-break integration)*

---
