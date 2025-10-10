Here’s a ready-to-drop **`utils/image/overview.md`** you can paste into your repo.

---

# Image Preprocessing — Overview

This package implements **Step 1** of the OCR pipeline: turning raw scans/photos into pages that are upright, de-skewed, perspective-corrected, denoised, and binarized for high OCR yield (English + Japanese).

It’s organized as a thin façade (`ImageProcessor`) over focused modules. The façade keeps your public API stable while the internals stay modular and testable.

## Directory layout

```
utils/
└─ image/
   ├─ image_preprocessing.py   # Facade (public API) + legacy helpers
   ├─ orientation.py           # 0/90/180/270 chooser + line metrics
   ├─ geometry.py              # rotate/trim/crop/deskew/perspective/resize
   ├─ binarize.py              # Sauvola/adaptive thresholds + unsharp/normalize
   ├─ denoise.py               # blur metric, preboost, NL-means
   ├─ ocr_osd.py               # Tesseract: OSD hint + 0° vs 180° sniff
   ├─ debug_io.py              # debug sink (optional image/text dumps)
   └─ types.py                 # light type aliases (optional)
```

> If you later subpackage this into `image/preprocess/`, this document still applies; only import paths change.

---

## What the pipeline does (high level)

1. **Auto-orientation (0/90/180/270)**

   * Uses projection peakiness of text lines and Hough line density.
   * Adds a portrait/landscape prior and (optionally) a Tesseract OSD bonus.

2. **0° vs 180° tie-break**

   * Tiny OCR sniff over a central band (eng+jpn TSV) chooses upright when portrait is ambiguous.

3. **Sanity +90° check**

   * If baselines look more horizontal after +90°, keep that.

4. **Small-angle de-skew (±30°)**

   * Median Hough line angle; canvas-expanding rotation.

5. **Perspective correction (keystone fix)**

   * If a dominant quadrilateral page is detected, warp to a rectangle.

6. **Denoise (light)**

   * Fast NL-Means (gray or BGR), tuned to preserve typewritten strokes.

7. **Binarization (Sauvola)**

   * Robust local thresholding; CLAHE pre-contrast; optional unsharp mask.

---

## Public API (façade)

```python
from utils.image.image_preprocessing import ImageProcessor

proc = ImageProcessor({"debug_dir": "debug_output"})   # optional
upright = proc.deskew_image(img)                       # Phase-1: orientation + small-angle deskew
binary  = proc.preprocess_for_ocr(img)                 # full pipeline → binary uint8 for OCR

# Debug strings for harness / CSV:
proc._last_phase1_debug      # str like: "chosen=90 via Proj; small=0.84; blur=162.1; osd=(None,None)"
proc._last_osd_deg_hint      # Optional[int] in {0,90,180,270}
proc._last_choose_scores     # dict with "proj"/"horiz"/"combo" per angle
```

### Back-compat helpers (legacy imports)

The façade also exposes:

* `preprocess_for_captions_np(np_img)`
* `preprocess_for_fullpage_np(np_img)`
* `preprocess_for_japanese_np(np_img)`

These mirror your older helpers for callers that haven’t migrated yet.

---

## Module by module

### `image_preprocessing.py` (façade)

* Owns **configuration** and **debug sink** wiring.
* Implements:

  * `deskew_image(image, max_angle=30.0)` → oriented/deskewed image (keeps channels).
  * `preprocess_for_ocr(image, …)` → final binary image.
* Exposes debug notes for your harness (`_last_phase1_debug`, `_last_choose_scores`, `_last_osd_deg_hint`).
* Provides a **compat wrapper**: `proc._choose_coarse_orientation(img)` for tests.

**Config keys read (optional):**

* `IMAGE_PROCESSING.debug_dir` — folder for chooser/perspective dumps.
* `IMAGE_PROCESSING.tesseract_bin` — path to `tesseract` if not on PATH.
* `IMAGE_PROCESSING.resize_factor` — default `1.2` up-scale for typewriter dots.
* `IMAGE_PROCESSING.deskew` — enable phase-1 (`True`).
* `IMAGE_PROCESSING.denoise` — enable NL-Means (`True`).

---

### `orientation.py`

* **Goal:** choose among {0, 90, 180, 270}.
* Steps:

  1. Trim black borders; crop dominant page region.
  2. Compute **projection peakiness** (rows vs columns) from a quick binarization.
  3. Compute **horizontal line score** using Hough (closeness to 0° + line density).
  4. Add **portrait/landscape prior**; add **+0.30** bonus if matches OSD hint.
  5. 0 vs 180 tie-break via line angles if portrait scores are close.

Returns `(best_deg, scores_dict)` where `scores_dict` includes `proj`, `horiz`, and `combo` per angle.

---

### `geometry.py`

* **rotate_deg(image, deg)** — fast 0/90/180/270 rotations.
* **auto_trim_black_borders(gray)** — remove thick scanner borders.
* **crop_page_for_scoring(gray)** — Otsu+close/open to isolate page rectangle.
* **sanity_plus_90_if_better(gray)** — rotate +90 if baselines improve.
* **deskew_small_angle(image, max_angle)** — median Hough angle; expands canvas; returns `(image, meta)`.
* **apply_perspective_correction(gray)** — find quad contour and warp; safe no-op on failure.
* **resize(image, factor)** — cubic/area interpolation depending on scale.

---

### `binarize.py`

* **sauvola(gray, window=25, k=0.2)** — `float32` math; CLAHE pre-contrast; returns binary `uint8`.
* **adaptive_gaussian / adaptive_mean** — OpenCV adaptive thresholds.
* **otsu(gray, invert=False)** — global Otsu.
* **unsharp(image, strength=1.5, sigma=1.0)** — edge emphasis (use before final binarization or re-threshold).
* **normalize(image)** — ensure `uint8 [0..255]`.

---

### `denoise.py`

* **blur_var(gray)** — variance of Laplacian (focus proxy).
* **preboost_blurry(gray, blur_threshold=180.0)** — CLAHE + unsharp if blurry (helps OSD/chooser).
* **nl_means(image, strength='light')** — fast NL-Means for gray or BGR.

---

### `ocr_osd.py`

* **find_tesseract_bin()** — locate `tesseract` on common paths.
* **guess_tessdata_dir()** — locate tessdata (prefers repo `./tessdata`).
* **osd_rotate_deg(gray, tess_bin)** — Tesseract `--psm 0` → `(deg, conf)` or `(None, None)`.
* **upright_0_vs_180(img, tess_bin)** — central-band OCR sniff (eng+jpn TSV) to decide 0 vs 180.

All calls are defensive; return safe fallbacks if Tesseract is missing.

---

### `debug_io.py`

* **DebugSink(dirpath, prefix="", limit=None)**

  * `write(tag, mat)` — saves images (auto gray/BGR handling).
  * `text(tag, string)` — saves small text notes.
  * Thread-safe, sequential filenames, silent on errors.

---

### `types.py` (optional)

Light type aliases:

* `GrayU8`, `BGRU8`, `ImgU8`, `SizeHW`, `PointXY`
* `DebugWriter` protocol with `.write(tag, mat)` for mocks/tests.

---

## Typical usage

```python
from utils.image.image_preprocessing import ImageProcessor
from utils.image.image_io import load_image

img = load_image("scan001.tif")
proc = ImageProcessor({"debug_dir": "debug_output"})
phase1 = proc.deskew_image(img)
binary = proc.preprocess_for_ocr(phase1)  # or directly on img
```

---

## Test harness (orientation / deskew)

Use `phase1_test.py` (your harness) to produce side-by-sides and a CSV:

```
python phase1_test.py \
  --input all_DFD_Notes_Master_File \
  --output stuff_results \
  --workers 6 \
  --max-side 1600 \
  --sus-thresh 0.12 \
  --debug-dir debug_output
```

CSV fields include timing, blur score, OSD hint, score margin, and chooser scores.

---

## Performance & quality tips

* Install Tesseract (improves 0° vs 180° on portrait pages).
* Adjust **`resize_factor`** in config if your scans are very small/large.
* Tune **`sauvola(window,k)`** for different DPI:

  * ~300 dpi → `window=25–31`
  * 600 dpi → `window=35–45`
* To reduce artifacts on photos (uneven lighting), try `correct_illumination()` (currently in façade as optional helper if needed).

---

## Troubleshooting

* **“cannot import name …”**
  Clear caches:
  `find utils -name "__pycache__" -type d -prune -exec rm -rf {} +`
  `find utils -name "*.pyc" -delete`

* **OSD returns (None, None)**
  Tesseract not found or tessdata missing. Set `IMAGE_PROCESSING.tesseract_bin` or install via Homebrew/Apt; ensure `osd.traineddata` is present.

* **Outputs too large**
  Use harness `--max-side` to scale previews; in production, keep originals full-res for OCR.

* **Over- or under-binarized pages**
  Try `sauvola(window=31, k=0.25)` or fall back to `adaptive_gaussian`.

---

## Extending

* New denoiser? Add `denoise2.py`, wire from façade behind a config toggle.
* New threshold? Add `sauvola_fast.py` (integral images) and A/B via config.
* Language-specific tweaks? Inject different OCR sniff languages in `ocr_osd.upright_0_vs_180`.

---

*Maintainer note:* Keep `utils/image/__init__.py` **light** and prefer **relative imports** inside this package to avoid circulars.
