# 🖼️ `utils/image` — Complete Overview

> **Purpose:**
> Unified image-processing and layout-analysis framework for OCR, diagram extraction, and document understanding.
> Combines deep-learning orientation models, classical CV pipelines, modular detectors, and consistent region representations.

---

## 📁 Directory Map

```
utils/image/
├── __init__.py
├── api.py                     # Facade API (read, write, resize, extract, regions)
├── cli.py                     # Command-line interface (python -m utils.image.cli)
├── HowtoProcessImages.md       # Developer how-to guide
├── improvementplan.md          # Future enhancement roadmap
│
├── io/                        # File/byte I/O and metadata
├── ops/                       # Core image operations (resize, tone, extract, thumb)
├── regions/                   # Geometry, grouping, filters, text fixups
├── layout/                    # Page layout analyzer (detectors + filters + post + utils)
├── preprocessing/              # Orientation, deskew, denoise, binarization, masks
├── pipelines/                 # Composite OCR prep pipelines
│
├── shared_utils.py             # Shared helpers
└── types.py                    # Type aliases and global dataclasses
```

---

## 🧱 1. IO Layer (`image/io/`)

Handles all **file and byte-level image I/O** safely and consistently.

| File          | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| `read.py`     | Load image from disk or bytes; EXIF-aware orientation                      |
| `write.py`    | Save or encode (JPEG/PNG/TIFF/WebP) with quality control and atomic writes |
| `validate.py` | Check existence, extension, and verify readability via Pillow              |
| `meta.py`     | Fast metadata extraction (format, DPI, EXIF, ICC profile)                  |
| `image_io.py` | Legacy compatibility shim (re-exports `read`/`write` APIs)                 |
| `Overview.md` | Local documentation for IO subpackage                                      |

---

## ⚙️ 2. OPS Layer (`image/ops/`)

Implements pixel-level transformations and utilities shared across modules.

| File         | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `resize.py`  | Aspect-aware resizing, fit/fill/letterbox, thumbnails          |
| `tone.py`    | Brightness/contrast, gamma, CLAHE, white balance, unsharp mask |
| `extract.py` | Crop rectangular/polygonal regions from arrays                 |
| `thumb.py`   | Generate thumbnails (array → file/bytes)                       |

---

## 🧭 3. Preprocessing Model (`image/preprocessing/`)

End-to-end **image cleanup and orientation correction** before OCR.

| File                 | Function                                                          |
| -------------------- | ----------------------------------------------------------------- |
| `facade.py`          | `ImageProcessor` façade (orientation → deskew → binarize)         |
| `orientation_cnn.py` | ConvNeXt / EfficientNet orientation classifier (0°/90°/180°/270°) |
| `orientation.py`     | Heuristic orientation fallback & polarity tie-breaker             |
| `ocr_osd.py`         | Tesseract OSD integration (angle hints & tie sniff)               |
| `geometry.py`        | Rotation, trim, crop, deskew, perspective correction              |
| `denoise.py`         | NL-means, blur metric, pre-boost blurry scans                     |
| `binarize.py`        | Sauvola/adaptive/Otsu binarization with CLAHE & unsharp           |
| `textmask.py`        | Build non-text mask via black-hat + MSER join                     |
| `debug_io.py`        | Safe debug dump writer (images/text per step)                     |
| `Overview.md`        | Local overview of preprocessing pipeline                          |

---

## 🧩 4. Layout Analyzer (`image/layout/`)

Detects **figures, diagrams, photos, and backgrounds** within pages.

| Directory/File                                | Description                                     |
| --------------------------------------------- | ----------------------------------------------- |
| `analyzer.py`                                 | Main `LayoutAnalyzer` orchestrator              |
| `image_layout.py` / `_legacy_image_layout.py` | Legacy shims for backward compatibility         |
| `refactoringPlan.md`                          | Design roadmap for layout refactor              |
| `Overview.md`                                 | Local architecture doc for the layout subsystem |

### ➕ Detectors (`layout/detectors/`)

Each detector subclasses `BaseDetector` and outputs `List[ImageRegion]`.

| Detector             | File                        | Core Heuristic                       |
| -------------------- | --------------------------- | ------------------------------------ |
| `FigureDetector`     | `figure.py`                 | Morph close + halo isolation         |
| `ContourDetector`    | `contours.py`               | Canny + dilate + Hough lines         |
| `VarianceDetector`   | `variance.py`               | Local variance + gradient smoothness |
| `UniformDetector`    | `uniform.py`                | Morph close + std-dev range          |
| `YOLOFigureDetector` | `yolo_figure.py`            | Ultralytics YOLO inference           |
| `overview.md`        | Docs for detector internals |                                      |

### ➕ Filters (`layout/filters/`)

| File             | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| `text_filter.py` | MSER text detection + text-like region filtering |

### ➕ Post (`layout/post/`)

| File       | Purpose                                 |
| ---------- | --------------------------------------- |
| `merge.py` | IoU + gap-based merging & deduplication |

### ➕ Utils (`layout/utils/`)

| File             | Purpose                          |
| ---------------- | -------------------------------- |
| `halo.py`        | Halo brightness validation       |
| `masks.py`       | Apply & normalize non-text masks |
| `projections.py` | Text-line regularity metrics     |
| `cc_metrics.py`  | Connected-component statistics   |

---

## 🧮 5. Regions Layer (`image/regions/`)

Defines **core data structures** and **geometry logic** shared across the stack.

| File                                | Role                                                  |
| ----------------------------------- | ----------------------------------------------------- |
| `core_types.py`                     | `ImageRegion`, `ImageInfo`, `_Box`, and bbox helpers  |
| `convert.py`                        | `_Box` ↔ tuple ↔ `ImageRegion` adapters               |
| `geometry.py`                       | bbox math: IoU/IoA, expand/shrink, aspect, clamp      |
| `grouping.py`                       | Merge small boxes → lines; proximity clustering       |
| `layout.py`                         | Column split + reading-order sorting                  |
| `filters.py`                        | Size/aspect/area filters; NMS/dedup/sorters           |
| `text_fixups.py`                    | Post-OCR cleanup (hyphens, wraps, quotes, whitespace) |
| `core_image.py`, `image_regions.py` | Legacy compatibility stubs                            |
| `Overview.md`                       | Local documentation for regions subsystem             |

---

## 🧰 6. Pipelines (`image/pipelines/`)

| File          | Purpose                                                     |
| ------------- | ----------------------------------------------------------- |
| `ocr_prep.py` | Composite OCR preprocessing: grayscale → binarize → denoise |

---

## ⚙️ 7. Shared Utilities

| File              | Description                                     |
| ----------------- | ----------------------------------------------- |
| `shared_utils.py` | Generic helper functions used across packages   |
| `types.py`        | Common type aliases (e.g., `BBox`, `ColorMode`) |

---

## 🔌 API & CLI

### `api.py`

Unified interface to load, process, and save images:

```python
from utils.image import api as img

arr  = img.ensure_valid_then_load("page.jpg")
meta = img.get_image_meta("page.jpg")
thumb_ok = img.quick_thumb_to_file("page.jpg", "thumbs/page_320.jpg")
```

### `cli.py`

```bash
python -m utils.image.cli info input.jpg
python -m utils.image.cli thumb input.jpg out.jpg --size 320x320
python -m utils.image.cli ocrprep input.jpg out.png --long 1664 --png
```

---

## 🔄 Typical Flow

```
        ┌──────────────┐
        │ io.read/load │  →  ndarray (BGR/GRAY)
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ preprocessing │  →  upright + binarized page
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ layout.analyzer │ →  figures, diagrams, photos
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ regions.*    │ →  canonical ImageRegion objects
        └──────────────┘
```

---

## 🧪 Testing & Validation

* **Unit tests**: geometry math, text fixups, tone ops (fast, deterministic).
* **Integration**: façade pipelines (orientation, Sauvola binarization).
* **Layout**: verify detector IoU merges and text filter rejection.
* **CLI smoke tests**: ensure all subcommands complete successfully.

---

## 💡 Design Guidelines

* Keep **`core_types.py`** and geometry logic dependency-free.
* Maintain separation between **IO**, **OPS**, **REGIONS**, **LAYOUT**, and **PREPROCESSING**.
* All image functions must accept and return **NumPy arrays** (no side effects).
* Use **atomic saves** and **debug sinks** for safe batch processing.
* New detectors → subclass `BaseDetector`, output `ImageRegion`.
* Always log, never crash, on per-image errors.

---

## ✅ Summary

| Layer             | Purpose                         | Key Output                  |
| ----------------- | ------------------------------- | --------------------------- |
| **IO**            | Safe image read/write/meta      | np.ndarray + `ImageInfo`    |
| **OPS**           | Tone & resize utilities         | transformed arrays          |
| **PREPROCESSING** | Orientation → deskew → binarize | clean OCR-ready page        |
| **LAYOUT**        | Detect visual regions           | `List[ImageRegion]`         |
| **REGIONS**       | Shared geometry & text cleanup  | canonical region data       |
| **PIPELINES**     | End-to-end OCR preparation      | binarized + denoised output |

---
