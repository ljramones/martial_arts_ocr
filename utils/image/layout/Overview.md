# 🧭 Layout Analyzer Module Overview

## Purpose

This package provides a modular system for **detecting and classifying visual regions** on document images (e.g., scanned pages, manuscripts, or research papers).
It identifies **text**, **figures**, **diagrams**, **photos**, and **uniform regions**, then merges and filters them into structured, analyzable metadata.

The refactor separates logic into **detectors**, **filters**, and **utilities** to eliminate circular imports, improve testability, and make adding new detection algorithms straightforward.

---

## Package Structure

```
utils/image/layout/
├── __init__.py            # Re-exports LayoutAnalyzer
├── analyzer.py            # Thin orchestrator for all detectors/filters
│
├── detectors/             # Independent region detectors
│   ├── __init__.py        # Defines BaseDetector interface
│   ├── figure.py          # Detects embedded figures/photos
│   ├── contours.py        # Detects line-art/diagrams
│   ├── variance.py        # Detects photo-like high-variance regions
│   └── uniform.py         # Detects low-variance blocks (backgrounds)
│
├── filters/
│   └── text_filter.py     # Filters out text-like regions; MSER-based text detector
│
├── post/
│   └── merge.py           # Overlap merging and deduplication
│
└── utils/
    ├── halo.py            # Bright-halo check for image isolation
    ├── masks.py           # Apply non-text masks to gray images
    ├── projections.py     # Regularity heuristics (text line detection)
    └── cc_metrics.py      # Connected-component statistics
```

---

## Core Concepts

### 1. **LayoutAnalyzer (Orchestrator)**

Located in `analyzer.py`, it’s the entrypoint for layout detection:

* Converts inputs to `uint8` grayscale.
* Applies optional **non-text masks** (to ignore known text zones).
* Runs a configurable sequence of **detectors**:

  1. `FigureDetector` – large, bordered figures.
  2. `ContourDetector` – diagrams, line art.
  3. `VarianceDetector` – photo-like textures.
  4. `UniformDetector` – shaded or low-variance blocks.
* Postprocesses detections using:

  * `remove_overlaps()` (IoU-based pruning)
  * `TextRegionFilter.filter()` (reject text-like false positives)
* Computes statistics like coverage ratios and page size.

---

### 2. **Detectors**

Each detector implements `BaseDetector.detect(gray: np.ndarray) -> List[ImageRegion]`.

| Detector           | Purpose                                                   | Core Heuristic                   |
| ------------------ | --------------------------------------------------------- | -------------------------------- |
| `FigureDetector`   | Finds images with clear white halos and strong boundaries | Morph close + halo check         |
| `ContourDetector`  | Finds diagrams or charts via edge density                 | Canny + Hough + CC stats         |
| `VarianceDetector` | Finds natural photos or textured regions                  | Local variance thresholding      |
| `UniformDetector`  | Finds large flat color blocks (backgrounds, panels)       | Morph close + std deviation gate |

All detectors use local intensity variance, aspect ratio, and connected-component checks to ensure geometric plausibility.

---

### 3. **Filters**

* **TextRegionFilter** serves dual purposes:

  * Detect text blocks using **MSER** (`detect_mser()`).
  * Filter text-like candidates among image regions (`filter()`).
* It applies connected-component analysis and projection regularity (peak spacing) to reject patterns typical of text.

---

### 4. **Post-processing**

* **merge.py** removes overlapping detections via simple IoU-based deduplication.
* Resulting regions are represented as `ImageRegion` objects, each exposing geometry, type, and confidence.

---

### 5. **Utilities**

| Utility          | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| `halo.py`        | Validates white halos around figures (`halo_ok`) |
| `masks.py`       | Normalizes and applies binary/boolean masks      |
| `projections.py` | Measures text-line regularity in projections     |
| `cc_metrics.py`  | Reports connected-component area statistics      |

---

## Why This Refactor Matters

| Problem (Pre-Refactor)                  | Solution                                            |
| --------------------------------------- | --------------------------------------------------- |
| 800+ lines monolithic `image_layout.py` | Split into small, testable modules                  |
| Tight coupling & circular imports       | Defined `BaseDetector` and clear orchestration flow |
| Difficult to extend                     | Plug-in detector model (`enabled_detectors` config) |
| Hard to test individual parts           | Each detector/filter is unit-testable               |
| Redundant preprocessing                 | Centralized `_to_gray_u8()` and mask utilities      |

---

## Configuration Keys (Excerpt)

```python
LAYOUT_DETECTION = {
    "enabled_detectors": ["figure", "contours", "variance", "uniform"],
    "halo_ring": 4,
    "halo_min_white": 0.85,
    "figure_margin": 20,
    "uniform_close_kernel": 15,
    "uniform_min_area_ratio": 0.03,
    "uniform_max_area_ratio": 0.50,
    "uniform_std_min": 10.0,
    "uniform_std_max": 100.0,
}
```

---

## Testing Guidance

**Unit tests**

* Create synthetic grayscale images with simple geometric shapes.
* Validate that each detector returns expected bounding boxes.
* Ensure text filter rejects dense repeated patterns.

**Integration tests**

* Run `LayoutAnalyzer.analyze_page_layout()` on a mixed-content document.
* Validate that `num_text_regions + num_image_regions > 0`.
* Compare area ratios and ensure no overlapping boxes remain.

---

## Summary

The `layout` package transforms low-level image heuristics into a maintainable, extensible architecture.
It provides a solid foundation for hybrid pipelines combining classical CV with OCR or deep-learning models later on.

---
