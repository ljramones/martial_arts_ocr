# 📐 `layout/detectors` Package — Overview

## Purpose

This directory contains **modular region detectors** used by the layout-analysis layer of your Image Utils system.
Each detector isolates a visual type — figure, diagram, photo, or uniform background — from grayscale page images.
They all inherit from a simple abstract `BaseDetector` and return standardized `ImageRegion` objects.

---

## Module Map

```
utils/image/layout/detectors/
├── __init__.py      # Defines BaseDetector interface
├── figure.py        # FigureDetector – bordered/haloed figures
├── contours.py      # ContourDetector – line-art / diagrams
├── variance.py      # VarianceDetector – photo-like textures
├── uniform.py       # UniformDetector – low-variance shaded blocks
├── yolo_figure.py   # YOLOFigureDetector – DL-based figure finder
└── Overview.md      # Detailed architectural documentation
```

---

## 1️⃣ Base Interface — `__init__.py`

```python
class BaseDetector:
    def detect(self, gray: np.ndarray) -> List[ImageRegion]:
        raise NotImplementedError
```

All detectors subclass this and must return a list of `ImageRegion` objects (from `utils.image.regions`).

---

## 2️⃣ Detector Summary

| Detector               | File             | Purpose                                        | Key Technique                                       | Region Type |
| ---------------------- | ---------------- | ---------------------------------------------- | --------------------------------------------------- | ----------- |
| **FigureDetector**     | `figure.py`      | Finds large bordered figures with bright halos | Morphological closing + halo check                  | `"figure"`  |
| **ContourDetector**    | `contours.py`    | Finds diagrams and line-art                    | Canny → Dilate → Contour + Hough + CC stats         | `"diagram"` |
| **VarianceDetector**   | `variance.py`    | Finds photo-like textured regions              | Sliding-window local variance + gradient smoothness | `"photo"`   |
| **UniformDetector**    | `uniform.py`     | Finds large low-variance panels/backgrounds    | Morph close + Otsu + std-deviation range            | `"image"`   |
| **YOLOFigureDetector** | `yolo_figure.py` | Deep-learning alternative for figure detection | Ultralytics YOLO inference                          | `"figure"`  |

---

## 3️⃣ Typical Flow in `LayoutAnalyzer`

1. Convert page to grayscale (`uint8`).
2. Optionally apply non-text mask (to skip text zones).
3. Sequentially call each enabled detector.
4. Merge overlapping boxes and filter text-like patterns.
5. Return unified list of `ImageRegion` objects with metadata.

```python
regions = []
for det in enabled_detectors:
    regions.extend(det.detect(gray))
regions = merge_overlaps(regions)
regions = text_filter.filter(regions)
```

---

## 4️⃣ Configuration Highlights

Each detector uses keys from your global `LAYOUT_DETECTION` config:

```python
"enabled_detectors": ["figure", "contours", "variance", "uniform"],
"figure_min_area": 10000,
"contour_min_area": 15000,
"uniform_close_kernel": 15,
"variance_window_min": 128,
"yolo_model_path": "models/figure_detector.pt",
```

---

## 5️⃣ Integration Points

* **Upstream:** fed by `LayoutAnalyzer` in `utils/image/layout/analyzer.py`.
* **Downstream:** outputs are consumed by `regions` package (`ImageRegion`, `geometry`, `filters`).
* **Shared utilities:** use halo checks, masks, and connected-component stats from `utils/image/layout/utils`.

---

## 6️⃣ Testing & Validation

### Unit

* Provide synthetic grayscale pages (draw rectangles, lines, textures).
* Assert each detector returns the correct count and bounding boxes.

### Integration

* Run `LayoutAnalyzer.analyze_page_layout(gray_page)`.
* Expect combined detections (`figures + diagrams + photos + backgrounds`) > 0.
* Verify IoU de-duplication and confidence weighting behave correctly.

---

## 7️⃣ Extensibility

To add a new detector:

1. Subclass `BaseDetector`.
2. Implement `detect(gray) -> List[ImageRegion]`.
3. Register it in `LAYOUT_DETECTION["enabled_detectors"]`.
4. Optionally include config defaults and halo/mask utilities.

---

## ✅ Summary

The `layout/detectors` package gives your layout system:

* Clean separation between detector types.
* Unified return type (`ImageRegion`).
* Plug-and-play extensibility (add classical CV or DL detectors easily).
* Compatibility with post-processing and region analytics.

---


