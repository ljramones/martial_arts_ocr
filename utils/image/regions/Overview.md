# Image Regions Overview

This package provides the **core geometric and textual region utilities** used across the OCR and page-layout pipeline.
It defines lightweight dataclasses (`ImageRegion`, `ImageInfo`) and modular helpers for region math, grouping, filtering, and text cleanup.

---

## Goals

* Keep **types lightweight and dependency-free** (safe to import anywhere).
* Centralize **bbox math and region geometry** (no more scattered ad-hoc math).
* Provide high-level building blocks for **layout analysis** and **text assembly**.
* Maintain clean separation between **geometry**, **layout grouping**, and **text normalization**.

---

## Module Map

```
utils/image/regions/
  core_types.py     # Dataclasses and basic geometry; defines ImageRegion, ImageInfo
  convert.py        # Convert between tuples, _Box, and ImageRegion
  geometry.py       # Pure bbox math and transforms (expand, clamp, IoU, etc.)
  grouping.py       # Merge/cluster regions into lines and proximity groups
  layout.py         # Split into columns, sort in reading order
  filters.py        # Size/aspect/overlap filters and NMS
  text_fixups.py    # Post-OCR text normalization (hyphens, wraps, whitespace)
```

---

## `core_types.py` — Core Structures

**Purpose**
Lightweight, dependency-free dataclasses that define the shape and semantics of regions and image metadata.

**Key Types**

| Type          | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `BBox`        | Tuple `(x1, y1, x2, y2)` — inclusive-exclusive box               |
| `_Box`        | Internal immutable helper with area/IoU/translate/scale methods  |
| `ImageRegion` | Region dataclass: `bbox`, optional `points`, and metadata fields |
| `ImageInfo`   | File/array metadata: width, height, dtype, format, dpi, etc.     |

**Highlights**

* All geometry methods (`translate`, `scale`, `expand`, `shrink`, `clamp`) return new immutable instances.
* Optional `points` list supports polygonal regions (for segmentation masks).

---

## `convert.py` — Conversions

**Purpose**
Adapt between internal `_Box`, tuple bboxes, and `ImageRegion`.

**APIs**

* `normalize_bbox()` / `to_int_bbox()` / `to_float_bbox()` — sanitize coordinates.
* `region_to_box()` / `box_to_region()` — bridge `_Box` ↔ `ImageRegion`.
* `bbox_to_region()` / `region_to_bbox()` — tuple ↔ region helpers.

**Usage**

```python
r = ImageRegion(bbox=(10,20,50,60))
b = region_to_box(r)
r2 = box_to_region(b, region_type="word")
```

---

## `geometry.py` — Math & Transforms

**Purpose**
Pure math for bounding boxes and region transforms.

**Highlights**

* Measurements: `bbox_area`, `bbox_iou`, `bbox_ioa`, `center_distance`
* Overlaps: `horizontal_overlap`, `y_overlap_ratio`
* Transforms: `translate_bbox`, `scale_bbox`, `expand_bbox`, `clamp_bbox`
* Aspect helpers: `grow_to_aspect`, `fit_bbox_into`
* Region wrappers: `expand_region`, `scale_region`, `clamp_region`

**Example**

```python
from utils.image.regions.geometry import bbox_iou, expand_region
iou = bbox_iou(r1.to_tuple(), r2.to_tuple())
r_pad = expand_region(r1, 5)
```

---

## `grouping.py` — Line & Cluster Merging

**Purpose**
Combine adjacent regions (words, characters) into logical **lines** or **clusters**.

**APIs**

* `merge_regions_into_lines()` → `List[List[ImageRegion]]`
* `lines_to_regions()` → merge lines into one region per line
* `group_regions_by_proximity()` → cluster nearby boxes into groups

**Parameters**

* `y_overlap_min` — minimum vertical overlap ratio to merge
* `max_x_gap` / `max_dx` / `max_dy` — horizontal & vertical thresholds

**Example**

```python
lines = merge_regions_into_lines(word_boxes, y_overlap_min=0.35)
line_regions = lines_to_regions(lines, pad=4)
groups = group_regions_by_proximity(word_boxes, max_dx=48, max_dy=20)
```

---

## `layout.py` — Columns & Reading Order

**Purpose**
Detect page **columns** and sort regions in **reading order**.

**APIs**

* `split_regions_into_columns()` → auto or fixed column splits based on x-center gaps.
* `sort_regions_reading_order()` → flatten to reading sequence (columns L→R, rows T→B).

**Example**

```python
cols = split_regions_into_columns(line_regions, min_gutter=40)
ordered = sort_regions_reading_order(line_regions, y_tolerance=10)
```

---

## `filters.py` — Selection & Deduplication

**Purpose**
Generic filters for region lists.

**APIs**

* Size/shape: `filter_regions_by_size`, `filter_by_aspect_ratio`, `filter_by_area`
* Overlap/NMS: `dedupe_overlaps`, `nms`
* Sorters: `sort_top_left`, `sort_reading_order_like`

**Example**

```python
filtered = filter_regions_by_size(regions, min_width=20, min_height=10)
unique = dedupe_overlaps(filtered, iou_threshold=0.7)
ordered = sort_reading_order_like(unique)
```

---

## `text_fixups.py` — Post-OCR Cleanup

**Purpose**
Normalize and merge OCR text lines for readable output.

**Pipeline**

1. `normalize_whitespace`
2. `fix_hyphenated_breaks`
3. `merge_soft_wrapped_lines`
4. `collapse_blank_lines`
5. (optional) `remove_duplicate_lines`
6. (optional) `normalize_quotes_dashes`

**Wrapper**

* `post_ocr_fixups(text, opts: FixupOptions)` — runs a conservative default pipeline.

**Example**

```python
from utils.image.regions.text_fixups import post_ocr_fixups
clean = post_ocr_fixups(raw_ocr_text)
```

---

## Typical Workflow

```python
from utils.image.regions import (
    ImageRegion, merge_regions_into_lines, lines_to_regions,
    split_regions_into_columns, sort_regions_reading_order,
    filter_regions_by_size, post_ocr_fixups
)

# 1) Filter small OCR boxes
regions = filter_regions_by_size(raw_regions, min_width=5, min_height=5)

# 2) Merge characters/words into lines
lines = merge_regions_into_lines(regions)
line_regions = lines_to_regions(lines, pad=2)

# 3) Detect columns and reading order
ordered = sort_regions_reading_order(line_regions)

# 4) Extract text, then clean up
clean_text = post_ocr_fixups(ocr_engine.extract_text(ordered))
```

---

## Dependencies & Design Rules

* **Pure Python**, stdlib only; safe in any environment.
* Avoid `cv2`/`numpy` to prevent heavy imports in lightweight modules.
* `ImageRegion` is the canonical region container; always operate on it, not raw tuples.
* Each submodule handles **one responsibility** — geometry, grouping, layout, etc.

---

## See Also

* [`utils/image/io`](../io/overview.md) — for image loading/saving and metadata extraction.
* [`utils/image/ops`](../ops) — for image-level operations (resize, tone, extraction).

---
