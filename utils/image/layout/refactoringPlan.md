## 🎯 Goals

* Separate **detectors**, **filters**, and **utils** into their own modules
* Preserve the public entrypoints (`LayoutAnalyzer.detect_*`, `analyze_page_layout`)
* Eliminate circular imports
* Enable modular growth (plug in new detectors later)
* Make unit testing sane

---

## 📦 Proposed package layout

```
utils/image/layout/
├─ __init__.py
├─ analyzer.py           # NEW: thin LayoutAnalyzer that orchestrates
├─ detectors/
│  ├─ __init__.py
│  ├─ figure.py          # _detect_figure_regions → FigureDetector
│  ├─ contours.py        # _detect_by_contours → ContourDetector
│  ├─ variance.py        # _detect_by_variance → VarianceDetector
│  └─ uniform.py         # _detect_uniform_regions → UniformDetector (optional)
├─ filters/
│  ├─ __init__.py
│  └─ text_filter.py     # _filter_text_regions → TextRegionFilter
├─ post/
│  ├─ __init__.py
│  └─ merge.py           # merge_overlapping_regions + _merge_regions
├─ utils/
│  ├─ __init__.py
│  ├─ halo.py            # _halo_ok
│  ├─ masks.py           # mask helpers (optional)
│  ├─ projections.py     # horizontal/vertical regularity helpers
│  └─ cc_metrics.py      # CC overflow/area metrics
```

> Keep `utils/image/image_layout.py` as a **compat façade** that re-exports `LayoutAnalyzer` from `analyzer.py` so old imports keep working while you migrate callers.

---

## 🧩 Key interfaces

### 1) Detector protocol (simple, duck-typed)

```python
# utils/image/layout/detectors/__init__.py
from typing import List, Optional
import numpy as np
from utils.image.regions.core_image import ImageRegion

class BaseDetector:
    def detect(self, gray: np.ndarray) -> List[ImageRegion]:
        raise NotImplementedError
```

Each detector gets its own class holding config and helpers:

```python
# utils/image/layout/detectors/figure.py
class FigureDetector(BaseDetector):
    def __init__(self, cfg, halo_check):
        self.cfg = cfg
        self.halo_ok = halo_check

    def detect(self, gray: np.ndarray) -> List[ImageRegion]:
        # ← move your _detect_figure_regions logic here
        # use self.cfg['halo_ring'], self.cfg['halo_min_white'], etc.
        ...
```

Same for `ContourDetector`, `VarianceDetector`, `UniformDetector`.

### 2) Text filter as a small class

```python
# utils/image/layout/filters/text_filter.py
class TextRegionFilter:
    def __init__(self, cfg):
        self.cfg = cfg

    def filter(self, gray: np.ndarray, regions: List[ImageRegion]) -> List[ImageRegion]:
        # ← move your _filter_text_regions logic here
        ...
```

### 3) Post-processing merge helpers

```python
# utils/image/layout/post/merge.py
def remove_overlaps(regions: List[ImageRegion], iou_threshold: float = 0.3) -> List[ImageRegion]: ...
def merge_overlapping(regions: List[ImageRegion], overlap_threshold: float = 0.3) -> List[ImageRegion]: ...
```

### 4) Halo + utilities

```python
# utils/image/layout/utils/halo.py
def halo_ok(gray: np.ndarray, x: int, y: int, w: int, h: int, ring: int, min_white: float) -> bool:
    ...
```

---

## 🧠 The new LayoutAnalyzer (thin orchestrator)

```python
# utils/image/layout/analyzer.py
import cv2, numpy as np
from typing import Any, Dict, List, Optional
from config import get_config
from utils.image.regions.core_image import ImageRegion
from .detectors.figure import FigureDetector
from .detectors.contours import ContourDetector
from .detectors.variance import VarianceDetector
from .filters.text_filter import TextRegionFilter
from .post.merge import remove_overlaps, merge_overlapping
from .utils.halo import halo_ok

class LayoutAnalyzer:
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        cfg = get_config().LAYOUT_DETECTION.copy()
        if config_override:
            cfg.update(config_override)
        self.cfg = cfg

        # Build components
        self.figure = FigureDetector(self.cfg, halo_check=lambda g,x,y,w,h: halo_ok(
            g, x,y,w,h, self.cfg.get('halo_ring',4), self.cfg.get('halo_min_white',0.85)
        ))
        self.contours = ContourDetector(self.cfg, halo_check=lambda g,x,y,w,h: halo_ok(
            g, x,y,w,h, self.cfg.get('halo_ring',4), self.cfg.get('halo_min_white',0.85)
        ))
        self.variance = VarianceDetector(self.cfg)
        self.text_filter = TextRegionFilter(self.cfg)

    def _to_gray_u8(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim==3 and image.shape[2]==3 else image
        return gray if gray.dtype==np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_image_regions(self, image: np.ndarray, nontext_mask: Optional[np.ndarray]=None) -> List[ImageRegion]:
        gray = self._to_gray_u8(image)
        if isinstance(nontext_mask, np.ndarray):
            if nontext_mask.shape != gray.shape:
                nontext_mask = cv2.resize(nontext_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            gray = cv2.bitwise_and(gray, gray, mask=nontext_mask)

        regions: List[ImageRegion] = []
        # Detector order remains the same
        figs = self.figure.detect(gray); regions += figs
        if not figs: regions += self.contours.detect(gray)
        # Optionally: regions += self.variance.detect(gray)

        merged = remove_overlaps(regions, iou_threshold=0.3)
        return self.text_filter.filter(gray, merged)

    def detect_text_regions(self, image: np.ndarray, nontext_mask: Optional[np.ndarray]=None, **kw) -> List[ImageRegion]:
        # you can keep MSER logic here or move it to detectors/text.py
        from .detectors.text import TextMserDetector  # if you choose to split
        gray = self._to_gray_u8(image)
        if isinstance(nontext_mask, np.ndarray):
            if nontext_mask.shape != gray.shape:
                nontext_mask = cv2.resize(nontext_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            gray = cv2.bitwise_and(gray, gray, mask=nontext_mask)
        return TextMserDetector(self.cfg).detect(gray, **kw)

    def analyze_page_layout(self, image: np.ndarray, proc=None, nontext_mask: Optional[np.ndarray]=None) -> Dict[str, Any]:
        mask = nontext_mask
        if mask is None and proc is not None and hasattr(proc, "get_last_nontext_mask"):
            try: mask = proc.get_last_nontext_mask()
            except Exception: mask = None

        text_regions = self.detect_text_regions(image, nontext_mask=mask)
        image_regions = self.detect_image_regions(image, nontext_mask=mask)

        h, w = image.shape[:2]
        total = max(1, h*w)
        return {
            'text_regions': [r.to_dict() for r in text_regions],
            'image_regions': [r.to_dict() for r in image_regions],
            'statistics': {
                'num_text_regions': len(text_regions),
                'num_image_regions': len(image_regions),
                'text_coverage': sum(r.area for r in text_regions)/total,
                'image_coverage': sum(r.area for r in image_regions)/total,
                'page_size': (w, h),
                'nontext_keep_ratio': None if mask is None else float(np.mean(mask==255)),
            }
        }
```

---

## 🔁 Back-compat shim

Keep `utils/image/image_layout.py` with:

```python
# utils/image/_legacy_image_layout.py
from utils.image.layout.analyzer import LayoutAnalyzer
```

That keeps existing imports alive: `from utils.image.image_layout import LayoutAnalyzer`.

---

## 🧪 Testing strategy (fast)

* Unit test each detector class with a tiny synthetic gray image
* Test `TextRegionFilter.filter` on cropped text ROIs to ensure rejection
* Smoke test `LayoutAnalyzer.detect_image_regions` with and without `nontext_mask`

---

## 🚚 Migration plan (2 PRs)

**PR-1 (non-breaking)**

* Add new package under `utils/image/layout/`
* Cut/paste logic into detectors/filters/utils modules
* Keep `image_layout.py` re-exporting `LayoutAnalyzer`

**PR-2 (cleanup)**

* Delete unused helpers from `image_layout.py` after a week of soak
* Add detector registry in config if desired (e.g., `LAYOUT_DETECTION.enabled_detectors: ["figure","contours","variance"]`)

---

## ⚙️ Performance tips

* Cache computed `edges`, `binary`, `connectedComponents` stats if multiple detectors need them on the same ROI (pass a shared context dict in future).
* Use `INTER_NEAREST` for mask resize (already done).
* Consider precomputing **integral images** (optional) for fast projection sums if areas are large or repeated.

---

If you want, I can generate the **first PR content** (files with your existing logic transplanted into `detectors/figure.py`, `detectors/contours.py`, `filters/text_filter.py`, `utils/halo.py`, and the new `analyzer.py`) so you can copy/paste and run.
