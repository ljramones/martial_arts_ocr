That’s outstanding feedback — and it’s exactly what I hoped would happen: the two analyses now form a single coherent roadmap, where every piece has a clearly defined purpose and order of implementation.

You’ve essentially reached a *stable architecture plan* for the entire detection subsystem — one that balances low-level robustness with high-level flexibility. Here’s how we can now operationalize it:

---

## 🧭 **Unified Execution Plan**

### **Phase 1 – Pre-processing and Orientation Normalization**

**Goal:** guarantee a geometrically valid page before any analysis.

**Implementation sketch:**

* Extend `deskew_image()`:

  ```python
  if abs(abs(angle) - 90) < 3:
      rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
      return rotated
  ```
* Keep the original skew correction for < 15 °.
* Add an `orientation` field to `ImageInfo` so downstream detectors know what bias (left/right) means.

### **Phase 2 – Text Suppression Mask**

**Goal:** remove text before any figure search.

Combine:

* **Black-hat morphology** (dark text extraction)
* **MSER** for small glyphs
* **Dilation** to join characters into blocks
  Then:

```python
nontext = cv2.bitwise_and(bin_inv, cv2.bitwise_not(text_mask))
```

Store this in `ImageInfo.nontext_mask` for reuse across detectors.

### **Phase 3 – Detector Hardening**

**Goal:** make each detector trustworthy.

| Sub-Fix                        | Where                                          | Action                                           |
| ------------------------------ | ---------------------------------------------- | ------------------------------------------------ |
| Smaller closing kernel         | `_detect_figure_regions`                       | `(5,5)` instead of `(10,10)`                     |
| Whitespace halo                | new helper `_halo_ok()`                        | Evaluate ring white ratio ≥ 0.85                 |
| CC filtering                   | `_detect_by_contours` + `_filter_text_regions` | Reject `num_labels > 400 and median_area < 150`  |
| Bi-directional projection      | `_filter_text_regions`                         | Check both horizontal & vertical line regularity |
| Left-bias aware of orientation | scoring step                                   | Use orientation flag from Phase 1                |

### **Phase 4 – Unified Multi-Detector Pass**

**Goal:** fuse results from all strategies.

```python
cands = []
for fn in (self._detect_figure_regions,
           self._detect_by_contours,
           self._detect_by_variance,
           self._detect_uniform_regions):
    cands += fn(nontext)
regions = self._filter_text_regions(gray, self._remove_overlapping_regions(cands))
```

Each detector benefits from the clean `nontext` mask and hardened gates.

### **Phase 5 – Contextual Validation & Grouping**

**Goal:** combine signals beyond pure geometry.

1. **Caption anchoring:** lightweight OCR search for `r"Fig\.?\s*\d+"`; boost nearby candidates.
2. **Grid/Panel merging:** analyze candidate centroids; if variance of x/y spacing < threshold → merge into one “gallery” region.

---

## 🧮 **Verification Framework (Recommended)**

Before large-scale rollout:

* Build a mini benchmark using the 10 sample pages.

  * Ground-truth boxes in CSV (already started).
  * Compute IoU ≥ 0.5 for TP/FP/FN.
  * Output precision, recall, F1.
  * Save overlay images for manual inspection.
* Measure deltas after each phase to confirm incremental gain.

---

## ✅ **Implementation Order (Pragmatic Milestones)**

| Week | Deliverable                        | Expected Outcome                                   |
| ---- | ---------------------------------- | -------------------------------------------------- |
| 1    | Orientation + nontext mask         | Fix 90° pages & stop false text positives          |
| 2    | Halo + CC + bi-directional filters | Stable single-column accuracy                      |
| 3    | Multi-detector fusion              | Coverage of line-art + photo + uniform backgrounds |
| 4    | Caption + grid logic               | Contextual precision; clean final output           |
| 5    | Benchmark regression harness       | Quantified reliability baseline                    |

---

With this combined roadmap, your codebase evolves cleanly:

* **Phase 1–3** = hardening (no architectural risk)
* **Phase 4–5** = scalability & intelligence layers

If you’d like, I can now draft a **compact implementation plan** (function-level pseudocode + config diffs) for Phase 1 and 2, so your team can start coding immediately without touching later phases.
Would you like me to prepare that next?
