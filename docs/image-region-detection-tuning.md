# Image Region Detection Tuning

## Observed Failure Pattern

The first Donn Draeger review pass found that crop saving and basic contour detection work mechanically, but semantic filtering is weak. Typewritten text fragments, bold headings, and rotated/vertical text columns were frequently returned as `diagram` image regions.

Reviewed examples:

- `original_img_3288`: bold title text detected as diagrams.
- `original_img_3292`: body-outline diagram detected, but a vertical text fragment was also detected.
- `original_img_3312`, `3331`, `3389`: typewritten text columns detected as diagrams.
- `original_img_3340`: hand/gesture diagram detected, plus one body-text false positive.

## Current Detector Behavior

`LayoutAnalyzer` combines figure and contour detectors, then merges candidates and applies `TextRegionFilter`. The previous config exempted `diagram` regions from filtering, so contour-produced diagram candidates bypassed the final text-like rejection stage.

## Text-Like Rejection Heuristics

This pass keeps the classical CV path and adds OCR-free rejection signals:

- connected-component count and median component area
- foreground/stroke density
- small-component fraction
- row/column occupancy
- horizontal text-line shape
- narrow rotated/vertical text-column shape
- title-like bold text fragments

`figure` remains exempt for optional YOLO output. Classical `diagram` candidates are no longer exempt.

## Added Config Values

The layout config now includes `region_*` options such as:

- `region_reject_text_like`
- `region_reject_rotated_text_like`
- `region_text_like_min_components`
- `region_text_like_min_density`
- `region_text_like_max_density`
- `region_text_like_min_median_component_area`
- `region_text_like_max_median_component_area`
- `region_text_line_max_height`
- `region_vertical_text_max_aspect_ratio`

These values are represented by `RegionDetectionOptions` in `utils/image/layout/options.py`.

## Diagnostics

`LayoutAnalyzer.detect_image_regions_with_diagnostics()` returns accepted regions and rejected candidates with rejection reasons. Notebook 05 now prints those rejected candidates so real-page review can compare accepted diagrams against rejected text fragments.

## Remaining Risks

- The heuristics are tuned against first-pass failures and synthetic tests, not the full 112-page corpus.
- Real captions near diagrams remain difficult because they can be spatially attached to legitimate figures.
- Some photo-like or shaded image regions may need separate handling.
- A detector strategy abstraction may still be useful if threshold tuning cannot balance text rejection and diagram recall.

## Post Text-Filter Recall Tuning

The second Donn Draeger review showed that text-like rejection fixed the major title/body/vertical-text false positives, but recall was still weak for several real page cases:

- `original_img_3335`: a large illustration/photo-like figure was missed.
- `original_img_3397`: sparse symbols, arrows, and hand-drawn clusters were missed.
- `original_img_3344`: a labeled diagram was over-rejected as `text_like_components`.
- `original_img_3352` and `3353`: accepted crops were useful, but labels/annotations are naturally included in the figure area.

This pass keeps the classical CV detector as the default and improves recall without adding OCR or ML dependencies.

### Candidate Generation Changes

- `contour_max_area_ratio` is raised from `0.4` to `0.5` so large real illustrations can become candidates.
- `contour_topk` is raised from `2` to `6` so smaller valid drawings are not dropped before filtering when larger text-like candidates are present.
- Broad early connected-component text rejection remains disabled by default because it hid sparse drawings and labeled diagrams.
- Top/left page-edge text-like bands are still rejected early with `contour_reject_page_edge_text_like`. These large exterior contours can otherwise hide nested diagrams when `cv2.RETR_EXTERNAL` is used. Right/bottom edge candidates are not rejected early because real diagrams can touch those edges.
- Final overlap removal now happens after text-like filtering, so rejected text candidates cannot suppress smaller real diagrams.

### Labeled Diagram Preservation

`TextRegionFilter` now preserves candidates that look like mixed labeled diagrams. The key signal is component-size diversity:

- typewritten text has relatively uniform component areas
- labeled diagrams mix small label glyphs with larger arrows, outlines, and hand-drawn strokes

New `region_*` options:

- `region_preserve_labeled_diagrams`
- `region_labeled_diagram_min_component_area_ratio`
- `region_labeled_diagram_min_small_component_fraction`
- `region_labeled_diagram_max_density`

The filter also adds `sparse_text_band` rejection for wide, shallow bands made of small repeated components. This recovers the previous behavior on sparse text-like pages without blocking square or vertical sparse symbol clusters.

### Real-Page Spot Check

With default settings after this pass:

- `3335` accepts the large figure and rejects the top text block.
- `3344` accepts the labeled diagram and rejects the text fragment.
- `3397` accepts sparse symbol/arrow clusters.
- `3292`, `3327`, and `3340` still retain their real diagrams.
- `3331`, `3378`, and `3389` still reject text-like candidates.
- `3352` and `3353` retain labeled/hand-drawn figure areas after final NMS was moved behind filtering.

### Remaining Risks After Recall Tuning

- Higher recall now returns additional candidates on some pages (`3292`, `3312`, `3327`, `3345`, `3352`). These should be visually reviewed before runtime integration.
- Captions and labels are still part of many useful crops. That is acceptable for review, but later runtime integration may need caption-aware postprocessing.
- Sparse symbol grouping is heuristic and should be rechecked on the same 19 reviewed pages before integration.
