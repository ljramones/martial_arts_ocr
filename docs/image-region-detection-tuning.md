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
