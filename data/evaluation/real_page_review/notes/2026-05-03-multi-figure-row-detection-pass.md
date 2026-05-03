# Multi-Figure Row Detection Pass

## Purpose

Improve review-mode region proposal recall for pages that contain multiple adjacent figure, photo, or diagram panels in the same visual band.

## Observed Issue

Workbench recognition diagnostics showed that the workbench imported the final detector output correctly. The missing side figures were not lost during workbench import and were not an orientation problem.

For the page under review, the final lifecycle was:

- `raw_candidate_count`: 2
- `accepted_count`: 1
- `rejected_count`: 1
- `imported_count`: 1

The accepted/imported region was the central figure. A broad text-heavy candidate was rejected as `text_like_components`, which appeared appropriate. The left and right figures were not present as standalone final candidates.

## Implementation

Added a conservative review-mode-only multi-figure row proposal detector:

- file: `utils/image/layout/detectors/multi_figure_rows.py`
- detects separated figure-like visual masses in horizontal bands;
- filters out small caption strips and dense paragraph-like text;
- emits advisory `diagram` candidates with `detector=multi_figure_rows` metadata;
- records compact detector diagnostics including raw and returned counts.

The detector is enabled only in the review-mode image-region extraction path used by the workbench. It does not replace the existing figure or contour detectors.

## Diagnostics

Recognition diagnostics now include per-detector summaries. The multi-figure row detector reports:

- `detector`
- `raw_count`
- `returned_count`
- candidate boxes and metadata

This makes it possible to distinguish:

- candidates never proposed;
- candidates proposed but rejected;
- candidates suppressed or merged;
- candidates accepted and imported.

## Synthetic Test Result

A synthetic page with three separated figure-like panels in one row produces three multi-figure row candidates. Caption strips and dense paragraph text do not produce row candidates.

Workbench recognition on the synthetic row imports three final regions and exposes the multi-figure row detector diagnostics.

## Page 025 Manual Result

The original page should be rerun in the local workbench after this pass. The expected useful signal is whether `recognition_diagnostics.detector_diagnostics` now shows additional `multi_figure_rows` candidates for the side figures.

If side figures are still absent from the row detector diagnostics, the next pass should tune candidate generation for that specific visual pattern. If they are proposed but rejected or suppressed, the next pass should inspect the rejection/consolidation reason rather than changing workbench import.

## Known Limitations

- This is review-mode candidate generation, not production layout classification.
- It uses exact visual/component heuristics and does not understand semantic figure groupings.
- It intentionally does not OCR text.
- It may still miss low-contrast, fragmented, or text-adjacent side figures.
- False positives are acceptable only because the reviewer can ignore, delete, resize, or retype regions.

## Behavior Boundaries

- Orientation behavior unchanged.
- OCR behavior unchanged.
- Workbench import mapping unchanged.
- Runtime extraction defaults unchanged outside the opt-in review-mode extraction path.
