# Broad Rejected Child Visuals Follow-Up

## Purpose

Harden review-mode child visual proposals after the first broad-parent pass produced paragraph/text false positives on `page_025`.

## Observed Issue

The broad rejected child proposal pass correctly fired, but two of the three imported boxes covered upper text blocks rather than figures.

This was not an orientation, OCR, display, bbox-editing, or workbench import issue. The problem was specifically:

```text
broad_rejected_child_visuals proposed text chunks as visual children
```

## Diagnosis

The false-positive child boxes showed text-like features:

```text
regular_text_projection: 1
text_like_score: 0.58
large_dark_component_count: 0
max_dark_component_area_ratio: near 0
```

The central figure showed a different profile:

```text
regular_text_projection: 0
figure_like_score: 0.93
photo_like_score: 0.85
large_dark_component_count: 4
max_dark_component_area_ratio: 0.204
```

So the child proposal path needed the same kind of text-like/visual-mass check used elsewhere in layout filtering.

## Implementation

Child visual candidates now run through `TextRegionFilter.candidate_diagnostics`.

Child candidates are rejected when they show paragraph/text-like signals such as:

- regular text projection;
- high text-like score;
- low or missing large dark visual mass;
- no photo-like evidence.

Rejected child diagnostics include:

```text
reason: child_text_like
scores
features
```

Child visual proposals still require review:

```text
needs_review: true
```

The broad rejected parent remains rejected and is not imported.

## Synthetic Tests

Added tests for:

- broad rejected parent with figure-like silhouettes still emits child visual proposals;
- dense paragraph text does not emit child visual proposals;
- connected text false positives are recorded as `child_text_like`;
- caption strips remain rejected;
- clean multi-figure row behavior still passes.

## Page 025 Check

A direct local check on effective-oriented `IMG_3312.jpg` now gives:

```text
accepted_count: 1
rejected_count: 2

detectors:
  contours: returned 6
  multi_figure_rows: returned 0
  broad_rejected_child_visuals: returned 0

accepted:
  [463, 853, 340, 455] contours

child rejection reasons:
  area_too_large
  child_text_like
```

This fixes the immediate false-positive bug: the upper paragraph/text boxes are no longer imported as image candidates.

The side figures are still not proposed correctly on this page. That should be treated as a separate recall problem, not solved by accepting text-like child proposals.

## Remaining Limitation

The workbench still may require manual region creation for the side figures on this page. That is acceptable for now; importing text paragraphs as image candidates was worse than requiring manual boxes.

The next improvement, if needed, should focus on a more precise visual/silhouette detector for the side figures rather than broadening child proposal acceptance.

## Behavior Boundaries

- Orientation unchanged.
- OCR unchanged.
- Workbench import mapping unchanged.
- Broad rejected parent remains rejected.
- Review-mode scope preserved.
