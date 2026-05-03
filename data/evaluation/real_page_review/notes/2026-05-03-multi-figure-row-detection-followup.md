# Multi-Figure Row Detection Follow-Up

## Purpose

Diagnose why the first multi-figure row proposal pass returned zero candidates on the real failure page and add the narrowest review-mode improvement.

## Finding

The real page did not match the clean synthetic row pattern.

Existing runtime project diagnostics for `page_025` showed:

- `ContourDetector found 6 regions`
- `MultiFigureRowDetector found 0 regions`
- final import still contained only the central figure

After adding detailed row-detector diagnostics, the reason became explicit: the row detector saw the page as one oversized connected visual component rather than separated panel candidates.

Representative diagnostic:

```text
detector: multi_figure_rows
raw_count: 0
returned_count: 0
rejected:
  bbox: [0, 0, 1200, 1600]
  reason: area_too_large
```

The broad parent candidate rejected by the text filter remained relevant:

```text
bbox: [92, 75, 783, 760]
reason: text_like_components
```

That broad region can contain smaller visual children, but accepting the broad parent itself would be wrong.

## Implementation

Added two review-mode improvements:

1. `MultiFigureRowDetector` now records richer diagnostics:
   - page size;
   - thresholds;
   - rejected component reasons;
   - bands considered;
   - candidates returned.

2. Added opt-in child visual proposals from broad rejected candidates:
   - enabled only for review-mode extraction with `enable_broad_rejected_child_visuals`;
   - leaves the broad rejected parent rejected;
   - proposes smaller visual children with `detector=broad_rejected_child_visuals`;
   - marks child proposals as `needs_review=true`;
   - records parent bbox and parent rejection reason.

## Synthetic Test Result

Synthetic tests now cover:

- clean three-panel rows still produce sibling figure candidates;
- zero-result diagnostics explain rejection reasons;
- broad text-like parents with embedded visual children produce review-required child proposals;
- caption strips are not emitted as child visuals;
- dense paragraph text is not emitted as child visuals.

## Page 025 Manual Check

A direct local check on the effective-oriented `IMG_3312.jpg` produced:

```text
raw_candidate_count: 3
accepted_count: 3
rejected_count: 2

detectors:
  figure: returned 0
  contours: returned 6
  multi_figure_rows: returned 0
  broad_rejected_child_visuals: returned 3
```

Accepted boxes included:

```text
[444, 91, 398, 277]    broad_rejected_child_visuals, needs_review=true
[457, 489, 325, 154]   broad_rejected_child_visuals, needs_review=true
[463, 745, 412, 563]   contours
```

This is not perfect layout understanding, but it gives the reviewer more useful boxes than the single central figure result. The workbench can now show child visual proposals from the broad rejected region for review.

## Limits

- This remains review-mode candidate generation, not canonical layout truth.
- The child proposal pass may still merge nearby child visuals during consolidation.
- The reviewer must inspect, resize, retype, ignore, or delete these proposals.
- OCR, orientation, and workbench import behavior are unchanged.

## Next Step

Rerun `Run Recognition` on the page in the workbench and inspect `recognition_diagnostics.detector_diagnostics`.

If the proposed children are useful but too broad, the next pass should tune child bbox refinement. If they are noisy, add a stricter visual-child score threshold rather than accepting broad parents.
