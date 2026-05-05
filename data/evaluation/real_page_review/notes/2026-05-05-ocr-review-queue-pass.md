# OCR Review Queue Pass

## Purpose

The OCR-all batch review showed that page-level OCR now creates attempts efficiently, but the next bottleneck is reviewing those generated OCR attempts. This pass adds a small current-page OCR review queue so the reviewer can move through attempts without manually selecting each region.

## Added Behavior

- Added an `OCR Review Queue` section to the selected-region OCR panel.
- Queue is current-page only.
- Default filter is pending/unreviewed OCR attempts.
- Queue can also show accepted, edited, rejected, or all attempts.
- `Previous` and `Next` select OCR attempts and their associated regions.
- `Accept & Next`, `Save Edit & Next`, and `Reject & Next` reuse the existing OCR attempt review endpoint and then advance through the queue.

## State / Semantics

The queue is client-side navigation over existing page OCR attempt state:

```text
state.page.ocr_attempts
```

It does not add OCR routes, rerun OCR, change export behavior, or mutate raw OCR text. Existing non-destructive OCR review rules still apply:

```text
raw text preserved
cleaned text preserved
reviewed_text separate
source_text_mutated=false
```

## Scope Kept Small

Deferred:

- project-wide OCR review queue;
- selected/range/all-page queue;
- OCR rerun controls;
- OCR variant queueing;
- export changes;
- DOCX/PDF.

## Validation

Automated checks verify the workbench route exposes the queue controls. JavaScript syntax and the normal test suite were run for this pass.

## Next Use

Use this after `OCR Reviewed Text Regions`:

```text
OCR all reviewed text regions
  -> queue pending attempts
  -> accept/edit/reject and advance
  -> export review bundle + HTML
```

The next useful expansion, if this current-page queue proves comfortable, is a selected-pages or project-level OCR review queue.
