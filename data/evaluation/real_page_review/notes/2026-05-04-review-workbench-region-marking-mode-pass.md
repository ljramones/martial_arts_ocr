# Review Workbench Region Marking Mode Pass

## Purpose

Simplify the local research review workbench around the reviewer task: look at the page, mark meaningful regions, correct machine suggestions, and save clean review state.

## Context

Recognition debugging on `page_025` showed that the current CV path can find a useful central figure and reject text false positives, but it does not reliably propose every side figure in a row. Continuing detector tuning for that page risks reintroducing paragraph/text false positives.

The workbench should therefore make manual region marking fast instead of exposing every engineering control as the primary workflow.

## Changes

- Added region marking tools:
  - `Select / Move`
  - `Draw Image Region`
  - `Draw Text Region`
  - `Draw Japanese Region`
- Added a right-panel region inventory grouped by image, text, Japanese, ignored, and other regions.
- Added quick type buttons for common reviewer labels.
- Moved duplicate and nudge controls into an advanced section.
- Added review feedback metadata for manual additions, accepted/resized machine regions, rejected machine regions, and ignored manual regions.

## Review Feedback Labels

The workbench now records local feedback metadata such as:

- `manually_added`
- `accepted_positive`
- `resized_positive`
- `type_corrected`
- `false_positive`
- `ignored`

This is not live learning. It is auditable review data for later detector evaluation, regression fixtures, or training exports.

## Non-Goals

- No detector tuning.
- No OCR execution.
- No orientation changes.
- No extraction default changes.
- No source image mutation.

## Expected Workflow

For a page with a row of figures:

1. Run orientation if needed.
2. Run recognition if useful.
3. Keep or adjust good machine boxes.
4. Use `Draw Image Region` to mark missed figures.
5. Use the region inventory as a checklist.
6. Save reviewed state.

## Decision

Manual region marking is the right near-term path for pages where detection gives a useful starting point but not complete coverage. The reviewed regions and feedback metadata should be used later to evaluate or improve detector behavior in batch.
