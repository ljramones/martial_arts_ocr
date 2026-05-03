# Review Workbench Region Duplication Pass

## Purpose

Improve reviewer ergonomics for pages where recognition finds a useful region but misses nearby sibling regions.

## Reason

The page 025 figure-row debugging showed:

- the central figure is detected;
- paragraph/text false positives are now rejected;
- the side figures are still not reliably proposed by the current CV path.

Continuing detector tuning for this page risks reintroducing text false positives. For the local research workbench, a better next step is to make manual correction faster.

## Implementation

Added selected-region duplication controls:

- `Duplicate`
- `Duplicate Left`
- `Duplicate Right`
- `Duplicate Up`
- `Duplicate Down`

The left/right/up/down variants copy the selected region size and offset by one region width or height. The resulting bbox is clamped to page bounds.

Added nudge controls:

- button nudges move by 10 pixels;
- arrow keys move by 1 pixel when focus is not inside an input;
- Shift + arrow moves by 10 pixels.

## State Behavior

Duplicated regions are not treated as machine detections.

They use:

```text
source=reviewer_manual_duplicate
status=reviewed
detected_type=null
detected_bbox=null
reviewed_type=<selected effective type>
reviewed_bbox=<new bbox>
metadata.duplicated_from_region_id=<source region>
```

The original region remains unchanged.

## Use Case

For a row of three figures where only the center figure is detected:

1. Select the center figure.
2. Click `Duplicate Left`.
3. Move/resize the duplicate to the left figure.
4. Click `Duplicate Right` from the center or adjusted duplicate as appropriate.
5. Save reviewed bboxes and types.

This is faster and safer than loosening detector thresholds for this page.

## Verification Scope

Tests cover:

- duplicate region ID generation;
- left/right bbox offsets;
- clamping to page bounds;
- effective type preservation;
- duplicated region source/metadata;
- original region preservation;
- state reload behavior;
- API duplicate route.

## Behavior Boundaries

- Detector behavior unchanged.
- Orientation behavior unchanged.
- OCR behavior unchanged.
- Extraction defaults unchanged.
- No generated review projects committed.
