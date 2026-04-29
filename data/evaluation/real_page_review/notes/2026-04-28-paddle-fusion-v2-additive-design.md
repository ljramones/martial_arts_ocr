# Paddle Fusion V2 Additive Design

## Failure Analysis Summary

V1 containment-only fusion improved `2/5` Corpus 2 broad/mixed cases. The three
misses were classified as:

- `17_10_58`: B/F, Paddle found a larger broad image region and the classical
  parent was too narrow.
- `17_19_36`: A/F, Paddle found a useful right-side visual region that was only
  partly contained by the classical parent.
- `18_54_00`: B/F, Paddle found a larger photo-grid region and the classical
  parent was partial.

The dominant problem is not label mapping. The classical parent crop is often
wrong or partial relative to Paddle's useful visual region.

## V2 Rule

V2 preserves V1 containment replacement. It adds a second path:

```text
if a high-confidence Paddle visual region is related to an unresolved
mixed/needs-review classical region, and is not represented by a clean
classical region, add it as a separate ImageRegion candidate.
```

Eligible Paddle labels:

- `figure`
- `image`
- `photo`
- `diagram`

Ineligible labels:

- `text`
- `title`
- `caption`
- `table`
- `unknown`

## Safety Constraints

- Image extraction remains disabled by default.
- Paddle fusion remains disabled by default.
- PaddleOCR remains an optional dependency.
- V2 does not lower V1 containment thresholds.
- V2 does not replace classical regions with larger Paddle boxes.
- V2 does not add all Paddle visual regions globally.
- V2 emits at most one Paddle-added region per unresolved mixed classical
  parent.
- Clean classical regions are preserved.
- Paddle regions duplicating a clean classical region are not added.
- Small micro-crops are rejected.

## Numeric Shared-Span Rule

For a Paddle/classical pair:

```text
horizontal_span_overlap_ratio =
  horizontal_overlap_width / min(paddle_width, classical_width)

vertical_span_overlap_ratio =
  vertical_overlap_height / min(paddle_height, classical_height)
```

A Paddle region is related to an unresolved mixed classical region if it has:

- partial bbox overlap, or
- `horizontal_span_overlap_ratio >= 0.50`, or
- `vertical_span_overlap_ratio >= 0.50`, or
- conservative proximity.

Relation diagnostics are emitted as:

- `horizontal_span_overlap_ratio`
- `vertical_span_overlap_ratio`
- `relation_reason`

## Improvement Definition

A Corpus 2 broad/mixed case counts as improved only when the Paddle-derived
region materially captures intended visual content better than the broad
classical parent and does not mostly capture unrelated body text.

Adding any Paddle region is not enough. The added region must be a better visual
crop, and the original broad parent should remain available for review unless
V1's original inside-parent replacement applies.

## DFD Regression Definition

A DFD regression occurs only when V2 emits an incorrect Paddle-derived region.
Incorrect means it chops off important visual content, mostly captures unrelated
text/noise, duplicates an already-correct clean classical region, or causes a
known-good visual to be lost or downgraded.

## Expected Improvement Cases

- `17_19_36`: expected to add the right-side Paddle image region.
- `18_54_00`: expected to add the broader Paddle photo-grid region as
  `needs_review`.
- `17_10_58`: not expected to improve because Paddle confidence is below the
  V2 additive confidence floor and the Paddle crop is broad.

V1 containment replacement should still account for:

- `16_55_48`
- `18_29_28`

## Tests Added

Tests cover:

- V1 containment replacement still passes.
- Partly outside Paddle visual regions can be added.
- Larger Paddle visual regions can be added without replacement.
- Unrelated Paddle visuals are not added.
- Duplicates of clean classical regions are not added.
- Text/table Paddle regions are not added.
- Shared-span relation uses the numeric `0.50` threshold.
- Additive metadata is emitted.
- ExtractionService passes additive fused regions through to `DocumentResult`.

## What Remains Before Validation

Run the same 12-page Paddle evaluation set and judge actual visual usefulness.
The hard gate is:

- at least `4/5` Corpus 2 broad/mixed cases improve
- `0` incorrect Paddle-derived additions on DFD hard pages
- `0` incorrect Paddle-derived additions on known-good pages
- no region-count explosion
- no obvious plain-text Paddle additions
