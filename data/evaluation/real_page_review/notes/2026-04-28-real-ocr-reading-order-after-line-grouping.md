# Real OCR Reading Order After Line Grouping

Date: 2026-04-28  
Run date: 2026-04-29  
Temporary output: `data/notebook_outputs/ocr_text_reading_order_after_line_grouping/`

## Purpose

Spot-check the adaptive line grouping change on the same 8-page OCR readability
sample.

This was a validation pass only. Runtime defaults, OCR engine settings,
extraction behavior, corpus data, and database schema were unchanged.

## Change Validated

Line grouping now uses:

- median word-height based tolerance
- close vertical-center matching
- conservative y-overlap rescue for near-line words
- left-to-right word ordering
- extra spacing for large x gaps
- line metadata:
  - `line_grouping_method=adaptive_center_overlap_v1`
  - `reading_order_uncertain=true|false`

## Summary Table

| Sample ID | Lines Before | Lines After | Readable Length Before | Readable Length After | Outcome |
|---|---:|---:|---:|---:|---|
| `original_img_3337` | 58 | 59 | 3354 | 3356 | stable |
| `original_img_3288` | 10 | 10 | 343 | 343 | stable |
| `original_img_3344` | 56 | 56 | 2328 | 2348 | stable/noisy |
| `original_img_3330` | 10 | 10 | 549 | 549 | stable/noisy |
| `corpus2_new_doc_2026_04_28_16_56_38` | 6 | 6 | 125 | 126 | stable |
| `corpus2_new_doc_2026_04_28_18_29_28` | 48 | 48 | 1992 | 2004 | stable/noisy |
| `corpus2_new_doc_2026_04_28_16_55_48` | 22 | 22 | 623 | 626 | stable |
| `corpus2_new_doc_2026_04_28_20_26_02` | 12 | 12 | 105 | 107 | stable/fail source OCR |

## Observations

- The first version of the y-overlap rescue was too permissive and merged
  staggered neighboring lines on `original_img_3337`.
- The rule was tightened so y-overlap only rescues near-line cases.
- After tightening, simple text/list pages stayed stable.
- Large horizontal gaps now preserve visible spacing in `readable_text`.
- Mixed/figure/article pages remain limited by OCR quality and simple
  line-level ordering.

Examples:

```text
original_img_3337 before:
  three hours, you may fumble and stumble three, four,
  two or six times. So now, if you're a stupid man, you walk away,

original_img_3337 after:
  three hours, you may fumble and stumble three, four,
  two or six times. So now, if you're a stupid man, you walk away,
```

```text
original_img_3288 before:
  THE DRAEGER LECTURES
  AT

original_img_3288 after:
  THE DRAEGER LECTURES
  AT
```

## Decision

The adaptive grouping change is acceptable as a conservative text-normalization
improvement. It adds useful metadata and large-gap spacing without regressing
simple pages.

It does not solve:

- multi-column reading order
- figure/caption interleaving
- noisy OCR recognition
- vertical Japanese
- final page reconstruction

## Recommended Next Pass

Move to `DocumentResult` / `data.json` readability polish before attempting
Japanese analysis promotion.

Specific next work:

- expose line regions and `readable_text` more clearly in artifacts
- reduce review noise around word-level boxes where possible
- keep word boxes available for geometry and OCR-aware filtering
