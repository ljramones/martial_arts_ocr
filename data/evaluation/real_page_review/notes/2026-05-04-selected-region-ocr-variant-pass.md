# Selected-Region OCR Variant Pass

## Purpose

Add review-mode OCR variants for one selected workbench region at a time.

The default selected-region OCR path remains available. Variants are a comparison aid when the default route is poor; they do not run OCR over all regions and do not promote any OCR output to canonical text.

## Implementation Summary

- Added deterministic OCR route variants by region type.
- Added alternate PSMs and preprocessing profiles for selected text/Japanese region types.
- Added `Run Variants` for the selected region.
- Stored each variant as a normal page-level OCR attempt.
- Added variant metadata:
  - `variant_group_id`
  - `variant_index`
  - `variant_total`
  - `is_variant`
- Kept raw OCR text and cleaned text inspectable.
- Kept `source_text_mutated=false`.

## Variant Examples

For `english_text`, variants include:

- default `eng` / PSM 6 / none
- `eng` / PSM 4 / none
- `eng` / PSM 11 / none
- `eng` / PSM 6 / grayscale
- `eng` / PSM 6 / threshold
- `eng` / PSM 6 / upscale 2x
- `eng` / PSM 6 / contrast/sharpen

For Japanese regions, variants stay aligned with the prior region OCR evidence:

- horizontal Japanese: `jpn` / PSM 6, plus threshold and `eng+jpn` comparison
- vertical Japanese: `jpn_vert` / PSM 5, plus preprocessing comparisons

## IMG_3312 Real-Page Validation

- Source image:
  `/Users/larrymitchell/ML/martial_arts_ocr/data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg`
- Source orientation:
  270 degrees
- Orientation detection:
  detected current orientation 270 degrees, confidence 0.468
- Correction applied:
  90 degrees clockwise
- Region bbox used:
  `[180, 70, 940, 180]` on the effective-oriented page
- Expected visible text:
  `[Question.] Aaaa yes.`

### Default OCR Output

```text
(Question. ]

Aaaa yes. I'm glad you're frank. Okay. You state your
-ion so everybody can be brought up to date.
```

The default route did not preserve the opening bracket/punctuation exactly, but it did recover `Aaaa yes.` and the surrounding line. This is materially better than the earlier bad-region output described as `Le opmgageet`, which suggests bbox/orientation/region selection were part of that failure.

### Variant Outputs

| Variant | Output Summary | Notes |
|---|---|---|
| default `eng` PSM 6 | `(Question. ] Aaaa yes...` | Good body text, punctuation imperfect. |
| `eng` PSM 4 | `(Question. ] Aaaa yes...` | Similar, but more line artifacts later. |
| `eng` PSM 11 | Reordered sparse words | Worse reading order. |
| grayscale | `(Question. ] Aaaa yes...` | Similar to default. |
| threshold | empty | Too destructive for this crop. |
| upscale 2x | `(Question. ] Aaaa yes...` | Highest confidence, similar opening line. |
| contrast/sharpen | `(Question. ] Aaaa yes...` | Similar to default. |

### Best Variant

- Best variant by deterministic score:
  `eng` / PSM 6 / `upscale_2x`
- Did any variant improve the opening line?
  Partially. The good bbox and orientation recovered `Aaaa yes.` across several variants, but no variant perfectly recovered `[Question.]`.

## Decision

OCR variants are useful as a selected-region review tool, especially when default OCR is poor. They should remain review-only attempts. No automatic route selection should be treated as truth yet, and no OCR-all workflow should be added before attempt review controls exist.

## Non-Goals Preserved

- No OCR-all regions.
- No translation.
- No canonical Japanese field promotion.
- No automatic text correction.
- No private image or generated OCR output committed.
