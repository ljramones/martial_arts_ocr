# Paddle Fusion Validation

## Purpose

Validate whether initial containment-based Paddle fusion improves broad/mixed
crops before treating it as a preferred review-mode path.

## Setup

- PaddlePaddle version: `3.3.0`
- PaddleOCR version: `3.5.0`
- API used: `PPStructureV3`
- Runtime defaults changed? no
- Image extraction default: disabled
- Paddle fusion default: disabled

Fusion constants:

- `PADDLE_INSIDE_CLASSICAL_MIN`: `0.75`
- `PADDLE_AREA_TIGHTNESS_MAX`: `0.80`
- `PADDLE_AREA_TIGHTNESS_MIN`: `0.05`

## Summary Table

| Corpus | Sample ID | Classical Result | Paddle Result | Fused Result | Judgment | Regression? | Notes |
|---|---|---|---|---|---|---|---|
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | broad mixed parent `53,39,351,391` | two image regions plus text/title regions | tightened to `42,248,161,188` | better | no | One Paddle visual region met containment/tightness. V1 emits one best region only. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_10_58` | broad mixed parent `108,102,355,148` | one very broad image over most of page | unchanged | same | no | Paddle visual is larger than the classical parent, so containment tightening is not valid. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_19_36` | broad mixed parent `96,10,359,400` | right-side image plus separated text | unchanged | same | no | Paddle image is useful, but only about 68% contained by the classical parent with the recorded bbox. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | broad mixed parent `77,44,317,472` | top-right image plus separated text | tightened to `200,44,219,224` | better | no | Clean containment/tightness match. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_00` | partial photo-grid parent `53,44,139,170` | broad image around photo grid | unchanged | same | no | Paddle visual is larger than the classical parent, so V1 does not replace it. |
| DFD | `original_img_3335` | known large visual retained | broad image retains visual | unchanged | same | no | Paddle did not produce a qualifying tighter region inside the classical parent. |
| DFD | `original_img_3344` | labeled diagram retained | partial visual/text separation | unchanged | same | no | Labeled diagram not destroyed. |
| DFD | `original_img_3397` | sparse symbols retained | sparse symbols retained in broad/partial regions | unchanged | same | no | Clean classical regions are not fused. |
| DFD | `original_img_3352` | known candidates retained | broad image regions retained | unchanged | same | no | Clean classical regions are not fused. |
| DFD | `original_img_3330` | tall/narrow strips retained | broad image region retained | unchanged | same | no | Clean classical regions are not fused. |
| DFD known-good | `original_img_3292` | known-good diagrams retained | known-good diagrams retained | unchanged | same | no | No regression expected because clean classical regions are preserved. |
| DFD known-good | `original_img_3340` | known-good diagram retained | known-good diagram retained | unchanged | same | no | No regression expected because clean classical regions are preserved. |

## Corpus 2 Broad/Mixed Results

Containment fusion improved `2/5` Corpus 2 broad/mixed cases:

- `16_55_48`: tightened to one clean Paddle image region.
- `18_29_28`: tightened to the top-right visual region.

The remaining three cases did not meet V1 containment/tightness rules:

- `17_10_58`: Paddle found a broad page-level image, not a tighter crop.
- `17_19_36`: Paddle found a useful right-side image, but the recorded
  classical parent did not contain enough of the Paddle bbox for the `0.75`
  containment gate.
- `18_54_00`: Paddle found a useful photo-grid region that is larger than the
  recorded classical partial parent, so it is not a tightening replacement.

## DFD Hard Page Results

DFD hard pages did not regress. The implementation only fuses regions already
marked `mixed_region` or `needs_review`, and no qualifying Paddle replacement
destroyed known DFD visual content.

## Known-Good Regression Checks

Known-good pages stayed unchanged by design: clean classical regions are
preserved when they are not marked as mixed or needing review.

## Threshold Reality Check

- Containment `>= 0.75` is safe but misses one semantically useful Paddle
  region on `17_19_36` because the recorded classical parent only partially
  overlaps it.
- Area tightness `<= 0.80` correctly prevents replacing a parent with a broader
  Paddle region.
- Area tightness `>= 0.05` protects against tiny accidental micro-crops.

No constants were changed in this pass. The strict gate prevents over-claiming
Paddle fusion as solved when some Paddle improvements are not containment-style
refinements.

## Gate Result

Required:

- at least 4 of 5 Corpus 2 broad/mixed cases produce measurably better fused
  output than classical-only
- 0 of 5 DFD hard pages regress
- 0 of 2 known-good pages regress

Result:

- [ ] pass
- [x] fail

Observed:

- Corpus 2 improved count: `2/5`
- DFD hard-page regressions: `0/5`
- known-good regressions: `0/2`

## Decision

- [ ] Commit fusion defaults
- [ ] Adjust constants and rerun validation
- [x] Keep Paddle fusion disabled and experimental; re-plan before broader use

The optional implementation is useful infrastructure, but this validation does
not justify treating containment fusion as the preferred review-mode path yet.
The next branch should either test a lower containment threshold against more
annotated examples, support multi-region/layout-driven additions explicitly, or
move toward annotation/fine-tuning.
