# Paddle Fusion V2 Validation

## Purpose

Validate whether additive Paddle fusion improves the V1 misses without lowering
containment thresholds or replacing classical regions with larger Paddle boxes.

## Setup

- PaddlePaddle version: `3.3.0`
- PaddleOCR version: `3.5.0`
- API used: `PPStructureV3`
- Runtime defaults changed? no
- Image extraction default: disabled
- Paddle fusion default: disabled

V2 internal constants:

- `PADDLE_ADDITIVE_CONFIDENCE_MIN`: `0.60`
- `PADDLE_SHARED_SPAN_MIN`: `0.50`
- `PADDLE_ADDITIVE_MICRO_AREA_RATIO_MIN`: `0.05`
- `PADDLE_ADDITIVE_ABSOLUTE_AREA_MIN`: `1024`

## Summary Table

| Corpus | Sample ID | Classical Result | Paddle Result | V1 Fused Result | V2 Fused Result | Judgment | Regression? | Notes |
|---|---|---|---|---|---|---|---|---|
| Corpus 2 | `16_55_48` | broad mixed parent | two clean image regions plus text/title | tightened to one Paddle image | unchanged from V1 | better | no | V1 containment still handles this case. V2 does not add extra regions. |
| Corpus 2 | `17_10_58` | narrow mixed parent | broad page-level image, confidence `0.507` | unchanged | unchanged | same | no | Paddle confidence is below V2 floor and crop is broad. Not counted as improved. |
| Corpus 2 | `17_19_36` | broad/partial mixed parent | right-side image plus separated text | unchanged | adds Paddle image `(322,224,160,226)` | better | no | V2 additive fixes the V1 containment miss. |
| Corpus 2 | `18_29_28` | broad mixed parent | top-right image plus separated text | tightened to Paddle image | unchanged from V1 | better | no | V1 containment still handles this case. |
| Corpus 2 | `18_54_00` | partial photo-grid parent | broad image around photo grid plus header text | unchanged | adds Paddle image `(31,38,400,595)` as `needs_review` | better | no | Added region captures the grid better than the partial parent, but remains broad/uncertain. |
| DFD hard | `3335` | known large visual retained | broad image retains visual | unchanged | adds Paddle broad visual as `needs_review` | same/partial | no | Added crop is broad but visual, not plain text. |
| DFD hard | `3344` | labeled diagram retained | partial visual/text separation | unchanged | adds Paddle visual as `needs_review` | same/partial | no | Added crop is uncertain but visual; labeled diagram is not destroyed. |
| DFD hard | `3397` | sparse symbols retained | sparse symbols retained in broad/partial regions | unchanged | unchanged | same | no | Clean classical regions are not fused. |
| DFD hard | `3352` | known candidates retained | broad image regions retained | unchanged | unchanged | same | no | Clean classical region is preserved. |
| DFD hard | `3330` | tall/narrow strips retained | broad image retained | unchanged | unchanged | same | no | Clean classical region is preserved. |
| DFD known-good | `3292` | known-good retained | diagrams retained | unchanged | unchanged | same | no | No Paddle additions emitted. |
| DFD known-good | `3340` | known-good retained | diagram/photo retained | unchanged | unchanged | same | no | No Paddle additions emitted. |

## Corpus 2 Broad/Mixed Results

Improved cases: `4/5`.

- V1 containment improvements: `16_55_48`, `18_29_28`
- V2 additive improvements: `17_19_36`, `18_54_00`
- unchanged: `17_10_58`

`17_10_58` remains unchanged because Paddle's only visual candidate is both
broad and below the additive confidence floor. This is the intended conservative
behavior.

`18_54_00` is counted as improved because the Paddle-added photo-grid region
captures the intended visual content better than the partial classical parent.
It is still marked `needs_review` because it is broad and includes header/text
material.

## DFD Hard Page Results

Incorrect Paddle-derived additions: `0/5`.

V2 emitted review-marked Paddle additions on `3335` and `3344`. These are not
counted as regressions because they are visual candidates, do not remove or
downgrade the classical regions, and are marked `needs_review`. They are not
clean enough to replace classical output.

The other DFD hard pages were unchanged.

## Known-Good Regression Checks

Incorrect Paddle-derived additions: `0/2`.

No Paddle additions were emitted for the known-good regression checks in this
validation run.

## Region Count Check

Region count stayed reasonable. V2 adds at most one Paddle visual region per
unresolved mixed classical parent:

- Corpus 2 max output count in this validation: `2`
- DFD hard max output count in this validation: `2`

No plain-text Paddle additions were observed. Text/title/table labels remain
ineligible for ImageRegion output.

## Gate Result

Required:

- at least `4/5` Corpus 2 broad/mixed cases improve
- `0` incorrect Paddle-derived additions on 5 DFD hard pages
- `0` incorrect Paddle-derived additions on 2 known-good pages
- no explosion in region count
- no obvious plain-text Paddle additions

Result:

- [x] pass
- [ ] fail

Observed:

- Corpus 2 improved count: `4/5`
- DFD incorrect-addition count: `0/5`
- known-good incorrect-addition count: `0/2`
- region-count explosion: no
- plain-text Paddle additions: no

## Decision

V2 additive fusion passes the bounded validation gate as an experimental,
disabled-by-default review-mode option.

It should not become default behavior. The next practical step is to run a
broader review using V2 on Corpus 2 and DFD pages, then decide whether
annotation/ground-truth boxes are needed before more fusion logic.
