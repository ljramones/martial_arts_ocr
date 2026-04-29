# OCR-Aware Real-Box Validation

Date: 2026-04-28

## Purpose

Compare visual-only image extraction against OCR-aware image extraction using
actual OCR engine boxes.

## OCR Environment

- OCR engine used: Tesseract via `pytesseract.image_to_data`
- OCR binary/model availability: Tesseract 5.5.2 available locally
- Languages used: `eng+jpn`
- Tesseract page segmentation mode: `--psm 6`
- Were real word/line boxes available? yes
- Box levels used: word
- Notes: line/block aggregation was not generated in this pass. The validation
  used true Tesseract word boxes, not fake or layout-derived text regions.

## Summary Table

| Corpus | Pages Reviewed | Text FP Before | Text FP After | Known Visuals Retained Before | Known Visuals Retained After | Broad/Mixed Crops Before | Broad/Mixed Crops After | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DFD hard pages | 5 | 0 | 0 | 5/5 pages | 5/5 pages | 2 | 2 | OCR-aware filtering did not regress known hard DFD visual regions. |
| Corpus 2 hard/generalization pages | 18 | 1 | 1 | no visual regressions observed | no visual regressions observed | 5 | 5 | One child candidate was suppressed, but a broad/mixed parent crop remained. |

## Method

For each page, the current classical detector was run twice:

1. Visual-only: no OCR boxes were passed to `LayoutAnalyzer`.
2. OCR-aware: real Tesseract word boxes were passed as `ocr_text_boxes`.

Counts are assisted metrics from detector diagnostics. They should be treated as
review counts, not fully automatic ground truth.

## DFD Results

| Sample ID | OCR Boxes | Accepted Before | Accepted After | Text FP Before | Text FP After | Broad/Mixed Before | Broad/Mixed After | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `original_img_3335` | 219 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged |
| `original_img_3344` | 705 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged; labeled diagram retained as mixed |
| `original_img_3397` | 72 | 2 | 2 | 0 | 0 | 0 | 0 | unchanged |
| `original_img_3352` | 192 | 2 | 2 | 0 | 0 | 0 | 0 | unchanged |
| `original_img_3330` | 116 | 2 | 2 | 0 | 0 | 0 | 0 | unchanged |

### DFD Observations

- Known visual regions remained present on all five hard pages.
- `3344` retained the labeled diagram with `labeled_diagram_ocr_rescue`.
- Tall/narrow visual strips on `3330` remained retained with low OCR overlap.
- OCR-aware filtering did not reduce broad crops on `3335` or `3344`, but it
  also did not damage those known visual detections.

## Corpus 2 Results

| Sample ID | OCR Boxes | Accepted Before | Accepted After | Text FP Before | Text FP After | Broad/Mixed Before | Broad/Mixed After | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `corpus2_image12342` | 94 | 1 | 1 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_16_55_48` | 81 | 2 | 1 | 1 | 1 | 1 | 1 | partial improvement |
| `corpus2_new_doc_2026_04_28_16_56_38` | 55 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_16_59_35` | 115 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_17_03_18` | 29 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_17_10_58` | 94 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged; mixed crop retained |
| `corpus2_new_doc_2026_04_28_17_19_36` | 101 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged; mixed crop retained |
| `corpus2_new_doc_2026_04_28_18_16_34` | 331 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_18_29_28` | 147 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged; broad visual/mixed crop retained |
| `corpus2_new_doc_2026_04_28_18_40_35` | 229 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_18_54_00` | 63 | 1 | 1 | 0 | 0 | 1 | 1 | unchanged; photo-grid recall still partial |
| `corpus2_new_doc_2026_04_28_19_43_28` | 31 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_19_55_36` | 169 | 1 | 1 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_20_05_24` | 149 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_20_17_15` | 93 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_new_doc_2026_04_28_20_26_02` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_unknown_3` | 98 | 0 | 0 | 0 | 0 | 0 | 0 | unchanged |
| `corpus2_unknown` | 177 | 1 | 1 | 0 | 0 | 0 | 0 | unchanged |

### Corpus 2 Observations

- OCR-aware filtering suppressed one accepted child region on
  `corpus2_new_doc_2026_04_28_16_55_48`, reducing accepted candidates from 2 to
  1.
- The remaining accepted crop on that same page was still broad/mixed with high
  OCR overlap and strong visual evidence, so it was retained as uncertain rather
  than rejected.
- Broad/mixed crop counts did not improve overall.
- Photo-grid recall did not materially improve; OCR geometry is a suppression
  signal, not a recall mechanism.

## Failure Patterns

- OCR-aware filtering is conservative around mixed visual/text regions. This
  protects labeled diagrams, but it also preserves broad mixed crops when visual
  evidence is strong.
- Real OCR boxes do not solve photo-grid false negatives.
- Box-level OCR overlap is useful for suppressing obvious text child crops, but
  the current fusion logic does not split or refine broad parent regions.

## Regressions

- No DFD hard-page visual regressions were observed.
- No Corpus 2 visual regressions were observed in this run.
- No known labeled-diagram regression was observed.

## Does OCR-Aware Filtering Solve The Current Problem?

Partially. The real-box path works and is safe on the reviewed DFD hard pages,
but it did not materially reduce Corpus 2 broad/mixed crops. It helps suppress
obvious OCR-overlapping text candidates, but the current remaining problem is
broader: parent crop refinement, crop splitting, and semantic fusion.

## Recommendation

- [ ] OCR-aware filtering is validated and should become the preferred review-mode path
- [x] OCR-aware filtering helps but needs fusion refinements
- [ ] OCR-aware filtering is inconclusive because OCR boxes were unavailable/poor
- [ ] OCR-aware filtering caused regressions and should remain experimental

Recommended next pass: add a fusion/refinement layer that can downgrade or split
broad mixed parent crops using OCR mask coverage, while preserving labeled
diagrams and known DFD visual regions. Layout-model comparison remains useful
for photo-grid recall because OCR-aware suppression alone does not add missing
visual candidates.
