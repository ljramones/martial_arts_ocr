# Mixed-Region Refinement Pass

Date: 2026-04-28

## Scope

This pass added a disabled-by-default mixed-region refinement layer and tested it
against the same empirical set used for OCR-aware real-box validation:

- 5 DFD hard pages
- 18 Corpus 2 sampled pages

Refinement used real Tesseract word boxes generated with `eng+jpn` and `--psm 6`.
No runtime defaults were changed.

## Synthetic Test Summary

Synthetic tests cover:

- broad figure-left / paragraph-right crop refinement
- embedded mixed crop downgrade to `needs_review`
- labeled diagram preservation
- plain text rejection before refinement
- clean low-OCR-overlap photo no-op
- high OCR overlap with weak visual evidence no refined bbox
- metadata serialization
- no-OCR no-op
- disabled-refinement no-op

The labeled-diagram preservation test requires the emitted bbox to contain all
labels and connected diagram components.

## Empirical Gate Result

Gate status: pass.

Requirements:

- DFD known visuals retained: `5/5`
- DFD text false positives: `0`
- No labeled diagram destroyed: passed; DFD `3344` was preserved as
  `mixed_labeled`
- No net loss of true visual content: passed; accepted region counts stayed
  stable
- At least 3 of 5 Corpus 2 broad/mixed crops refined or downgraded:
  passed; `5/5` were downgraded to `needs_review`

Important caveat: no real-page Corpus 2 crop was tightened in this pass. The
empirical success comes from correctly flagging broad/mixed crops for review,
not from automatically producing better crop boundaries.

## Summary Table

| Corpus | Pages | Accepted Before | Accepted After | Text FP Before | Text FP After | Broad/Mixed Before | Broad/Mixed After | Refined | Needs Review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFD hard pages | 5 | 8 | 8 | 0 | 0 | 1 | 1 | 0 | 1 |
| Corpus 2 sample | 18 | 8 | 8 | 2 | 0 | 5 | 5 | 0 | 5 |

## Corpus 2 Broad/Mixed Crop Disposition

| Sample ID | Original BBox | Result BBox | Disposition | Reason |
|---|---|---|---|---|
| `corpus2_new_doc_2026_04_28_16_55_48` | `x=53, y=39, w=351, h=391` | unchanged | downgraded to `needs_review` | `ocr_labels_interspersed_with_visual_mass` |
| `corpus2_new_doc_2026_04_28_17_10_58` | `x=108, y=102, w=355, h=148` | unchanged | downgraded to `needs_review` | `ocr_labels_interspersed_with_visual_mass` |
| `corpus2_new_doc_2026_04_28_17_19_36` | `x=96, y=10, w=359, h=400` | unchanged | downgraded to `needs_review` | `ocr_labels_interspersed_with_visual_mass` |
| `corpus2_new_doc_2026_04_28_18_29_28` | `x=77, y=44, w=317, h=472` | unchanged | downgraded to `needs_review` | `ocr_labels_interspersed_with_visual_mass` |
| `corpus2_new_doc_2026_04_28_18_54_00` | `x=53, y=44, w=139, h=170` | unchanged | downgraded to `needs_review` | `ocr_labels_interspersed_with_visual_mass` |

## DFD Hard Page Results

| Sample ID | Accepted Before | Accepted After | Text FP After | Notes |
|---|---:|---:|---:|---|
| `original_img_3335` | 1 | 1 | 0 | Known large visual retained. |
| `original_img_3344` | 1 | 1 | 0 | Labeled diagram retained and marked `mixed_labeled`; no bbox tightening. |
| `original_img_3397` | 2 | 2 | 0 | Sparse/symbol regions retained. |
| `original_img_3352` | 2 | 2 | 0 | Known candidates retained. |
| `original_img_3330` | 2 | 2 | 0 | Tall/narrow visual strips retained. |

## Newly Rejected Candidates

No new accepted-region removals were observed from mixed-region refinement.
Refinement operates on accepted candidates and either tightens, preserves, or
marks them for review.

## Interpretation

The refiner is conservative. On the real-page sample, it did not automatically
split broad Corpus 2 parent crops, but it did mark all five broad/mixed cases as
`needs_review` with stable metadata while preserving DFD visual content.

This is useful for review-mode triage but is not yet a crop-quality improvement.
The remaining automatic crop-boundary problem still likely needs a document
layout backend or a more explicit annotation/evaluation workflow.

## Recommendation

Keep mixed-region refinement disabled by default. It is safe as a review-mode
diagnostic flagger, but not sufficient as an automatic crop-refinement solution.

Recommended next step: configure and evaluate one real document-layout backend
against these same broad/mixed cases, or add manual annotation ground truth
before attempting more refinement heuristics.
