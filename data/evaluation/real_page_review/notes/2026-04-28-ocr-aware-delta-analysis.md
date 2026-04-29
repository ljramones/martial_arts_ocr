# OCR-Aware Delta Analysis

Date: 2026-04-28

## Purpose

Explain why Corpus 2 accepted regions changed from 9 to 8 during OCR-aware
real-box validation.

## Dropped Candidate

- Sample ID: `corpus2_new_doc_2026_04_28_16_55_48`
- Input path:
  `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg`
- Visual-only candidate bbox: `x=220, y=257, width=161, height=128`
- OCR-aware result: candidate rejected with `high_ocr_text_overlap`
- Classification: `false_positive_removed`

OCR diagnostics:

- `ocr_text_mask_overlap_ratio`: `0.6038`
- `ocr_text_area_ratio`: not separately recorded in the prior summary output
- `ocr_text_box_count`: not separately recorded in the prior summary output
- `ocr_text_confidence_mean`: not separately recorded in the prior summary output
- `ocr_text_overlap_ratio`: `0.5608`

Visual diagnostics:

- `figure_like_score`: `0.43`
- `photo_like_score`: `0.58`
- `sparse_symbol_score`: `0.0`
- `crop_quality_score`: `0.75`
- `text_like_score`: `0.93`
- visual score: `0.58`

Notes:

- The dropped candidate had high OCR mask overlap and high text-like score with
  only borderline photo evidence.
- The candidate was a child region inside the larger accepted parent crop on the
  same page.
- The broader parent crop remained accepted as uncertain/mixed, so the page-level
  broad-crop problem was not solved.
- This is not evidence of a true visual regression. It is a conservative removal
  of an OCR-overlapping text-like child crop.

## Conclusion

- Did OCR-aware filtering cause a true visual regression? No evidence of a true
  visual regression was found for the 9 -> 8 delta.
- Should OCR-aware filtering remain available in review-mode experiments? Yes.
  It safely removed one likely text false-positive child crop in this sample.
- Is an immediate bug fix needed before further work? No. The remaining problem
  is not this dropped candidate; it is broad/mixed parent crops that remain
  accepted when visual evidence is strong.

## Diagnostic Gap

The prior validation summary preserved OCR overlap and mask overlap, but did not
persist per-candidate OCR text area ratio, OCR box count, or mean OCR confidence
in the reduced JSON summary. The detector can emit those diagnostics; future
review scripts should retain the full `diagnostic_features` object for accepted
and rejected candidates.
