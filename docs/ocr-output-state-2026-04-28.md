# OCR Output State — 2026-04-28

## Status

OCR/text output has a stable baseline for the current phase.

The accepted canonical flow is:

```text
selected/best OCR result
  -> canonical word regions
  -> adaptive line regions
  -> readable_text
  -> DocumentResult text_summary
  -> readable data.json / text.txt
```

The selected/best OCR result is canonical. Alternate Tesseract PSM candidates
remain compact diagnostics only. Word regions are preserved for geometry. Line
regions are derived for readability. `readable_text` is now used by `text.txt`
and by `data.json["text"]`. `DocumentResult` exposes `text_summary`,
`line_regions`, and `word_regions` aliases while preserving legacy
`text_regions`.

Extraction architecture is frozen separately in
`docs/extraction-architecture-freeze-2026-04-28.md`. Region extraction and
Paddle fusion remain review-mode only and disabled by default.

## Current OCR/Text Pipeline

```text
OCR engine output
  -> selected/best result
  -> OCRPostProcessor / TextCleaner
  -> pipeline adapter
  -> word TextRegions
  -> adaptive line grouping
  -> readable_text
  -> DocumentResult.to_dict()
  -> data.json / page_data.json / text.txt / page_1.html
```

The current pipeline keeps OCR geometry and readable text separate:

- word `TextRegion` objects represent OCR engine geometry
- line `TextRegion` objects represent derived readable structure
- page `readable_text` is built from derived line regions
- document `text_summary` aggregates readable page text and counts

## Stable / Accepted

- Duplicate PSM word-box inflation is fixed.
- Canonical word regions come from the selected/best OCR result only.
- Alternate PSM candidates are retained as compact metadata diagnostics.
- Cleanup preserves useful line breaks.
- Full-chain tests protect Japanese/macron/punctuation preservation for
  synthetic OCR fixtures.
- Adaptive line grouping is accepted as v1:
  `adaptive_center_overlap_v1`.
- Line metadata records `reading_order_uncertain` when spacing/layout suggests
  review risk.
- `DocumentResult.to_dict()` exposes `text_summary`, `line_regions`, and
  `word_regions` without removing legacy `text_regions`.
- `data.json["text"]` and `text.txt` prefer `readable_text`.
- Legacy artifact aliases remain preserved.

## Current Artifact Shape

`WorkflowOrchestrator` still writes:

```text
data.json
page_data.json
page_1.html
text.txt
```

`data.json`:

- `text` prefers document `readable_text`
- `text_summary` appears at document level
- `pages[].text_summary` appears at page level
- `pages[].line_regions[]` exposes derived readable lines
- `pages[].word_regions[]` exposes OCR word geometry
- `pages[].text_regions[]` remains for compatibility
- `pages[].image_regions[]` remains available when review-mode extraction is
  enabled
- alternate OCR candidates stay compact under page metadata

`text.txt`:

- prefers document `readable_text`
- falls back to combined page text when readable text is unavailable

`page_data.json`:

- remains page-focused structured reconstruction data

`page_1.html`:

- remains useful for debugging/review
- is not yet polished page reconstruction

## How to Inspect Outputs

Useful manual tools:

- `experiments/review_real_ocr_text_quality.py`
- `experiments/run_review_mode_extraction.py`

Generated inspection outputs should remain under ignored paths:

```text
data/notebook_outputs/
data/runtime/
```

Do not commit generated OCR artifacts, runtime outputs, crop outputs, overlays,
private corpus images, or Paddle model files.

## Validation Evidence

Review notes:

- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-text-quality-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-text-quality-review-after-box-selection.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-readability-and-japanese-sampling.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-reading-order-after-line-grouping.md`
- `data/evaluation/real_page_review/notes/2026-04-28-document-result-serialization-review.md`

Summary:

- Initial real OCR review found that canonical text regions were polluted by
  boxes from `best_ocr_result` plus all alternate PSM candidates.
- Canonical OCR box selection fixed duplicate word/line inflation.
- Example improvement:
  - `original_img_3337`: 2386 canonical words -> 598
  - `corpus2_new_doc_2026_04_28_16_55_48`: 621 canonical words -> 110
- Readability/Japanese sampling showed `readable_text` is usable on simple
  text/list pages after deduplication.
- `eng+jpn` ran successfully, but on the sampled pages it produced noisy
  Japanese-like glyphs rather than reliable Japanese terms.
- Real macronized OCR output was not observed in the sampled pages.
- Adaptive line grouping stabilized simple pages and added method/uncertainty
  metadata.
- Serialization polish made `data.json` and `text.txt` easier to inspect.

## Known Limitations

- Mixed layouts still have `reading_order_uncertain`.
- Line grouping is not full page-layout reconstruction.
- Figure captions, article/photo pages, and odd visual pages remain hard.
- Japanese OCR is not yet reliable on sampled real pages.
- Macronized real OCR has not yet been observed.
- Japanese analysis is not promoted into first-class canonical fields.
- Word boxes are still present in full `data.json`, so output is readable but
  not minimal.
- `page_1.html` remains a debugging artifact, not polished reconstruction.

## Do Not Change Right Now

- Do not reopen image-region extraction.
- Do not build more Paddle fusion logic.
- Do not promote Japanese analysis before real OCR text is reliable enough.
- Do not rewrite `PageReconstructor` without a focused review.
- Do not tune OCR config blindly.
- Do not add per-corpus OCR presets without evidence from a focused validation
  pass.

## Future Backlog

- Focused Japanese OCR evaluation with better-selected samples.
- OCR engine config / PSM / language selection strategy.
- Line grouping and reading-order v2 for mixed layouts.
- Page reconstruction from `line_regions` and `image_regions`.
- Japanese analysis promotion into canonical model fields.
- Review/export workflow for correcting OCR text and boxes.
- Annotation data for OCR/layout correction.

## Recommended Next Branches

1. Page reconstruction review

   Assess `page_1.html` and decide how to use `line_regions` and
   `image_regions` in a more useful review artifact.

2. Focused Japanese OCR sampling

   Find or curate pages likely to contain Japanese/macrons and evaluate OCR
   language/config choices.

3. Review/export workflow

   Make `DocumentResult` outputs easier to inspect, correct, annotate, and
   round-trip into future training/evaluation data.

## Recommended Immediate Next Step

Run a page reconstruction review.

`data.json` and `text.txt` are now useful enough that `page_1.html` is the next
user-facing artifact likely to expose remaining problems. The next pass should
inspect the current HTML output, document what it uses from `DocumentResult`,
and decide whether a narrow reconstruction improvement is warranted.

## Relevant Files

- `src/martial_arts_ocr/ocr/processor.py`
- `src/martial_arts_ocr/ocr/postprocessor.py`
- `src/martial_arts_ocr/pipeline/adapters.py`
- `src/martial_arts_ocr/pipeline/document_models.py`
- `src/martial_arts_ocr/pipeline/text_normalization.py`
- `src/martial_arts_ocr/pipeline/orchestrator.py`
- `src/martial_arts_ocr/reconstruction/page_reconstructor.py`
- `experiments/review_real_ocr_text_quality.py`
- `experiments/run_review_mode_extraction.py`
- `docs/ocr-text-quality-assessment.md`
- `docs/ocr-text-normalization-notes.md`
- `docs/document-result-serialization.md`

## Relevant Review Notes

- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-text-quality-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-text-quality-review-after-box-selection.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-readability-and-japanese-sampling.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-reading-order-after-line-grouping.md`
- `data/evaluation/real_page_review/notes/2026-04-28-document-result-serialization-review.md`
