# Page Reconstruction Review

## Purpose

Assess current `page_1.html` and `PageReconstructor` behavior using the current `DocumentResult` shape.

This is an assessment-only review. No reconstruction, OCR, extraction, serialization, or runtime behavior was changed.

## Files Reviewed

- `src/martial_arts_ocr/reconstruction/page_reconstructor.py`
- `src/martial_arts_ocr/pipeline/document_models.py`
- `src/martial_arts_ocr/pipeline/orchestrator.py`
- `src/martial_arts_ocr/pipeline/adapters.py`
- `src/martial_arts_ocr/pipeline/text_normalization.py`
- `docs/ocr-output-state-2026-04-28.md`
- `docs/document-result-serialization.md`
- `docs/review-mode-extraction-guide.md`
- `docs/extraction-architecture-freeze-2026-04-28.md`
- `data/evaluation/real_page_review/notes/2026-04-28-document-result-serialization-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-reading-order-after-line-grouping.md`

## Artifacts Inspected

Existing ignored artifacts were inspected under:

```text
data/notebook_outputs/document_result_serialization_review/
data/notebook_outputs/ocr_text_quality_review/
data/notebook_outputs/ocr_text_readability_sampling/
```

Representative outputs:

| Output | Source Page | Notes |
|---|---|---|
| `document_result_serialization_review/doc_920001` | `original_img_3288` | simple/list-like DFD page |
| `document_result_serialization_review/doc_920002` | `original_img_3337` | dense DFD text page |
| `document_result_serialization_review/doc_920003` | Corpus 2 `16_55_48` | mixed/noisy layout |
| `ocr_text_readability_sampling/eng_auto/doc_920003` | `original_img_3344` | noisy labeled diagram page |

Generated artifacts were inspected only. They remain ignored and should not be committed.

## Current PageReconstructor Inputs

`WorkflowOrchestrator._write_artifacts()` calls:

```text
PageReconstructor.reconstruct_page(document_result, image_path)
```

for canonical pipeline output. The canonical path accepts:

```text
DocumentResult
PageResult
PageResult.text_regions
PageResult.image_regions
PageResult.combined_text()
TextRegion.bbox
ImageRegion.bbox
region metadata
```

It does not directly use these newer convenience fields:

```text
DocumentResult.text_summary()
PageResult.text_summary()
PageResult.line_regions()
PageResult.word_regions()
PageResult.metadata["readable_text"]
PageResult.metadata["reading_order_uncertain"]
```

Because derived line regions and OCR word regions both live in `PageResult.text_regions`, the reconstructor currently creates elements for both unless the input has already been filtered.

For legacy `ProcessingResult`, `PageReconstructor` has a separate older path that uses:

```text
best_ocr_result.bounding_boxes
text_regions
cleaned_text
extracted_images
japanese_result
text_statistics
overall_confidence
quality_score
```

## Current page_1.html Output

There are two different artifact paths to distinguish:

1. `page_data.json`

   This is written from `PageReconstructor.reconstruct_page(...).to_dict()`.

2. `page_1.html`

   `WorkflowOrchestrator` prefers legacy `DocumentResult.metadata["legacy"]["html_content"]` when present. Only if that is missing does it fall back to `page_data.json["html_content"]`.

In the inspected real OCR outputs, `page_1.html` came from legacy HTML content. It rendered an `ocr-content` / `text-content` view rather than the canonical `PageReconstructor` element output.

Observed `page_1.html` behavior:

- displays OCR text as paragraphs / line breaks
- generally uses cleaned/readable text from the legacy path
- does not visibly expose `text_summary`
- does not visibly expose `line_regions` versus `word_regions`
- does not visibly expose `reading_order_uncertain`
- does not visibly expose `mixed_region` / `needs_review`
- does not visibly expose Paddle fusion metadata
- did not clearly display image region crops in the inspected samples

Observed `page_data.json` behavior:

- contains canonical reconstruction elements
- includes both line-derived text elements and word text elements
- includes image elements when image regions are present
- preserves region metadata inside element metadata
- records basic reconstruction metadata

Examples:

```text
doc_920001:
  line_regions: 10
  word_regions: 56
  image_regions: 2
  page_data elements: 68

doc_920003:
  line_regions: 22
  word_regions: 110
  image_regions: 2
  page_data elements: 134
  reading_order_uncertain: true in data.json
```

The high element counts show the current canonical reconstruction is still too word-heavy for review.

## Use of Current DocumentResult Shape

Current usage:

- `text_regions`: used directly
- `image_regions`: used directly
- region bbox: used for element placement
- region metadata: copied into element metadata
- `combined_text()`: fallback only

Current gaps:

- `text_summary` is ignored
- `readable_text` is ignored by the canonical reconstructor
- `line_regions` are not preferred over word regions
- `word_regions` are not treated as optional geometry/debug detail
- `reading_order_uncertain` is not surfaced in HTML
- image region metadata is not presented as review metadata
- mixed/needs-review/Paddle metadata is not presented in HTML

## Review Artifact Usefulness

Classification: `debug-only`.

Reason:

- `data.json` and `text.txt` are now better review artifacts than `page_1.html`.
- `page_1.html` often reflects legacy HTML rather than canonical reconstruction.
- The canonical `page_data.json` reconstruction is useful for debugging element generation, but it currently includes both line and word elements.
- The HTML does not surface the metadata that matters most for review: line/word distinction, reading-order uncertainty, image-region status, mixed/needs-review status, or Paddle fusion provenance.
- It is not yet a reliable visual review artifact for mixed layouts or extracted image regions.

## Gaps

- `page_1.html` artifact source is ambiguous: legacy HTML takes priority over `PageReconstructor` output.
- Canonical reconstruction should prefer line regions for readable text display.
- Word regions should remain available as optional geometry/debug overlays, not the default readable layer.
- Reading-order uncertainty should be visible in the HTML metadata or warning area.
- Image regions/crops should be visible or clearly listed when review-mode extraction is enabled.
- Mixed/needs-review and layout-fusion metadata should be visible for review-mode pages.
- The current canonical HTML fragment is not a complete review UI and does not explain what the reviewer is seeing.

## Recommended Next Implementation Pass

Primary recommendation: use line regions instead of raw/full `text_regions` for canonical page reconstruction.

Narrow implementation shape:

```text
PageReconstructor canonical path:
  prefer PageResult.line_regions() for visible text elements
  keep word regions out of default visible HTML
  preserve word count / word geometry in metadata only
  surface reading_order_uncertain in reconstruction metadata and HTML
```

Why this first:

- The current model already distinguishes word geometry from line readability.
- `data.json` and `text.txt` now prove the line hierarchy is inspectable.
- Rendering every word as a visible element makes canonical reconstruction noisy.
- This is a smaller and safer fix than rewriting page layout reconstruction.

## Backup Recommendation

If the team does not want to touch reconstruction behavior yet, add a clearer review metadata panel first:

```text
page_1.html:
  show text_summary counts
  show reading_order_uncertain
  show image region count
  show mixed/needs_review count
  show whether HTML came from legacy or canonical reconstruction
```

This would make the artifact more honest without attempting layout improvements.

## Do Not Do Yet

- Do not rewrite `PageReconstructor` wholesale.
- Do not attempt full multi-column layout reconstruction yet.
- Do not reopen image-region extraction or Paddle fusion.
- Do not promote Japanese analysis into canonical fields as part of reconstruction.
- Do not tune OCR configuration as part of this pass.
- Do not make word boxes the default visible HTML layer.

## Evidence / Notes

- `WorkflowOrchestrator` writes `page_data.json` from `PageReconstructor`, but `page_1.html` prefers legacy `html_content` when available.
- `PageReconstructor._reconstruct_canonical_result()` iterates over `page.text_regions`, which includes both derived line regions and OCR word regions.
- `PageResult.line_regions()` and `PageResult.word_regions()` now exist, but the reconstructor does not use those accessors.
- `DocumentResult.text_summary()` and `PageResult.text_summary()` now exist, but reconstruction does not use them.
- In inspected outputs, `page_data.json` had much higher element counts than visible readable lines because word regions were included as elements.
- `data.json` now provides a better source of review truth than `page_1.html`; reconstruction should catch up to that artifact shape next.
