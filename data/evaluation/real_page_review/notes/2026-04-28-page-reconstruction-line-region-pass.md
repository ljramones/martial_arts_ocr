# Page Reconstruction Line-Region Pass

## Purpose

Validate the first narrow reconstruction improvement: canonical page reconstruction uses line regions as visible text and keeps OCR word regions as geometry/debug data.

## Change Summary

- Canonical `PageReconstructor` visible text now prefers `PageResult.line_regions()`.
- If line regions are unavailable, reconstruction falls back to `readable_text`, then `raw_text`, then combined text.
- Word regions are not rendered as default visible text elements.
- Reconstruction metadata records:
  - `visible_text_source`
  - `ocr_word_count`
  - `ocr_line_count`
  - `reading_order_uncertain`
- Canonical HTML surfaces a reading-order warning when uncertainty is true.
- `WorkflowOrchestrator` now prefers canonical reconstruction HTML for `page_1.html` when available, while keeping legacy HTML as fallback.

## Synthetic Test Coverage

Tests cover:

- line regions are visible text when both line and word regions exist
- word-only debug text is not rendered visibly
- `readable_text` fallback
- `raw_text` fallback
- `reading_order_uncertain` metadata and HTML warning
- image regions remain represented as image elements
- `page_1.html` prefers canonical reconstruction over legacy HTML when canonical reconstruction is available

## Real-Output Check

Generated ignored outputs under:

```text
data/notebook_outputs/page_reconstruction_line_region_pass/
```

These outputs were inspected only and should not be committed.

| Sample ID | Output | Line Regions | Word Regions | Text Regions | Page Data Elements | Visible Text Elements | Image Elements | Visible Text Source | Reading Order Uncertain |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `original_img_3288` | `doc_920001` | 10 | 56 | 66 | 12 | 10 | 2 | `line_regions` | false |
| `corpus2_new_doc_2026_04_28_16_55_48` | `doc_920002` | 22 | 110 | 132 | 24 | 22 | 2 | `line_regions` | true |

## Observations

- `page_data.json` no longer emits a visible text element for every OCR word.
- Visible text element count now matches line-region count on the checked pages.
- Word regions remain present in `data.json` as serialized geometry.
- `page_1.html` now uses canonical reconstruction HTML in these checked outputs.
- The mixed Corpus 2 page shows the reading-order warning in canonical HTML.
- Image elements remain present.

## Remaining Limits

- The HTML is still a debug/review artifact, not polished reconstruction.
- The current image element rendering is basic.
- Mixed-layout reading order remains imperfect; it is now surfaced rather than hidden.
- Legacy HTML remains as fallback, not as the preferred canonical artifact when reconstruction succeeds.

## Decision

The pass succeeds as a narrow artifact-usability improvement. It makes `page_1.html` more consistent with the current `DocumentResult` hierarchy without changing OCR, extraction, or runtime defaults.
