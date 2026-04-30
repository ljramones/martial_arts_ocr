# Document Output State — 2026-04-28

## Status

Document output has a stable review/debug baseline.

Current behavior:

- document output prefers `readable_text` where appropriate
- `page_1.html` renders visible line regions instead of every OCR word
- word regions remain available for geometry/debug
- `reading_order_uncertain` is surfaced in reconstruction metadata and HTML
- legacy aliases remain preserved

The current output chain is:

```text
best OCR result
  -> canonical word regions for geometry
  -> adaptive line regions for readable structure
  -> text_summary / readable_text
  -> data.json and text.txt
  -> page_1.html visible line text
  -> reading_order_uncertain surfaced
```

## Current Artifact Contract

`data.json`:

- canonical/full output
- includes `text_summary`, `line_regions`, `word_regions`, and legacy aliases
- preserves `text_regions` for compatibility
- preserves `image_regions` when review-mode extraction is enabled

`page_data.json`:

- page-focused reconstruction data
- contains generated reconstruction elements and reconstruction metadata

`text.txt`:

- readable text artifact
- prefers `readable_text`

`page_1.html`:

- canonical reconstruction HTML when available
- visible text comes from line regions
- shows reading-order uncertainty warning when applicable
- falls back to legacy `html_content` only when canonical reconstruction HTML is unavailable

Region roles:

- `word_regions`: geometry/debug data, not default visible reconstruction text
- `line_regions`: readable visible text structure
- `reading_order_uncertain`: review signal, not a failure state

## data.json

Expected fields include:

```text
text
pages[]
pages[].text_summary
pages[].line_regions
pages[].word_regions
pages[].text_regions
pages[].image_regions
pages[].metadata
```

`data.json["text"]` prefers document readable text. `pages[].text_regions[]`
remains a legacy-compatible full region list. `pages[].line_regions[]` and
`pages[].word_regions[]` are aliases that make the readable/geometry split
explicit.

When image extraction is explicitly enabled, image regions remain available
under `pages[].image_regions[]`.

## page_data.json

`page_data.json` is written from `PageReconstructor`.

It contains:

- page id and title
- reconstruction elements
- HTML fragment
- CSS styles
- page dimensions
- reconstruction metadata

After the line-region reconstruction pass, canonical reconstruction elements
use line regions for visible text. Word regions are not emitted as default
visible text elements.

## text.txt

`text.txt` prefers `readable_text`.

It should not contain duplicated text from alternate Tesseract PSM candidates.
It is a compact text artifact, not full layout reconstruction.

## page_1.html

`page_1.html` is a debug/review artifact.

Current behavior:

- canonical reconstruction HTML is preferred when available
- visible text comes from `line_regions`
- OCR word regions are not rendered as default visible text
- `reading_order_uncertain` appears as a warning when true
- legacy `html_content` remains a fallback

This makes `page_1.html` more readable than the earlier word-level debug dump,
but it is not a polished reproduction of the source page.

## Line Regions vs Word Regions

`word_regions`:

- generated from selected/best OCR result boxes
- represent precise OCR geometry
- used for OCR-aware image filtering, debugging, and future correction/review UI
- remain serialized in `data.json`
- are not default visible reconstruction text

`line_regions`:

- derived from word regions
- represent readable text structure
- carry line grouping metadata
- used for `readable_text`
- used as visible text in canonical `page_1.html`

## reading_order_uncertain Semantics

`reading_order_uncertain=true` means line grouping/order should be reviewed.

It is not an error. It is expected on:

- mixed layouts
- caption-heavy pages
- photo/article pages
- odd visual pages
- pages with large horizontal gaps or uncertain local ordering

The flag should guide review and future reconstruction work. It should not
block artifact generation.

## Stable / Accepted

- Best OCR result is canonical.
- Alternate PSM candidates are diagnostics only.
- Duplicate PSM word inflation is fixed.
- Word regions are retained for geometry.
- Line regions are accepted as readable v1 structure.
- `readable_text` is used for text artifacts.
- `data.json` exposes `text_summary`, line aliases, and word aliases.
- Line-based HTML reconstruction is accepted as v1.
- Legacy aliases remain preserved.

## Known Limitations

- Mixed layouts remain imperfect.
- `page_1.html` is still review/debug quality, not polished reproduction.
- Figure captions and visual-page reading order remain hard.
- Japanese/macron real OCR has not been proven on representative pages.
- Japanese analysis is not promoted into first-class canonical fields.
- Reconstruction does not solve full multi-column/page-layout ordering.
- Image-region display in HTML is still basic.

## Do Not Change Right Now

- Do not reopen image-region extraction.
- Do not build more Paddle fusion.
- Do not rewrite reconstruction broadly.
- Do not promote Japanese analysis before real OCR evidence supports it.
- Do not tune OCR config blindly.
- Do not add per-corpus output formats before the next review need is concrete.

## Future Backlog

- Focused Japanese OCR sample curation/evaluation.
- OCR config/language strategy.
- Line grouping / reading order v2 for mixed layouts.
- `page_1.html` visual polish.
- Review/export workflow.
- Caption association with image regions.
- Multi-column handling.
- Canonical Japanese analysis fields when real OCR evidence supports it.

## Recommended Next Branch

Recommended next branch: focused Japanese OCR sample curation/evaluation.

Real OCR validation has not yet proven reliable Japanese/macron output. Before
promoting Japanese analysis into canonical fields, curate pages that actually
contain Japanese/macrons and evaluate OCR configs against them.

## Relevant Files

- `src/martial_arts_ocr/pipeline/document_models.py`
- `src/martial_arts_ocr/pipeline/orchestrator.py`
- `src/martial_arts_ocr/pipeline/text_normalization.py`
- `src/martial_arts_ocr/reconstruction/page_reconstructor.py`
- `docs/document-result-serialization.md`
- `docs/ocr-output-state-2026-04-28.md`
- `docs/ocr-text-normalization-notes.md`

## Relevant Review Notes

- `data/evaluation/real_page_review/notes/2026-04-28-document-result-serialization-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-page-reconstruction-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-page-reconstruction-line-region-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-readability-and-japanese-sampling.md`
