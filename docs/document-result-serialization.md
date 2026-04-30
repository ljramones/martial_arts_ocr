# DocumentResult Serialization

## Purpose

`DocumentResult` remains the canonical pipeline output. Its full serialization
keeps backward-compatible region lists while adding review-friendly OCR text
summaries.

## Full Fields Preserved

`DocumentResult.to_dict()` still emits:

- `document_id`
- `source_path`
- `pages`
- `language_hint`
- `detected_languages`
- `confidence`
- `metadata`

`PageResult.to_dict()` still emits:

- `text_regions`
- `image_regions`
- `raw_text`
- `confidence`
- `metadata`

Existing consumers can continue reading `pages[].text_regions[]`.

## Readable Text Summary

Each page now also emits:

```json
{
  "readable_text": "...",
  "text_summary": {
    "raw_text": "...",
    "readable_text": "...",
    "word_count": 123,
    "line_count": 24,
    "line_grouping_method": "adaptive_center_overlap_v1",
    "reading_order_uncertain": false
  }
}
```

The document-level `text_summary` aggregates page counts and readable text.

## Word and Line Regions

Full `text_regions` are preserved, but page serialization now also exposes:

```json
{
  "line_regions": [...],
  "word_regions": [...]
}
```

Line regions are derived from OCR word boxes for readability. Word regions are
kept for geometry, OCR-aware filtering, and future correction tools.

## OCR Diagnostics

Selected OCR boxes become canonical word regions. Alternate OCR candidates, such
as non-selected Tesseract PSM runs, remain compact metadata under:

```text
PageResult.metadata["ocr_alternative_candidates"]
```

These summaries include engine, PSM, confidence, text length, word-box count,
and selected status. Alternate candidate boxes are not promoted into canonical
text regions.

## Artifacts

`WorkflowOrchestrator` still writes:

```text
data.json
page_data.json
page_1.html
text.txt
```

`data.json` includes full canonical serialization plus legacy aliases.
`text.txt` now prefers document readable text when available, falling back to
combined page text.
`page_1.html` now prefers canonical `PageReconstructor` HTML when it is
available, while legacy `html_content` remains available in `data.json` and as
a fallback.

`PageReconstructor` now uses line regions as the default visible text layer for
canonical `DocumentResult` input. Word regions remain serialized for geometry
and debugging, but they are not rendered as visible text by default. When line
regions are unavailable, reconstruction falls back to page `readable_text`, then
`raw_text`, then combined text.

When line grouping marks a page as uncertain, reconstruction metadata and HTML
surface that state so reviewers know the visible text may need ordering review.

## Current Limits

- Word boxes are still present in full `data.json`; the output is not a minimal
  review-only format.
- Multi-column reading order is not solved.
- `page_1.html` is more readable than the word-level debug dump, but remains a
  debug/review artifact, not final reconstruction.
- Japanese analysis has not been promoted into first-class fields.
