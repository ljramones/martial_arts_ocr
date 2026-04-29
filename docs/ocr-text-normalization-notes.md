# OCR Text Normalization Notes

## Purpose

OCR engines can provide precise word geometry, but word boxes alone are not a
readable document structure. The current normalization layer keeps word boxes
for geometry while deriving line-level `TextRegion` objects for readability.

## Hierarchy

Word regions:

- source: OCR engine output
- metadata `source=ocr_engine`
- metadata `ocr_level=word`
- used for geometry, OCR-aware image filtering, and future correction tools

Line regions:

- source: deterministic normalization from word regions
- metadata `source=ocr_normalization`
- metadata `ocr_level=line`
- used for readable page text, `data.json` review, and future reconstruction

Readable page text:

- stored in `PageResult.metadata["readable_text"]`
- built by sorting line regions by reading order
- preserves line breaks between derived lines

## Line Grouping

Line grouping now uses an adaptive geometry rule:

- candidate words are ordered by vertical center and x position
- default y tolerance is based on median word height
- words can join a line by close vertical center or strong y-overlap
- words within a line are sorted left-to-right
- unusually large x gaps are preserved with extra spacing and marked as
  `reading_order_uncertain`

Derived line metadata includes:

```text
line_grouping_method=adaptive_center_overlap_v1
reading_order_uncertain=true|false
```

This improves simple paragraph/list/caption readability without attempting full
multi-column reconstruction.

## Preservation Rules

Japanese, macrons, and martial-arts punctuation must survive normalization:

```text
武道
柔術
koryū
budō
Daitō-ryū
ō
ū
—
・
「」
```

The line grouping helper joins word text without Unicode normalization or ASCII
conversion. Cleanup safety is still the responsibility of OCR post-processing
tests.

## Full Cleanup Chain Validation

The cleanup chain now has fixture coverage for:

```text
OCRPostProcessor
  -> TextCleaner
  -> OCR adapters
  -> TextRegion hierarchy
  -> DocumentResult.to_dict()
```

The tests assert that the preservation tokens above survive the full chain,
including macronized romanization, Japanese quotes, the interpunct, em dash, and
the Japanese long vowel mark `ー`.

The postprocessor now collapses repeated spaces/tabs during general correction
without flattening newlines. Final whitespace cleanup still normalizes repeated
blank lines, so useful line breaks survive while excessive whitespace is reduced.

Serialization expectations:

- word regions remain available for geometry
- derived line regions remain available for readability
- `PageResult.metadata["readable_text"]` preserves line-oriented text
- `DocumentResult.to_dict()` keeps both compact readable metadata and OCR box
  metadata

## Selected OCR Result as Canonical Text Source

When an OCR run produces multiple candidates, such as Tesseract PSM `11`, `3`,
and `6`, canonical word regions now come from the selected/best OCR result only.

```text
best_ocr_result.bounding_boxes
  -> word TextRegions
  -> derived line TextRegions
  -> readable_text
```

Alternate OCR candidates are retained as compact diagnostics in
`PageResult.metadata["ocr_alternative_candidates"]`. Their boxes are not mixed
into canonical `text_regions`, which prevents repeated words in
`readable_text`.

## Current Limits

- Grouping is still local and line-oriented, not full page layout analysis.
- Multi-column reading order is not solved.
- Vertical Japanese layout is not solved.
- Word boxes are still serialized alongside line regions, so `data.json` can
  remain verbose.
- This is not final page reconstruction.
- The tests use synthetic OCR outputs. Real DFD/Corpus 2 OCR text still needs a
  review fixture pass.
- Real Japanese/macron preservation still needs a language-enabled OCR sampling
  pass; current real-page validation used English-only OCR.

## Next Work

The next pass should evaluate whether `data.json` and `page_1.html` expose the
line hierarchy clearly enough for review before promoting Japanese analysis or
changing reconstruction.
