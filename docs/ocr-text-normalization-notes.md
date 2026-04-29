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

## Current Limits

- Grouping is simple: sort by vertical center, then horizontal position.
- Multi-column reading order is not solved.
- Vertical Japanese layout is not solved.
- Word boxes are still serialized alongside line regions, so `data.json` can
  remain verbose.
- This is not final page reconstruction.
- The tests use synthetic OCR outputs. Real DFD/Corpus 2 OCR text still needs a
  review fixture pass.

## Next Work

The next pass should use the validated cleanup/hierarchy path on representative
real OCR outputs before promoting Japanese analysis or changing page
reconstruction.
