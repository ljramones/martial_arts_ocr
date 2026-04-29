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

## Current Limits

- Grouping is simple: sort by vertical center, then horizontal position.
- Multi-column reading order is not solved.
- Vertical Japanese layout is not solved.
- Word boxes are still serialized alongside line regions, so `data.json` can
  remain verbose.
- This is not final page reconstruction.

## Next Work

The next pass should validate the full OCR cleanup chain on fixture text before
promoting Japanese analysis or changing page reconstruction.
