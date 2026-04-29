# OCR-Aware Filtering Pass

Date: 2026-04-28

## What Changed

Added OCR text-box geometry as an optional signal for image-region candidate
classification. The detector can now compute text overlap, text-mask overlap,
intersecting OCR box counts, and mean OCR confidence for each candidate.

The decision layer remains hybrid:

- visual-only behavior is unchanged when no OCR boxes are supplied
- high OCR overlap plus weak visual evidence rejects candidates as text
- low OCR overlap plus strong visual evidence accepts figure/photo/symbol crops
- partial OCR overlap plus strong visual evidence preserves labeled or mixed
  diagram candidates for review

## Validation Method

Real OCR word/line boxes are not yet consistently exposed by the runtime OCR
processor, so this pass was validated with:

- synthetic OCR boxes in unit tests
- canonical `TextRegion` box plumbing through `ExtractionService`
- Notebook 07 for manual/fake OCR geometry review

No OCR binaries, layout ML models, or corpus-private images are required by the
tests.

## DFD Known-Good Pages

Not reprocessed with real OCR boxes in this pass. With no OCR geometry, the
detector intentionally falls back to the current visual-only behavior from the
broader DFD review.

## Corpus 2 Text False Positives

API-level tests now cover the expected failure mode: a plain paragraph, title
fragment, or vertical text block with high OCR overlap is rejected as text when
OCR boxes are available.

## Broad / Mixed Crops

High OCR overlap with weak visual evidence is rejected. High or moderate OCR
overlap with strong figure/photo/symbol evidence is retained as mixed or
uncertain, preserving labeled diagrams for review instead of treating OCR boxes
as absolute ground truth.

## Runtime Status

Runtime image extraction remains disabled by default. Review-mode extraction can
now pass canonical `TextRegion` geometry or `ocr_text_boxes` metadata into the
detector when those boxes exist. If no boxes exist, no behavior changes.

## Recommended Next Step

Expose OCR word/line boxes from `OCRProcessor` into `DocumentResult` metadata or
`PageResult.text_regions`, then rerun review-mode extraction on the hard DFD and
Corpus 2 pages to measure actual text false-positive reduction with real OCR
geometry.
