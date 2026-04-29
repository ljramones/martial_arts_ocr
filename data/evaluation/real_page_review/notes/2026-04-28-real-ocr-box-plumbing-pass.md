# Real OCR Box Plumbing Pass

Date: 2026-04-28

## What Changed

OCR engine geometry can now survive into the canonical document model:

- Tesseract-style word boxes
- Tesseract TSV-style dict rows
- EasyOCR polygon outputs
- generic `words`, `lines`, `blocks`, and `ocr_text_boxes`
- existing `TextRegion`-like objects with bboxes

Converted boxes become canonical `TextRegion` objects with provenance metadata:

```text
source: ocr_engine
engine: tesseract | easyocr | paddleocr | fake_test | unknown
ocr_level: word | line | block
```

The page also preserves normalized OCR boxes under
`PageResult.metadata["ocr_text_boxes"]`.

## Runtime Availability

The Tesseract and EasyOCR wrappers already emit bounding boxes when those engines
run. This pass adds provenance metadata in those engine outputs and promotes the
boxes through `OCRProcessor.process_to_document_result()`.

No OCR binaries are required for normal tests. Validation used fake OCR output
and adapter-level tests.

## ExtractionService Behavior

`ExtractionService` now prefers true OCR-engine boxes over layout-derived text
boxes when passing geometry into OCR-aware image filtering. If no OCR boxes are
available, review-mode image extraction falls back to the visual-only detector.

Image extraction remains disabled by default.

## Remaining Validation Needed

The next pass should run actual review-mode processing with real OCR boxes on:

- DFD hard pages
- Corpus 2 hard pages

The review should compare plain-text false positives, known diagram retention,
broad/mixed crops, and labeled-diagram regressions before and after OCR-aware
suppression.
