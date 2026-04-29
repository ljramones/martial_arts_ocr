# OCR-Aware Image Region Filtering

## Why OCR Geometry Matters

The classical detector is useful as a fast proposal generator, but Corpus 2
showed that visual-only scoring can still confuse dense text, title fragments,
and mixed text/image crops with figures. OCR text boxes add the missing document
layout signal: they show where recognized text actually sits on the page.

OCR is not treated as perfect ground truth. Labels, captions, callouts, arrows,
and Japanese annotations may legitimately overlap a diagram. OCR overlap is a
strong diagnostic signal, not an automatic rejection rule.

## Geometry Convention

`utils/text/geometry.py` uses explicit `(x, y, width, height)` boxes. The module
normalizes common dict/object shapes at the boundary and exposes helpers for:

- candidate area covered by OCR text boxes
- OCR text area covered by a candidate
- intersecting OCR box count
- mean OCR confidence
- OCR text mask construction
- candidate overlap with the OCR text mask

Word boxes and line boxes are both preserved when an OCR engine exposes them.
Word boxes are preferred for text masks because their smaller footprint handles
labeled diagrams better than coarse paragraph/block boxes.

Each canonical OCR text region should preserve provenance:

```text
source: ocr_engine
engine: tesseract | easyocr | paddleocr | fake_test
ocr_level: word | line | block
```

Boxes derived from layout/image-region utilities can still travel through
`TextRegion`, but they are not the preferred suppression signal.

## Detector Behavior

When OCR boxes are supplied, candidate diagnostics include:

- `ocr_text_overlap_ratio`
- `ocr_text_area_ratio`
- `ocr_text_box_count`
- `ocr_text_confidence_mean`
- `ocr_text_mask_overlap_ratio`

Filtering combines these with existing visual scores:

- high OCR mask/area overlap plus weak visual evidence rejects the candidate as
  text
- low OCR overlap plus strong visual evidence accepts it as an image/figure
- moderate OCR overlap plus strong visual evidence preserves labeled diagrams
- high OCR overlap plus strong visual evidence marks the region as mixed or
  uncertain for review

`ocr_text_mask_overlap_ratio` is the primary suppression signal because the
dilated mask tolerates OCR jitter better than raw boxes. Box count is diagnostic
only: many small labels inside a diagram must not suppress the figure by itself.
Low OCR confidence should reduce confidence in suppression decisions during
future tuning.

When OCR boxes are absent, the detector uses the existing visual-only behavior.

## Runtime Status

Image extraction remains disabled by default. Review-mode extraction can pass
available canonical `TextRegion` boxes or `ocr_text_boxes` metadata into the
detector. If OCR output does not expose boxes, extraction falls back to the
current visual-only path.

The OCR adapters currently support Tesseract TSV-style rows, EasyOCR polygon
tuples, generic `words`/`lines`/`blocks` containers, and existing
`ocr_text_boxes` metadata.

## Mixed-Region Refinement

OCR-aware filtering passed the safety check on DFD hard pages, but it did not
reduce broad/mixed parent crops by itself. The optional mixed-region refiner uses
OCR-mask subtraction as a crop-cleanup signal:

1. Build a saliency mask inside an accepted candidate.
2. Subtract the OCR text mask.
3. Reconnect remaining visual mass.
4. Refine the bbox only when a coherent visual component can be safely isolated.
5. Preserve interspersed labels and mark the crop as `mixed_labeled` instead of
   over-cropping diagrams.

The refiner remains disabled unless explicitly enabled with
`region_enable_mixed_region_refinement=True`. Unresolved broad crops are marked
`needs_review` with `mixed_region` metadata. If corpus-level review does not
improve with this layer, the next step is a real document-layout backend
comparison rather than more heuristic tuning.

## Future Work

- Compare PaddleOCR PP-Structure, DocLayout-YOLO, and LayoutParser outputs in
  the layout model workbench.
- Add a fusion layer that combines OpenCV proposals, OCR geometry, and optional
  layout-model classes.
- Add annotation and correction workflows for broad/mixed crops.
- Preserve labeled diagrams while distinguishing captions and body text more
  reliably.
