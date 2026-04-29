# Layout Model Evaluation Plan

## Purpose

The current review-mode image detector is useful but still limited by classical
OpenCV heuristics. Corpus 2 showed that similar scanned martial-arts pages can
bring back text false positives and miss photo/figure regions. This workbench
compares document-layout approaches before any model becomes part of runtime.

## Detector Stack

- `classical_opencv`: existing contour, text-like rejection, recall, and
  consolidation path. Always available and remains the baseline.
- `paddle_ppstructure`: optional PaddleOCR PP-Structure adapter. Skips unless
  `paddleocr` is installed locally.
- `layoutparser`: optional LayoutParser/Detectron2 adapter. Requires a model or
  model factory supplied explicitly; no downloads happen automatically.
- `doclayout_yolo`: optional DocLayout-YOLO style adapter. Requires Ultralytics
  and a local model path.
- `generic_yolo`: optional YOLO adapter for locally trained document-region
  experiments. It is not a default detector.

## Evaluation Pages

Use the same difficult pages already reviewed:

- DFD: `3335`, `3344`, `3397`, `3352`, `3330`
- Corpus 2: pages from `2026-04-28-corpus2-generalization-review.md` that had
  text false positives, missed photo grids, or broad mixed crops

Do not commit private images or generated crops. Notebook output belongs under
`data/notebook_outputs/layout_model_comparison/`.

## Acceptance Questions

- Does the model separate text/title/caption regions from figures better than
  the classical detector?
- Are diagrams/photos retained without broad mixed text/image crops?
- Are labels inside diagrams kept as part of a useful figure crop?
- Are tall/narrow visual strips treated as figures or incorrectly rejected as
  vertical text?
- Does the backend run locally without downloads and fail gracefully when absent?

## Decision Criteria

- Keep classical OpenCV as the default fallback.
- If an optional document-layout model clearly improves text/figure separation,
  evaluate it in review mode only.
- If no off-the-shelf model improves results, prioritize OCR-aware text-box
  suppression before additional heuristic tuning.
- Do not enable any ML backend by default until it is tested on both DFD and
  Corpus 2 and its dependencies are documented.

## Next Implementation Boundary

This plan creates comparison infrastructure only. It does not turn on ML
detectors, change the `WorkflowOrchestrator`, or alter the default disabled
image-extraction setting.
