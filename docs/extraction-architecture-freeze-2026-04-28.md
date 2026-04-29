# Extraction Architecture Freeze — 2026-04-28

## Status

Region extraction is frozen for the current project phase. The stack is mature
enough for review-mode use, and further detector expansion should stop unless a
specific downstream workflow forces it.

The next technical domain should be OCR/text/document-output quality, not more
region-detection logic.

## Current Architecture

```text
classical detector
+ OCR-aware diagnostics
+ mixed-region triage
+ optional Paddle PPStructureV3 fusion
```

Classical detector:

- candidate generation
- text-like rejection
- recall recovery
- post-filter consolidation

OCR-aware diagnostics:

- OCR word/line box overlap
- OCR text-mask overlap
- candidate metadata for text-vs-visual decisions

Mixed-region triage:

- marks broad/mixed regions
- records `needs_review` metadata
- does not reliably split crops automatically

Optional Paddle PPStructureV3 fusion:

- optional semantic layout signal
- disabled by default
- adds high-confidence Paddle visual regions near unresolved mixed classical
  parents
- V2 additive rule validated on the focused evaluation set

## Stable / Accepted

- The classical detector remains the baseline.
- OCR-aware diagnostics are useful and safe.
- Real OCR boxes are preserved through the canonical model.
- Mixed-region triage is useful for review workflows.
- The review-mode runner exists for crop inspection.
- Paddle fusion V2 is validated as an optional review-mode enhancement.
- `DocumentResult`, `PageResult`, `TextRegion`, `ImageRegion`, and
  `BoundingBox` remain the canonical pipeline models.

## Review-Mode Only

- Image-region extraction is a candidate-region review aid.
- Paddle fusion is a review/evaluation tool, not a default runtime path.
- Generated crops should be inspected manually before downstream use.
- `experiments/run_review_mode_extraction.py` is for extraction/crop inspection
  and intentionally does not evaluate OCR text quality.

## Disabled by Default

Runtime defaults remain:

```text
ENABLE_IMAGE_REGION_EXTRACTION=false
IMAGE_REGION_EXTRACTION_SAVE_CROPS=true
IMAGE_REGION_EXTRACTION_FAIL_ON_ERROR=false
ENABLE_PADDLE_LAYOUT_FUSION=false
PADDLE_LAYOUT_MODEL_DIR=
```

PaddleOCR is not a required dependency. `.venv-eval` is optional. Generated
crops, overlays, runtime outputs, and model caches remain ignored.

## Experimental

- Paddle PPStructureV3 fusion is optional and evaluation-oriented.
- V2 additive fusion is validated on a limited hard-page set, not the full
  corpus.
- Mixed-region refinement is useful as triage, but not as reliable automatic
  crop splitting.
- Layout-model backends other than Paddle remain comparative workbench items.

## Validation Evidence

Review trail:

- `data/evaluation/real_page_review/notes/2026-04-28-dfd-first-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-dfd-post-text-filter-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-dfd-recall-tuning-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-dfd-consolidation-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-dfd-broader-consolidation-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-dfd-review-mode-runtime-validation.md`
- `data/evaluation/real_page_review/notes/2026-04-28-corpus2-generalization-review.md`
- `data/evaluation/real_page_review/notes/2026-04-28-cross-corpus-generalization-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-ocr-aware-real-box-validation.md`
- `data/evaluation/real_page_review/notes/2026-04-28-real-ocr-box-plumbing-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-mixed-region-refinement-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-paddle-layout-eval.md`
- `data/evaluation/real_page_review/notes/2026-04-28-paddle-fusion-validation.md`
- `data/evaluation/real_page_review/notes/2026-04-28-paddle-fusion-failure-analysis.md`
- `data/evaluation/real_page_review/notes/2026-04-28-paddle-fusion-v2-validation.md`

Summary:

- OCR-aware filtering was safe but did not solve broad/mixed crops alone.
- Mixed refinement became triage, not automatic crop splitting.
- Paddle PPStructureV3 materially improved Corpus 2 broad/mixed cases.
- V1 containment fusion failed the gate at `2/5` Corpus 2 improvements.
- Failure analysis identified wrong/partial classical parents as the dominant
  issue.
- V2 additive Paddle fusion passed the focused gate:
  - Corpus 2 broad/mixed improved: `4/5`
  - DFD incorrect additions: `0/5`
  - known-good incorrect additions: `0/2`
  - region count reasonable
  - no plain-text Paddle additions

## Known Limitations

- Automatic crop splitting is not fully solved.
- V2 additive fusion is validated on a limited evaluation set, not the whole
  corpus.
- Paddle dependency/model setup remains separate and optional.
- Tables are not `ImageRegion` objects.
- Multi-Paddle-region-per-classical-parent remains limited.
- DocLayout-YOLO has not been evaluated with real document-layout weights.
- Classical Japanese, densho, makimono, and other unusual layouts may require a
  separate strategy later.

## Do Not Continue Right Now

- Do not build V3 fusion now.
- Do not chase the remaining `1/5` Corpus 2 case.
- Do not add more detector presets.
- Do not tune per-corpus thresholds.
- Do not train or fine-tune yet.
- Do not make Paddle default.
- Do not enable image extraction by default.
- Do not broaden the detector stack until downstream needs force it.

## Future Backlog

- Broader V2 review on more pages.
- Optional annotation workflow for `needs_review` regions.
- DocLayout-YOLO evaluation with real document-layout weights.
- Table extraction support.
- Multi-region Paddle fusion.
- Paddle setup documentation for supported environments.
- Classical Japanese/densho-specific detector strategy.
- Broader review-mode UI for accept/reject/adjust bbox workflows.

## Recommended Next Project Focus

Move attention to:

- real OCR text quality
- Japanese-safe text normalization
- OCR word/line box reliability
- `DocumentResult` serialization quality
- page reconstruction quality
- downstream archive/review workflows

The next technical domain should be OCR/text/document-output quality, not more
region-detection logic.

## Relevant Files

- `utils/image/layout/analyzer.py`
- `utils/image/layout/fusion.py`
- `utils/image/layout/refinement.py`
- `utils/image/layout/strategies/paddle_layout.py`
- `utils/text/geometry.py`
- `src/martial_arts_ocr/pipeline/extraction_service.py`
- `src/martial_arts_ocr/pipeline/document_models.py`
- `experiments/check_paddle_layout.py`
- `experiments/run_review_mode_extraction.py`
- `docs/review-mode-extraction-guide.md`
- `docs/ocr-aware-image-region-filtering.md`
- `docs/image-region-detection-tuning.md`
- `docs/layout-model-evaluation-plan.md`

## Relevant Review Notes

The review notes under `data/evaluation/real_page_review/notes/` are the source
of record for why this architecture was accepted and where it remains limited.
Future extraction work should add new notes only when a concrete downstream need
justifies reopening this area.
