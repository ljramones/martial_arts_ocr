# Review-Mode Extraction Guide

## What Review Mode Is

Review-mode extraction is an opt-in workflow for inspecting candidate image,
diagram, and figure regions on scanned pages. It enriches `DocumentResult` with
candidate `ImageRegion` objects and can save crop files for visual review.

It is not the default OCR path. It is a research/review aid for understanding
page layout and deciding which regions are useful enough for downstream
processing.

## Why It Is Disabled by Default

Real-page review showed that image-region extraction is useful but not
authoritative. The detector can still produce broad, label-heavy, or
needs-review crops on difficult pages. For that reason, normal OCR processing
does not run image extraction unless explicitly configured.

Runtime defaults:

```text
ENABLE_IMAGE_REGION_EXTRACTION=false
IMAGE_REGION_EXTRACTION_SAVE_CROPS=true
IMAGE_REGION_EXTRACTION_FAIL_ON_ERROR=false
ENABLE_PADDLE_LAYOUT_FUSION=false
PADDLE_LAYOUT_MODEL_DIR=
```

## Stack Overview

Classical detection is the baseline. It uses OpenCV contour/region logic,
text-like rejection, recall rescue, and consolidation to propose candidate image
regions.

OCR-aware diagnostics use OCR word/line boxes when available. OCR geometry helps
identify regions that are likely text and preserve labeled diagrams that contain
some text.

Mixed-region triage marks broad or uncertain crops with metadata such as
`mixed_region` and `needs_review`. These crops should be inspected, not silently
trusted.

Paddle fusion is optional. When explicitly enabled in an evaluation environment,
Paddle PPStructureV3 layout regions can provide semantic layout evidence.
Paddle-derived regions are added only through gated fusion rules and remain
review-mode candidates.

## When To Use Paddle Fusion

Use Paddle fusion when:

- you are running in an isolated evaluation environment such as `.venv-eval`
- PaddleOCR/PaddlePaddle are installed locally
- you are reviewing broad/mixed crops or Corpus 2-style photo/grid pages
- generated outputs will stay under ignored directories

Do not use Paddle fusion as a normal runtime default. Paddle is optional,
heavier than the classical detector, and requires separate dependency/model
setup.

## Running Selected Pages

Use the manual experiment runner:

```bash
.venv/bin/python experiments/run_review_mode_extraction.py \
  data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg \
  --enable-image-extraction
```

With Paddle fusion in an environment where Paddle is installed:

```bash
.venv-eval/bin/python experiments/run_review_mode_extraction.py \
  data/corpora/ad_hoc/corpus2/original/example.jpg \
  --enable-image-extraction \
  --enable-paddle-fusion \
  --output-dir data/runtime/review_mode/doc_example
```

The runner uses a no-OCR processor and the canonical `WorkflowOrchestrator`
artifact path. It is intended for extraction inspection, not OCR text-quality
measurement.

## Outputs To Inspect

Expected output directory:

```text
data/runtime/processed/doc_<id>/
  data.json
  page_data.json
  page_1.html
  text.txt
  image_regions/
```

The manual runner may write under a custom output directory, but generated
review output should remain inside ignored paths such as `data/runtime/`.

Important files:

- `data.json`: canonical `DocumentResult` plus legacy artifact aliases
- `page_data.json`: reconstruction output when available
- `page_1.html`: reconstructed or fallback HTML
- `text.txt`: text emitted by the review processor
- `image_regions/`: saved crop files when crop saving is enabled

## Metadata To Inspect

Look under `pages[].image_regions[].metadata` in `data.json`.

Useful fields:

```text
mixed_region
needs_review
refinement_applied
layout_fusion_applied
layout_backend
fusion_mode
fusion_reason
region_role
layout_source_bbox
related_classical_bbox
horizontal_span_overlap_ratio
vertical_span_overlap_ratio
relation_reason
```

Interpretation:

- `needs_review=true`: inspect manually before using downstream
- `mixed_region=true`: crop likely combines visual content with text/labels
- `layout_fusion_applied=true`: Paddle layout evidence participated
- `fusion_mode=paddle_additive`: Paddle added a separate candidate region
- `region_role=paddle_added_mixed_or_uncertain`: useful but still broad or
  uncertain

## What Not To Commit

Do not commit:

```text
data/runtime/
data/notebook_outputs/
.venv-eval/
Paddle model/cache files
generated crop outputs
private corpus images
runtime SQLite databases
```

These paths are ignored so review experiments can be repeated without turning
the repository into a data/output dump.

## Related Documents

- `docs/ocr-aware-image-region-filtering.md`
- `docs/image-region-detection-tuning.md`
- `docs/layout-model-evaluation-plan.md`
- `docs/extraction-workbench-plan.md`
