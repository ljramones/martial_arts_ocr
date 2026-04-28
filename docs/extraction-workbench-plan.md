# Extraction Workbench Plan

## Current Inventory

`utils/image` contains image IO, validation, resizing, thumbnailing, crop extraction, tone/preprocessing helpers, region dataclasses and geometry, layout detectors, and CLI/API wrappers. The strongest production candidates are `utils.image.api`, `ImageProcessor`, `LayoutAnalyzer`, `ImageRegion`, `extract_region`, and `save_region_crops`.

`utils/text` is centered on `text_utils.py`: `TextCleaner`, `LanguageDetector`, `TextFormatter`, `TextStatistics`, and Japanese martial arts terminology helpers. `TextCleaner`, `LanguageDetector`, and `TextStatistics` are already used by OCR/Japanese processing.

## Production Candidates

- `utils/image/io/*`: image load/save/validation metadata.
- `utils/image/ops/*`: resize, tone, crop extraction, thumbnails.
- `utils/image/regions/core_types.py`: bbox-backed region data and geometry.
- `utils/image/layout/analyzer.py`: primary classical-CV layout entrypoint.
- `utils/text/text_utils.py`: OCR cleanup, segmentation, statistics, formatting.
- `src/martial_arts_ocr/pipeline/extraction_adapters.py`: boundary from utility outputs to canonical models.

## Experimental Or Legacy

- `utils/image/layout/_legacy_image_layout.py`: compatibility reference, not new runtime work.
- `utils/image/regions/image_regions.py`: compatibility shim.
- YOLO detector path: optional ML-assisted experiment; must stay opt-in and skipped gracefully when unavailable.
- Orientation/model experiments belong under `experiments/` or notebooks until stable.

## Overlap

`src/martial_arts_ocr/imaging` currently wraps or builds on `utils/image`, especially layout and content extraction. OCR postprocessing overlaps with `TextCleaner`; reconstruction overlaps with `TextFormatter`. Keep algorithms in `utils`, and keep workflow decisions, persistence, and `DocumentResult` construction in `pipeline`.

## Known Risks

- Layout detectors need visual review on real pages; synthetic tests only prove baseline behavior.
- Thresholds such as contour area, halo, text filtering, and merge/NMS are document-sensitive.
- Text cleanup must remain Japanese-safe; avoid ASCII-only normalization.
- YOLO imports should not instantiate models unless explicitly requested.

## Notebook Roles

Notebooks are for visual inspection, threshold tuning, and comparing algorithms. They should use generated images by default, accept optional sample paths, and write scratch output under `data/notebook_outputs/`.

## Acceptance Criteria

- Tiny noise is ignored in image-region detection.
- Diagram/image crops preserve bboxes and write stable metadata.
- Text cleanup preserves Japanese, macrons, punctuation, and useful line breaks.
- Utility outputs can map to `TextRegion`, `ImageRegion`, `PageResult`, and `DocumentResult`.
- No normal test requires OCR binaries, model downloads, YOLO, MeCab, UniDic, pykakasi, or Argos.

## Real-Page Evaluation Workflow

1. Keep source pages under a corpus `original/` folder, such as `data/corpora/donn_draeger/dfd_notes_master/original/`.
2. Generate or copy a corpus manifest to `data/corpora/<collection>/<corpus>/manifests/manifest.local.json`.
3. Edit sample paths, descriptions, expected counts, OCR-like sample text, and review notes.
4. Open `notebooks/05_real_page_extraction_review.ipynb`.
5. Review detected image regions, saved crops, text cleanup, and reading order notes.
6. Record threshold and failure notes in the local manifest or a separate review document.
7. Only after several representative pages pass, consider `WorkflowOrchestrator` integration.

## Using The Donn Draeger Master Page Corpus

1. Keep `data/corpora/donn_draeger/dfd_notes_master/original/` as the original source corpus. The old root-level `all_DFD_Notes_Master_File/` path has been superseded.
2. Generate a local manifest:

   ```bash
   .venv/bin/python scripts/generate_real_page_manifest.py \
     --input data/corpora/donn_draeger/dfd_notes_master/original \
     --output data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json
   ```

3. Open `notebooks/05_real_page_extraction_review.ipynb`.
4. Review image-region detection, crop quality, text cleanup, and reading order.
5. Record threshold and failure notes.
6. Only after original pages are reviewed, test augmented, rotated, or training data.

## Next Implementation Pass

Image-region extraction now has a gated runtime integration path through `ExtractionService`. It is disabled by default and should be enabled only for review/workbench processing:

```bash
ENABLE_IMAGE_REGION_EXTRACTION=true .venv/bin/python app.py
```

When enabled, extraction enriches `DocumentResult.pages[].image_regions`, saves crops under each processed document output directory, and records diagnostics in metadata. The default remains off because review found broad/label-heavy crops, fragile tall/narrow visual strips, and validation only against the DFD corpus so far.

Before enabling more broadly, run review-mode processing on selected DFD pages and a second corpus to check generalization.
