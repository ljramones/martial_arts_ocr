# Dataset Inventory

This inventory records observed dataset-like folders. Classifications are based on folder names and shallow file inspection, so uncertain entries are marked as inferred.

## Root-Level And Data Folders

| Path | Likely Purpose | Classification | Recommended Handling |
| --- | --- | --- | --- |
| Old `all_DFD_Notes_Master_File/` | Former root-level location for Donn Draeger lecture page images. | Superseded original source path | Replaced by `data/corpora/donn_draeger/dfd_notes_master/original/`. |
| `data/corpora/donn_draeger/dfd_notes_master/original/` | Donn Draeger lecture page images, mostly `IMG_*.jpg` files. | Original source corpus | Treat as ground-truth real-page review set. Do not mix with augmented data or notebook output. |
| `data/corpora/donn_draeger/dfd_notes_master/augmented/` | Reserved for rotated, noisy, skewed, tiled, or otherwise transformed copies of the original pages. | Augmented/derived | Use only after original pages have been reviewed. |
| `data/corpora/donn_draeger/dfd_notes_master/manifests/` | Corpus manifest examples and local review manifest. | Evaluation metadata | Commit examples; keep `manifest.local.json` ignored. |
| `data/corpora/ad_hoc/modern_japanese/original/` | Small set of Japanese sample images. | Original or ad hoc sample, inferred | Keep separate from Donn Draeger corpus; use as a secondary modern-Japanese review set if provenance is known. |
| `data/corpora/ad_hoc/corpus2/original/` | Second local corpus, 138 JPEG images formerly at `data/corpus2/`. | Original or ad hoc sample, inferred | Keep separate from Donn Draeger corpus. Treat images as local/private; use for generalization review after DFD validation. |
| `data/corpora/ad_hoc/corpus2/augmented/` | Reserved for derived variants from Corpus 2 originals. | Augmented/derived | Keep empty until transforms are intentionally generated from `original/`. |
| `data/corpora/ad_hoc/corpus2/manifests/` | Corpus 2 manifest example and local review manifest location. | Evaluation metadata | Commit examples; keep `manifest.local.json` ignored. |
| `data/evaluation/ad_hoc/test_folder/` | Two copied page images used for prior testing. | Local/ad hoc sample, inferred | Do not treat as canonical. Prefer manifests pointing to original source pages. |
| Old `data/uploads/` | Previous Flask upload output. | Superseded runtime/generated | Replaced by `data/runtime/uploads/`. |
| Old `data/processed/` | Previous processed artifact output. | Superseded runtime/generated | Replaced by `data/runtime/processed/`. |
| `data/runtime/` | Flask uploads, processed artifacts, local SQLite DB, and runtime model runs. | Runtime/generated | Ignored local output; do not use as original corpus. |
| `data/notebook_outputs/diagnostic_output/` | Diagnostic preprocessing variants and results. | Generated diagnostic output | Keep out of source corpus and quality manifests. |
| `data/runs/` | Detector/model run outputs. | Generated/model output | Treat as disposable runtime output. |
| `data/notebook_outputs/debug_output/` | Layout/debug images. | Generated diagnostic output | Local diagnostic output, not source data. |
| `data/notebook_outputs/` | Notebook crops, overlays, and review diagnostics. | Temporary notebook output | Gitignored; never use as input corpus. |
| `experiments/image_layout_model/dataset/` | YOLO/layout dataset lists, images, labels, and caches. | Training/derived, inferred | Classified as training data. Leave in place for now because experiment scripts assume this relative layout; future home is `data/training/image_layout/`. |
| `experiments/orientation_model/data/` | Train/validation/test data for orientation model. | Training/derived, inferred | Classified as training data. Future home is `data/training/orientation/`. |
| `experiments/orientation_model/checkpoints/` | Model checkpoint files. | Model output | Do not mix with source pages. |
| `static/extracted_content/` | Web/static extracted content. | Generated/runtime output, inferred | Output only. |
| `processors/data/` | Frequency lists, terminology, OCR correction data. | Reference data | Keep versioned if intentionally curated. Not a page corpus. |
| `tessdata/` | Tesseract language data. | External model/data dependency | Treat as dependency data, not page corpus. |
| Old `samples/` | Former location for local evaluation manifests. | Superseded evaluation metadata | Replaced by corpus manifests under `data/corpora/.../manifests/`. |

## Dataset Lineage Rule

`data/corpora/donn_draeger/dfd_notes_master/original/` is the primary original source corpus for real-page extraction review. `data/corpora/ad_hoc/corpus2/original/` is a secondary local corpus for generalization checks. Augmented, rotated, tiled, cropped, model-training, diagnostic, and notebook-output folders should remain separate until the original pages have been reviewed.

Use augmented/generated data later for robustness testing after the current image/text utility thresholds are acceptable on original pages.

## Manifest Workflow

Generate a local manifest from the original corpus:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input data/corpora/donn_draeger/dfd_notes_master/original \
  --output data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json
```

Then open `notebooks/05_real_page_extraction_review.ipynb` and review original pages first. Do not commit `manifest.local.json`, private scans, runtime output, or notebook output crops.

Generate a local manifest for Corpus 2:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input data/corpora/ad_hoc/corpus2/original \
  --output data/corpora/ad_hoc/corpus2/manifests/manifest.local.json \
  --collection-name corpus2 \
  --source-kind original
```
