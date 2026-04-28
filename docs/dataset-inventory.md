# Dataset Inventory

This inventory records observed dataset-like folders without moving or deleting any source data. Classifications are based on folder names and shallow file inspection, so uncertain entries are marked as inferred.

## Root-Level And Data Folders

| Path | Likely Purpose | Classification | Recommended Handling |
| --- | --- | --- | --- |
| `all_DFD_Notes_Master_File/` | Donn Draeger lecture page images, mostly `IMG_*.jpg` files. | Original source corpus | Treat as ground-truth real-page review set. Do not mix with augmented data or notebook output. |
| `modern_japanese/` | Small set of Japanese sample images. | Original or ad hoc sample, inferred | Keep separate from Donn Draeger corpus; use as a secondary modern-Japanese review set if provenance is known. |
| `test_folder/` | Two copied page images used for prior testing. | Local/ad hoc sample, inferred | Do not treat as canonical. Prefer manifests pointing to original source pages. |
| `data/uploads/` | Flask-uploaded runtime files. | Runtime/generated | Local-only runtime data; do not use as original corpus. |
| `data/processed/` | Processed document artifacts such as `data.json`, `page_data.json`, HTML, and text. | Runtime/generated | Output only; should not be mixed with input corpora. |
| `data/diagnostic_output/` | Diagnostic preprocessing variants and results. | Generated diagnostic output | Keep out of source corpus and quality manifests. |
| `data/runs/` | Detector/model run outputs. | Generated/model output | Treat as disposable runtime output. |
| `debug_output/` | Layout/debug images. | Generated diagnostic output | Local diagnostic output, not source data. |
| `notebooks/output/` | Notebook crops, overlays, and review diagnostics. | Temporary notebook output | Gitignored; never use as input corpus. |
| `experiments/image_layout_model/dataset/` | YOLO/layout dataset lists, images, labels, and caches. | Training/derived, inferred | Keep separate from original review. Use later for robustness/model work. |
| `experiments/orientation_model/data/` | Train/validation/test data for orientation model. | Training/derived, inferred | Keep separate from extraction-quality review. |
| `experiments/orientation_model/checkpoints/` | Model checkpoint files. | Model output | Do not mix with source pages. |
| `static/extracted_content/` | Web/static extracted content. | Generated/runtime output, inferred | Output only. |
| `processors/data/` | Frequency lists, terminology, OCR correction data. | Reference data | Keep versioned if intentionally curated. Not a page corpus. |
| `tessdata/` | Tesseract language data. | External model/data dependency | Treat as dependency data, not page corpus. |
| `samples/` | Manifest definitions for local evaluation. | Evaluation metadata | Commit examples and docs only; keep `manifest.local.json` and `private/` local. |

## Dataset Lineage Rule

`all_DFD_Notes_Master_File/` should be the primary original source corpus for real-page extraction review. Augmented, rotated, tiled, cropped, model-training, diagnostic, and notebook-output folders should remain separate until the original pages have been reviewed.

Use augmented/generated data later for robustness testing after the current image/text utility thresholds are acceptable on original pages.

## Manifest Workflow

Generate a local manifest from the original corpus:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input all_DFD_Notes_Master_File \
  --output samples/manifest.local.json
```

Then open `notebooks/05_real_page_extraction_review.ipynb` and review original pages first. Do not commit `samples/manifest.local.json`, private scans, or notebook output crops.
