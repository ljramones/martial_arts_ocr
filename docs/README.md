# Documentation Index

## Extraction Review

- [Review-Mode Extraction Guide](review-mode-extraction-guide.md): how to run
  explicit image-region review, inspect outputs, and avoid committing generated
  files.
- [OCR-Aware Image Region Filtering](ocr-aware-image-region-filtering.md): OCR
  text-box diagnostics, mixed-region triage, and optional Paddle fusion.
- [Image Region Detection Tuning](image-region-detection-tuning.md): detector
  tuning history, validation notes, and known limits.
- [Layout Model Evaluation Plan](layout-model-evaluation-plan.md): optional
  document-layout backend comparison plan and Paddle evaluation results.
- [Extraction Workbench Plan](extraction-workbench-plan.md): notebook and
  corpus review workflow.
- [Extraction Quality Rubric](extraction-quality-rubric.md): criteria for
  judging image, text, and layout extraction quality.

## Data And Corpus Organization

- [Dataset Inventory](dataset-inventory.md): current corpus/data folders and
  lineage notes.

## Current Rule

Region extraction is review-mode only. Image extraction and Paddle fusion remain
disabled by default in normal runtime.
