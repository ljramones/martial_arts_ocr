# Documentation Index

## Extraction Review

- [Review-Mode Extraction Guide](review-mode-extraction-guide.md): how to run
  explicit image-region review, inspect outputs, and avoid committing generated
  files.
- [Extraction Architecture Freeze](extraction-architecture-freeze-2026-04-28.md):
  current extraction architecture, accepted limits, backlog, and the decision to
  shift focus back to OCR/text/document-output quality.
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

## OCR And Document Output

- [OCR Output State](ocr-output-state-2026-04-28.md): current OCR/text artifact
  state, accepted limits, validation trail, and recommended next branch.
- [OCR Text Quality Assessment](ocr-text-quality-assessment.md): assessment of
  OCR output, cleanup, Japanese handling, and canonical model gaps.
- [OCR Text Normalization Notes](ocr-text-normalization-notes.md): word/line
  hierarchy, readable text, cleanup-chain guardrails, and current limits.
- [DocumentResult Serialization](document-result-serialization.md): current
  `data.json`, `text.txt`, line/word alias, and text summary behavior.

## Current Rule

Region extraction is review-mode only. Image extraction and Paddle fusion remain
disabled by default in normal runtime.
