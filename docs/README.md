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
- [Modern Japanese OCR Corpus Plan](modern-japanese-ocr-corpus-plan.md):
  organization, provenance notes, commit policy, and next steps for modern
  Japanese OCR evaluation resources.
- [Modern Japanese OCR Corpora](../data/corpora/modern_japanese_ocr/README.md):
  local corpus area for project-native, external, and synthetic/text fixture
  resources.

## OCR And Document Output

- [Document Output State](document-output-state-2026-04-28.md): current
  `data.json`, `page_data.json`, `text.txt`, and `page_1.html` artifact
  contract.
- [OCR Output State](ocr-output-state-2026-04-28.md): current OCR/text artifact
  state, accepted limits, validation trail, and recommended next branch.
- [Japanese OCR Experimental State](japanese-ocr-experimental-state-2026-04-28.md):
  current experiment-only Japanese OCR profiles, evidence, deferrals, and next
  recommended evidence branch.
- [Japanese Region OCR Routing Design](japanese-region-ocr-routing-design.md):
  evidence-based review-mode design for routing Japanese-bearing regions to
  language, PSM, and preprocessing profiles.
- [Macron Normalization Candidate Design](macron-normalization-candidate-design.md):
  review-only glossary-backed design for suggesting macronized romanization
  candidates without mutating OCR text.
- [Macron Candidate Review State](macron-candidate-review-state-2026-04-28.md):
  current glossary-backed candidate behavior, review artifact results, and the
  guardrail against blind automatic normalization.
- [Macron Candidate Workflow State](macron-candidate-workflow-state-2026-04-28.md):
  current end-to-end macron candidate review workflow, reviewed counts, case
  suggestions, and recommended pause point.
- [Macron Candidate Review Workflow Design](macron-candidate-review-workflow-design.md):
  review/export workflow design for accepting, rejecting, or deferring macron
  candidates without mutating original OCR text.
- [Macron Candidate Review Operator Guide](macron-candidate-review-operator-guide.md):
  commands and local-file workflow for running macron candidate review exports
  and editing `decisions.local.json`.
- [OCR Text Quality Assessment](ocr-text-quality-assessment.md): assessment of
  OCR output, cleanup, Japanese handling, and canonical model gaps.
- [OCR Text Normalization Notes](ocr-text-normalization-notes.md): word/line
  hierarchy, readable text, cleanup-chain guardrails, and current limits.
- [DocumentResult Serialization](document-result-serialization.md): current
  `data.json`, `text.txt`, line/word alias, and text summary behavior.

## Current Rule

Region extraction is review-mode only. Image extraction and Paddle fusion remain
disabled by default in normal runtime.
