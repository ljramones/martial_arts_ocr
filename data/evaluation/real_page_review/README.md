# Real-Page Evaluation

This directory is for review notes and evaluation metadata that are not specific to one corpus.

For the Donn Draeger master pages, keep manifests with the corpus:

```text
data/corpora/donn_draeger/dfd_notes_master/manifests/
```

For local review:

1. Generate or edit the corpus `manifest.local.json`.
2. Open `notebooks/05_real_page_extraction_review.ipynb`.
3. Record threshold notes, failure modes, and page-class observations under `data/evaluation/real_page_review/notes/`.

The manifest is intentionally descriptive, not a strict benchmark. Use it to record representative pages, expected extraction behavior, OCR-like sample text, and failure notes before runtime integration.
