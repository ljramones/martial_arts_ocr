# Modern Japanese OCR Corpora

This area holds resources for modern Japanese OCR evaluation. It is separate
from the DFD and Corpus 2 source corpora and from any future
classical/kuzushiji/densho evaluation.

The current goal is to build small, trustworthy OCR evaluation sets before any
model training or canonical Japanese analysis promotion.

## Layout

```text
data/corpora/modern_japanese_ocr/
  project_native/
    Pages selected from this repo's DFD / Corpus 2 material.
  external/
    Third-party modern Japanese OCR/document datasets.
  synthetic_or_text_fixtures/
    Generated images, font datasets, bilingual/text corpora, or fixtures.
```

## Commit Policy

Track:

- README files.
- `manifests/manifest.example.json`.
- Empty `.gitkeep` placeholders where useful.
- Planning and provenance notes.

Do not track by default:

- Dataset payloads under `original/`.
- Generated derivatives under `derived/`.
- Local manifests, especially `manifest.local.json`.
- Large archives, downloaded model files, generated OCR output, or private data.

If a dataset license and size make it appropriate for version control, change
that policy explicitly in the dataset README before staging payload files.

## Evaluation Use

Use this area to assemble modern Japanese OCR evaluation manifests. Prefer
small, inspected samples with known purpose:

- Image + ground truth for quantitative OCR checks.
- Image-only documents for qualitative OCR stress tests.
- Synthetic/font fixtures for cleanup and serialization guardrails.
- Text-only corpora for terminology, normalization, and test fixture generation.

## Not For This Phase

Do not use this folder to begin classical Japanese, kuzushiji, densho, or model
training work. Those need separate corpus handling and evidence gates.
