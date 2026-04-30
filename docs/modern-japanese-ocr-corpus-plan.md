# Modern Japanese OCR Corpus Plan

## Purpose

Organize modern Japanese OCR resources so the project can build a small,
trustworthy evaluation set before changing OCR behavior, promoting Japanese
analysis into canonical fields, or beginning model work.

This plan covers modern printed/typed Japanese OCR only. Classical Japanese,
kuzushiji, densho, and handwritten historical material are out of scope for
this phase.

## Current Resources Found

| Path | Inferred Source | Type | Original / External / Synthetic / Text-only / Unknown | Recommended Location | Commit Policy | Notes |
|---|---|---|---|---|---|---|
| `data/IMG_OCR_JP_CN/` | Unknown third-party OCR dataset inferred from filenames | Image files paired with LabelMe-style JSON region/text annotations | External, original image+annotation data, provenance unknown | `data/corpora/modern_japanese_ocr/external/img_ocr_jp_cn/original/` | Do not commit payloads until source/license are confirmed. Track README and manifest example only. | Around 112 files observed; categories include contracts, papers, newspapers, forms, notes, identity cards, receipts, and book covers. |
| `data/P-font-image-dataset/` | Unknown generated/Kaggle-style font-image dataset inferred from `labels.csv` | Synthetic Japanese text images plus CSV labels | Synthetic fixture data, provenance unknown | `data/corpora/modern_japanese_ocr/synthetic_or_text_fixtures/japanese_font_images/original/` | Do not commit payloads. Track README and manifest example only. | Around 150,000 image files plus `labels.csv`; useful for fixtures, not real scanned-page OCR validation. |
| `data/corpora/ad_hoc/modern_japanese/original/` | Unknown loose local image samples | Image-only samples | External/unknown qualitative samples | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/` | These three images were already tracked; preserve them as moved existing data. Do not add more payloads until provenance/license are confirmed. | Three images observed; no ground truth. |
| DFD and Corpus 2 focused Japanese candidates | Project-native source corpora | Existing source pages selected by manifest/contact-sheet review | Project-native | `data/corpora/modern_japanese_ocr/project_native/` manifests should reference original source paths | Do not duplicate private images unless necessary; prefer manifests pointing at existing source paths. | Useful for project-representative qualitative OCR review. |

## Target Layout

```text
data/
  corpora/
    modern_japanese_ocr/
      README.md

      project_native/
        README.md
        original/
        manifests/
          manifest.example.json
          manifest.local.json

      external/
        README.md

        img_ocr_jp_cn/
          README.md
          original/
          manifests/
            manifest.example.json
            manifest.local.json

        manual_modern_japanese_samples/
          README.md
          original/
          manifests/
            manifest.example.json
            manifest.local.json

      synthetic_or_text_fixtures/
        README.md

        japanese_font_images/
          README.md
          original/
          manifests/
            manifest.example.json
            manifest.local.json
```

## Dataset Handling Rules

- Keep dataset payloads under `original/`.
- Keep generated derivatives under `derived/` if created later.
- Track README files and `manifest.example.json`.
- Keep `manifest.local.json` ignored by default.
- Do not commit large image sets, archives, generated OCR outputs, or private
  images without an explicit policy change.
- If provenance or license is unknown, mark the dataset as provisional and do
  not use it as canonical benchmark data.
- Distinguish real scanned-page OCR resources from synthetic/font fixtures and
  text-only corpora.

## Evaluation Workflow

1. Confirm provenance and license for any external resource.
2. Create a small `manifest.local.json` with representative samples.
3. Label each sample by use:
   - quantitative OCR accuracy if image + ground truth is confirmed;
   - qualitative OCR stress test if image-only;
   - cleanup/serialization fixture if synthetic or text-only.
4. Run controlled OCR comparisons with current helper scripts.
5. Record results under `data/evaluation/real_page_review/notes/`.
6. Promote only the smallest useful sample set into tracked fixtures if
   licensing and size allow.

## Not for This Phase

- No model training.
- No dependency additions.
- No runtime OCR behavior changes.
- No classical/kuzushiji/densho OCR work.
- No automatic PDF/image conversion pipeline.
- No broad corpus import into git.

## Recommended Next Step

Create a focused modern Japanese OCR evaluation manifest from:

- `img_ocr_jp_cn` if provenance/license and JSON ground-truth semantics are
  acceptable;
- a small project-native set from DFD/Corpus 2 pages already identified in the
  focused Japanese OCR review;
- optionally a tiny synthetic fixture subset for cleanup/serialization tests.

Then run an OCR comparison against that manifest without changing runtime
defaults.
