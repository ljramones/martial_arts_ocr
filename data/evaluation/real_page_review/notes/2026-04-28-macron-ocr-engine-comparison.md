# Macron OCR Engine Comparison

## Purpose

Compare available OCR engines on controlled synthetic romanized Japanese terms with macrons.

## Scope

This evaluates Latin-script macron recognition, not kanji/kana OCR.

Runtime OCR behavior, OCR defaults, canonical fields, extraction, serialization, and schema were not changed.

## Environment

- Main Python env: `.venv`
- `.venv-eval` available: yes
- Tesseract version: 5.5.2
- EasyOCR available: yes, importable in main `.venv`
- EasyOCR models available locally: yes, `craft_mlt_25k.pth` and `japanese_g2.pth`
- PaddleOCR available in `.venv-eval`: not usable in this pass
- Notes:
  - EasyOCR was run with `download_enabled=False`.
  - PaddleOCR import in `.venv-eval` failed before OCR because PaddleX attempted to create `/Users/larrymitchell/.paddlex/temp` and hit a permission error.
  - Generated synthetic images and JSON output stayed under ignored `data/notebook_outputs/macron_ocr_engine_comparison/`.

Helper:

```text
experiments/compare_macron_ocr_engines.py
```

Command:

```bash
.venv/bin/python experiments/compare_macron_ocr_engines.py \
  --output-dir data/notebook_outputs/macron_ocr_engine_comparison
```

## Fixtures

| Fixture | Terms Included | Source |
|---|---|---|
| `arial_large_terms` | `koryū`, `budō`, `Daitō-ryū`, `jūjutsu`, `dōjō`, `ryūha`, `sōke`, `iaidō`, `kenjutsu`, `aikijūjutsu` | synthetic generated image |
| `times_large_terms` | same term list | synthetic generated image |
| `arial_small_sentence` | `koryū`, `budō`, `Daitō-ryū`, `dōjō`, `aikijūjutsu`, `iaidō` | synthetic generated image |

## Summary Table

Counts are across all fixture terms. `preserved` includes the plain non-macron term `kenjutsu`; `macron_preserved` counts only terms containing macrons.

| Engine | Config | Macron Preserved | Preserved | Stripped | Misread | Missing | Unavailable | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Tesseract | `eng`, PSM 3 | 0 | 2 | 12 | 1 | 18 | 0 | Preserved only `kenjutsu`; macron terms became forms such as `koryG`, `bud6`, `Dait6-rya`, or stripped variants. |
| Tesseract | `eng+jpn`, PSM 3 | 0 | 2 | 27 | 0 | 4 | 0 | Best at producing plain ASCII romanization; did not preserve macrons. |
| Tesseract | `jpn`, PSM 3 | 0 | 2 | 25 | 0 | 6 | 0 | Similar to `eng+jpn`; strips most macrons to ASCII romanization. |
| EasyOCR | `ja`, local models, downloads disabled | 0 | 0 | 0 | 0 | 33 | 0 | Ran locally but produced Japanese-like/noisy output, not Latin macron terms. |
| PaddleOCR | `.venv-eval` availability | 0 | 0 | 0 | 0 | 0 | 33 | Could not import due `.paddlex/temp` permission failure; not counted as OCR failure. |

## Per-Term Results

| Term | Tesseract Result | EasyOCR Result | PaddleOCR Result | Best Engine | Notes |
|---|---|---|---|---|---|
| `koryū` | stripped or missing | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Usually became `koryu`; `eng` sometimes produced `koryG`. |
| `budō` | stripped | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Usually became `budo` or `bud6`. |
| `Daitō-ryū` | stripped or misread | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `Daito-ryu`, `Dait6-rya`, or `Daito-ryt`. |
| `jūjutsu` | stripped or missing | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `jujutsu`, `jGjutsu`, `jajutsu`, or noisy variants. |
| `dōjō` | stripped | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `dojo` or `d6j6`. |
| `ryūha` | stripped or missing | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `ryuha`, `ryaha`, or noisy variants. |
| `sōke` | stripped | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `soke`. |
| `iaidō` | stripped | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `iaido`, `iaid6`, or noisy variants. |
| `kenjutsu` | preserved | missing | unavailable | Tesseract | Plain non-macron term was recoverable. |
| `aikijūjutsu` | stripped or missing | missing | unavailable | Tesseract `eng+jpn` / `jpn` | Became `aikijujutsu`, `aikijGjutsu`, or noisy variants. |
| `ō` | not preserved as standalone macron character | missing | unavailable | none | OCR did not emit the macron character. |
| `ū` | not preserved as standalone macron character | missing | unavailable | none | OCR did not emit the macron character. |

## Findings

- No tested engine preserved macron-bearing terms from the synthetic fixtures.
- Tesseract was the only engine that recovered useful Latin romanization at all, but it stripped or misread macrons.
- Tesseract `eng+jpn` and `jpn` were better than `eng` at producing ASCII romanization such as `koryu`, `budo`, and `Daito-ryu`.
- Tesseract `eng` more often misread macron-bearing vowels as digits or letters, such as `bud6`, `koryG`, and `Dait6-rya`.
- EasyOCR ran without downloads using local models, but the `ja` reader did not recover the Latin romanized terms.
- PaddleOCR could not be evaluated because `.venv-eval` import failed before OCR due local PaddleX cache permission behavior.
- The previous macron curation pass remains valid: cleanup and serialization preserve macrons if OCR emits them. The blocker is OCR recognition, not text cleanup.

## Decision

- [ ] One engine preserves macrons well enough for future review-mode evaluation
- [x] Engines mostly strip/misread macrons
- [ ] EasyOCR/PaddleOCR could not be evaluated because models were unavailable
- [x] Need real macron-bearing samples before deciding
- [x] Next branch should be glossary-backed normalization candidates with human review

Clarification: EasyOCR was available and ran, but did not recover terms. PaddleOCR could not be evaluated because the eval environment import failed.

## Recommendation

Do not add automatic blind normalization such as:

```text
koryu -> koryū
budo -> budō
Daito-ryu -> Daitō-ryū
```

The evidence supports a more cautious next branch:

```text
reviewed martial-arts glossary
  -> OCR ASCII/garbled term candidates
  -> suggested macron restoration
  -> human review
```

This should remain optional/review-mode and should be driven by a curated glossary plus confidence/ambiguity metadata. It should not silently rewrite OCR text in the canonical pipeline.

Also continue looking for real macron-bearing source images. Synthetic fixtures are useful for engine comparison and regression tests, but they are not proof of real scanned-page behavior.
