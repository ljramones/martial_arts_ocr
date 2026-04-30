# Macronized Romanization OCR Sample Curation

## Purpose

Evaluate whether the OCR/text pipeline can detect and preserve romanized Japanese terms with macrons.

## Scope

This pass is about Latin-script romanization fidelity, not kanji/kana OCR.

It separates three questions:

```text
OCR recognition:
  Does Tesseract read macronized terms from images?

Cleanup:
  If text contains macrons, does cleanup preserve them?

Serialization:
  If readable_text contains macrons, do DocumentResult-style serialized output
  and text.txt-style output preserve them?
```

Runtime OCR defaults, extraction behavior, canonical model fields, serialization behavior, and schema were not changed.

## Environment

- OCR engine: Tesseract
- OCR version: 5.5.2
- language configs tested: `eng`, `eng+jpn`, `jpn`
- PSMs tested: 3, 6, 11
- real samples found: no confirmed real image samples with visible macrons
- synthetic samples used: yes, generated under ignored `data/notebook_outputs/macron_ocr_eval/synthetic/`

Helper:

```text
experiments/review_macron_ocr.py
```

Command:

```bash
.venv/bin/python experiments/review_macron_ocr.py \
  --output-dir data/notebook_outputs/macron_ocr_eval
```

Generated output:

```text
data/notebook_outputs/macron_ocr_eval/summary.json
data/notebook_outputs/macron_ocr_eval/synthetic/*.png
```

These outputs are ignored and should not be committed.

## Terms Tested

| Term | Real Sample? | Synthetic Sample? | Notes |
|---|---|---|---|
| `koryū` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `budō` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `Daitō-ryū` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `jūjutsu` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `dōjō` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `ryūha` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `sōke` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `iaidō` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `kenjutsu` | no confirmed | yes | No macron; OCR recovered this term in several synthetic runs. |
| `aikijūjutsu` | no confirmed | yes | OCR misread/stripped. Cleanup and serialization controls preserved. |
| `ō` | no confirmed | yes | OCR did not preserve as a character. Cleanup and serialization controls preserved. |
| `ū` | no confirmed | yes | OCR did not preserve as a character. Cleanup and serialization controls preserved. |

## Real Source Sample Review

No confirmed real image sample with visible macronized romanized Japanese was found in this pass.

Repository text search found macronized terms in:

```text
docs/
tests/
tests/fixtures/
data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.example.json
```

Those hits are useful for tests and documentation, but they are not real scanned-page image evidence. Prior real OCR reviews also did not observe macronized real OCR output.

| Sample ID | Path | Visible Macron Terms | OCR Config | Output | Result |
|---|---|---|---|---|---|
| n/a | n/a | none confirmed | n/a | n/a | Need curated real macron-bearing samples. |

## Synthetic Fixture Review

Synthetic fixtures were generated only as ignored experiment artifacts. They are useful for separating OCR recognition failure from cleanup/serialization behavior, but they are not a substitute for real corpus evidence.

| Fixture | OCR Config | Expected | OCR Output | Cleaned Output | Serialized Output | Result |
|---|---|---|---|---|---|---|
| `arial_large_terms` | best: `eng`, PSM 3 | `koryū budō Daitō-ryū`; `jūjutsu dōjō ryūha sōke`; `iaidō kenjutsu aikijūjutsu` | `koryG bud6 Dait6-rya ... iaid6 kenjutsu aikijGjutsu` | same normalized text | same readable text | partial: recovered `kenjutsu`, but no macrons. |
| `times_large_terms` | best: `eng`, PSM 3 | same term list | `koryt budo Daito-ryt ... iaido kenjutsu aikiytjutsu` | same normalized text | same readable text | partial: recovered `kenjutsu`, but no macrons. |
| `arial_small_sentence` | best: `eng`, PSM 3 | `Draeger studied koryū budō and Daitō-ryū. The dōjō taught aikijūjutsu and iaidō.` | `Draeger studied koryG bud6 and Daité-rya. The d6j6 taught aikijdjutsu and iaidd.` | same normalized text | same readable text | missing/misread: macronized terms not recovered. |

Cleanup/serialization control:

For each synthetic fixture, the expected text was passed directly through:

```text
OCRPostProcessor
  -> TextCleaner
  -> DocumentResult.to_dict()
  -> text.txt-style readable text
```

The control preserved all expected macronized terms in readable and serialized output.

## Preservation Results

- macrons preserved:
  - Cleanup/serialization controls preserved all synthetic expected macron terms.
  - No OCR run preserved the macrons themselves.
- macrons stripped:
  - `eng+jpn` and `jpn` commonly converted terms to non-macron forms such as `koryu`, `budo`, `Daito-ryu`, `jujutsu`, `dojo`, and `iaido`.
- macrons misrecognized:
  - `eng` commonly produced substitutions such as `koryG`, `bud6`, `Dait6-rya`, `jGjutsu`, and `iaid6`.
  - Times New Roman output produced forms like `koryt`, `Daito-ryt`, and `jajutsu`.
- cleanup damage:
  - none observed in direct expected-text controls.
- serialization damage:
  - none observed in direct expected-text controls.

## Findings

- The current cleanup and serialization path can preserve macrons when they are already present in text.
- The current Tesseract configs did not reliably OCR macrons from synthetic Latin-script images.
- `eng` sometimes kept one non-macron term (`kenjutsu`) but misread macron-bearing terms.
- `eng+jpn` and `jpn` tended to strip macrons into ASCII romanization rather than preserve diacritics.
- This is primarily an OCR recognition/sample evidence problem, not a cleanup or serialization problem.
- Real source evidence is still missing; synthetic fixtures are not enough to claim real OCR support.

## Decision

- [ ] Real source samples prove macron OCR/preservation is reliable
- [x] Synthetic samples prove cleanup/serialization preserves macrons, but real OCR evidence is still missing
- [x] OCR strips/misreads macrons; need OCR config or engine comparison
- [x] Need curated real macron-bearing samples
- [x] Do not promote Japanese/romanization fields yet

## Recommendation

Do not promote Japanese or romanization fields yet.

Next evidence branch:

```text
curate real macron-bearing samples
```

The project needs source images where terms such as `koryū`, `budō`, `Daitō-ryū`, `jūjutsu`, `dōjō`, `ryūha`, `sōke`, `iaidō`, and `aikijūjutsu` are visibly present. Once those are available, rerun `eng`, `eng+jpn`, and any alternate OCR engine candidates on those real samples.

If real samples remain hard to find, keep synthetic macron fixtures for cleanup/serialization regression tests only, not OCR benchmark claims.
