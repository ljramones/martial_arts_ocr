# Modern Japanese OCR Focused Evaluation

## Purpose

Evaluate current OCR behavior on curated modern Japanese samples before promoting Japanese analysis into canonical fields.

This pass used the organized modern Japanese OCR corpus layout and safe/local sample sources only:

- `data/corpora/modern_japanese_ocr/project_native/`
- `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/`

The `img_ocr_jp_cn` dataset was not used for benchmark claims because its provenance/license remains unknown.

## Environment

- Tesseract version: 5.5.2
- Installed languages: 163 Tesseract language/script packs available
- Relevant installed languages: `eng`, `jpn`, `jpn_vert`
- OCR configs tested:
  - Tesseract `eng`
  - Tesseract `eng+jpn`
  - Tesseract `jpn`
  - PSM 3, 6, and 11 for each language config
- EasyOCR available: import available in the main `.venv`, but disabled by project config and not tested in this pass
- PaddleOCR available in main env: no
- Notes:
  - Local manifests were created under `modern_japanese_ocr/**/manifests/manifest.local.json`; these remain ignored.
  - Outputs were written under ignored `data/notebook_outputs/modern_japanese_ocr_eval/`.
  - Runtime OCR defaults were not changed.

## Samples Reviewed

| Source | Sample ID | Path | Visible Japanese? | Visible Macrons? | Why Selected |
|---|---|---|---|---|---|
| Project-native DFD | `original_img_3336` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg` | yes | no confirmed macrons | Large Japanese calligraphy plus romanized terms such as `katsujin no ken`, `satsujin no to`, `Yagyu Shinkage Ryu`. |
| Project-native DFD | `original_img_3344` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | yes | no confirmed macrons | Diagram/label page with Japanese-like labels and romanized terms. |
| Project-native Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_34` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg` | yes | no confirmed macrons | Dense martial arts term-list page with Japanese parentheticals and romanized terms. |
| Project-native Corpus 2 | `corpus2_new_doc_2026_04_28_19_41_28` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.41.28.jpg` | yes | no confirmed macrons | Bansenshukai page with Japanese blocks and English translation. |
| Project-native Corpus 2 | `corpus2_new_doc_2026_04_28_19_40_50` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.40.50.jpg` | yes | no confirmed macrons | Bansenshukai/index image with large Japanese cover/title text. |
| Manual sample | `manual_modern_japanese_japtext2` | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/JapText2.jpg` | yes | no | Clean horizontal modern Japanese text sample. |
| Manual sample | `manual_modern_japanese_s_34193423` | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/S__34193423.jpg` | yes | no | Vertical Japanese page with mixed layout/panel text. |
| Manual sample | `manual_modern_japanese_istock_calligraphy` | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/istockphoto-1015263364-612x612.jpg` | yes | no | Stylized vertical calligraphy-like Japanese sample. |

## Summary Table

The table reports the most informative rows from the full 3x3 Tesseract matrix. All samples were run through `eng`, `eng+jpn`, and `jpn` with PSM 3, 6, and 11.

| Sample ID | OCR Config | PSM | Japanese Recovered | Macron Terms Recovered | Output Quality | Main Issue |
|---|---|---:|---|---|---|---|
| `original_img_3336` | `eng` | 3 | no | no | partial | English/romanized terms readable; calligraphy ignored. |
| `original_img_3336` | `eng+jpn` | 6 | noisy fragments | no | partial/noisy | Adds a few Japanese-like characters but no meaningful calligraphy text. |
| `original_img_3336` | `jpn` | 11 | noisy fragments | no | noisy | More Japanese-like characters, but English degrades heavily. |
| `original_img_3344` | `eng` | 6 | no | no | partial/noisy | Diagram/text layout makes English OCR noisy. |
| `original_img_3344` | `eng+jpn` | 11 | noisy fragments | no | noisy | Japanese-like fragments increase without reliable labels. |
| `original_img_3344` | `jpn` | 11 | noisy fragments | no | noisy | Most Japanese-looking output, but not meaningful enough. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `eng` | 3 | no | no | partial | English and romanized terms are most readable; Japanese parentheticals become noise. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `eng+jpn` | 6 | noisy fragments | no | partial/noisy | Only a few Japanese characters/punctuation. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `jpn` | 3/11 | partial fragments | no | partial/noisy | Recovers isolated characters such as `術` or `剣`; not full terms. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `eng` | 6 | no | no | partial | English translation is partially readable; Japanese blocks become Latin noise. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `eng+jpn` | 6 | noisy fragments | no | partial/noisy | Adds Japanese-like fragments but no reliable text block. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `jpn` | 6 | noisy fragments | no | noisy | More Japanese fragments, English worsens. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `eng` | 3 | no | no | fail | Large Japanese cover text not recovered. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `eng+jpn` | 3 | noisy fragments | no | fail | Scattered kana/kanji-like fragments only. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `jpn` | 6 | noisy fragments | no | fail | Still no meaningful cover/title text. |
| `manual_modern_japanese_japtext2` | `eng` | 3 | no | no | fail | Clean Japanese text becomes Latin-like gibberish. |
| `manual_modern_japanese_japtext2` | `eng+jpn` | 3 | no | no | fail | Same as `eng`; mixed language did not recover Japanese. |
| `manual_modern_japanese_japtext2` | `jpn` | 3/6 | yes, incorrect | no | noisy/fail | Emits Japanese characters, but not the visible source text. |
| `manual_modern_japanese_s_34193423` | `eng` | 3 | no | no | fail | Vertical Japanese page becomes Latin-like noise. |
| `manual_modern_japanese_s_34193423` | `eng+jpn` | 6/11 | noisy fragments | no | noisy | More Japanese glyphs, but not reliable text. |
| `manual_modern_japanese_s_34193423` | `jpn` | 6 | partial/noisy | no | noisy | Large Japanese-character output, but ordering/accuracy are poor. |
| `manual_modern_japanese_istock_calligraphy` | `eng` | 3 | no | no | fail | Stylized Japanese becomes Latin-like noise. |
| `manual_modern_japanese_istock_calligraphy` | `eng+jpn` | 6 | noisy fragments | no | noisy/fail | Adds some Japanese fragments. |
| `manual_modern_japanese_istock_calligraphy` | `jpn` | 6 | noisy fragments | no | noisy/fail | Japanese-looking output, but not reliable text. |

## Per-Sample Notes

### `original_img_3336`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg`

Visible source evidence:
- Japanese visible: yes, large calligraphy block.
- Macrons visible: no.
- Expected terms: romanized `katsujin no ken`, `satsujin no to`, `Yagyu Shinkage Ryu`; Japanese calligraphy not manually transcribed.
- Uncertainty: the Japanese source is calligraphy/figure-like, not clean body text.

OCR results:
- `eng`: best for English prose and romanized terms.
- `jpn`: emits more Japanese-like characters but damages English.
- `eng+jpn`: only adds sparse Japanese-like fragments.
- EasyOCR: not tested.

Recovered terms:
- Japanese: not meaningfully recovered.
- Romanized/macron: non-macron romanized terms are partly readable; no macrons.
- Punctuation: Japanese punctuation/fragments appear under `jpn`.

Quality: partial for English; noisy/fail for Japanese.

Decision: usable for Japanese analysis? not yet.

### `original_img_3344`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg`

Visible source evidence:
- Japanese visible: yes, label/diagram-level text.
- Macrons visible: no.
- Expected terms: diagram labels and romanized esoteric/martial arts terms.
- Uncertainty: mixed diagram layout is not a clean Japanese OCR target.

OCR results:
- `eng`: partial/noisy English.
- `jpn`: most Japanese-like output but not reliable.
- `eng+jpn`: still mostly noisy fragments.
- EasyOCR: not tested.

Recovered terms:
- Japanese: no reliable terms.
- Romanized/macron: no macrons.
- Punctuation: `・`, `「`, and `」` appear in some `jpn` runs.

Quality: noisy.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_18_54_34`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg`

Visible source evidence:
- Japanese visible: yes, term-list parentheticals.
- Macrons visible: no.
- Expected terms: martial arts terms and Japanese characters in parentheses.
- Uncertainty: dense small text and mixed English/Japanese typography.

OCR results:
- `eng`: best overall readable text, but Japanese parentheticals are lost.
- `jpn`: recovers isolated Japanese characters/fragments.
- `eng+jpn`: only slight Japanese signal; worse than `jpn` for Japanese fragments.
- EasyOCR: not tested.

Recovered terms:
- Japanese: isolated `術` or `剣` in some `jpn` runs; not stable full terms.
- Romanized/macron: non-macron romanized terms partly readable under `eng`; no macrons.
- Punctuation: `・` appears.

Quality: partial/noisy.

Decision: usable for Japanese analysis? not yet.

### `corpus2_new_doc_2026_04_28_19_41_28`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.41.28.jpg`

Visible source evidence:
- Japanese visible: yes, Japanese blocks with English translation.
- Macrons visible: no.
- Expected terms: Japanese block text and Bansenshukai/translation context.
- Uncertainty: mixed-language page likely needs region-specific OCR.

OCR results:
- `eng`: partial English translation.
- `jpn`: noisy Japanese-like fragments and degraded English.
- `eng+jpn`: sparse fragments only.
- EasyOCR: not tested.

Recovered terms:
- Japanese: fragments only.
- Romanized/macron: no macrons.
- Punctuation: occasional Japanese punctuation under `jpn`.

Quality: partial/noisy.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_19_40_50`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.40.50.jpg`

Visible source evidence:
- Japanese visible: yes, large cover/title text.
- Macrons visible: no.
- Expected terms: cover/title Japanese text.
- Uncertainty: may need cropping/rotation or region-specific preprocessing.

OCR results:
- `eng`: fail.
- `jpn`: fail/noisy fragments.
- `eng+jpn`: fail/noisy fragments.
- EasyOCR: not tested.

Recovered terms:
- Japanese: no meaningful terms.
- Romanized/macron: none.
- Punctuation: not useful.

Quality: fail.

Decision: usable for Japanese analysis? no.

### `manual_modern_japanese_japtext2`

Input path: `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/JapText2.jpg`

Visible source evidence:
- Japanese visible: yes, clean horizontal Japanese.
- Macrons visible: no.
- Expected terms: modern Japanese sentence text, including visible `日本語`.
- Uncertainty: image provenance unclear, but it is a strong OCR smoke test.

OCR results:
- `eng`: Latin-like gibberish.
- `jpn`: emits Japanese characters, but not the visible sentence text.
- `eng+jpn`: same failure pattern as `eng`, no useful Japanese recovery.
- EasyOCR: not tested.

Recovered terms:
- Japanese: Japanese characters appear under `jpn`, but expected terms such as `日本` were not recovered.
- Romanized/macron: none.
- Punctuation: none meaningful.

Quality: noisy/fail.

Decision: usable for Japanese analysis? no, but useful as a clear failure case.

### `manual_modern_japanese_s_34193423`

Input path: `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/S__34193423.jpg`

Visible source evidence:
- Japanese visible: yes, vertical Japanese page.
- Macrons visible: no.
- Expected terms: vertical Japanese text and panel/sidebar text.
- Uncertainty: provenance unclear; layout is mixed and vertical.

OCR results:
- `eng`: Latin-like noise.
- `jpn`: strongest Japanese-character output, especially PSM 6, but text is not reliably ordered or accurate.
- `eng+jpn`: more Japanese fragments than `eng`, still unreliable.
- EasyOCR: not tested.

Recovered terms:
- Japanese: partial/noisy; one term-list search hit included `術`, but the output is not reliable.
- Romanized/macron: none.
- Punctuation: occasional `「` / `・`.

Quality: noisy.

Decision: usable for Japanese analysis? not yet.

### `manual_modern_japanese_istock_calligraphy`

Input path: `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/istockphoto-1015263364-612x612.jpg`

Visible source evidence:
- Japanese visible: yes, stylized vertical calligraphy-like text.
- Macrons visible: no.
- Expected terms: unknown; no ground truth.
- Uncertainty: likely a difficult OCR target and not representative clean body text.

OCR results:
- `eng`: Latin-like noise.
- `jpn`: Japanese-looking fragments but not reliable transcription.
- `eng+jpn`: noisy fragments.
- EasyOCR: not tested.

Recovered terms:
- Japanese: noisy fragments.
- Romanized/macron: none.
- Punctuation: none meaningful.

Quality: noisy/fail.

Decision: usable for Japanese analysis? no.

## Cross-Config Findings

`eng` remains the best full-page path for English-heavy project-native pages and romanized non-macron terms. It does not recover Japanese text.

`eng+jpn` proves mixed-language Tesseract can emit Japanese glyphs in the current OCR pipeline, but it is not a reliable solution. It often adds only a handful of Japanese characters on project-native pages and does not fix clean/manual Japanese samples.

`jpn` is the strongest signal for Japanese-character emission. It recovered isolated useful characters on the Corpus 2 term-list page and produced the most Japanese text on the vertical manual page. It also badly degrades English-heavy pages and does not reliably recover expected full Japanese terms.

PSM 3/6/11 did not produce a clear universal winner:

- PSM 6 gave the most Japanese-character output on the vertical manual page.
- PSM 3 and 11 were sometimes less noisy for project-native pages.
- The clean horizontal `JapText2.jpg` sample failed across all tested PSMs.

EasyOCR was import-available in the main `.venv`, but the project config has EasyOCR disabled and the helper does not currently expose a safe EasyOCR-only review path. It was not tested in this pass.

## Failure Patterns

- Current full-page Tesseract language selection is too coarse for mixed English/Japanese pages.
- Clean Japanese visibility in the source does not guarantee meaningful current OCR output.
- `jpn` emits Japanese characters but does not reliably recover the visible source text.
- `eng+jpn` is not enough as a default mixed-language setting.
- Vertical Japanese and calligraphy-like text need separate orientation/preprocessing evaluation.
- No selected source sample visibly contained macronized romanized Japanese, so real macron OCR remains unproven.

## Decision

- [ ] Current OCR stream is reliable enough to promote Japanese analysis into canonical fields
- [ ] More project-native sample selection is needed
- [x] Tesseract config/language strategy needs work
- [x] Preprocessing for Japanese text needs work
- [x] Another OCR engine should be evaluated for Japanese
- [x] Do not promote Japanese fields yet

## Recommended Next Step

Build a region-specific Japanese OCR comparison before changing canonical models:

1. Add an experiment-only path that can run cropped Japanese regions rather than full pages.
2. Compare Tesseract `jpn`, `jpn_vert`, `eng+jpn`, and EasyOCR if it can run without new downloads.
3. Test light preprocessing on the manual clean horizontal sample, vertical page sample, and one project-native term-list region.
4. Keep English full-page OCR as the baseline document text stream.

Japanese analysis should remain out of first-class canonical fields until a reliable OCR source for Japanese text is demonstrated.
