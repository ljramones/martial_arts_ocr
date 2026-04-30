# Focused Japanese OCR Evaluation

## Purpose

Evaluate whether real source pages containing Japanese or romanized Japanese text produce reliable OCR output before promoting Japanese analysis into canonical model fields.

This pass is evidence-gathering only. Runtime defaults, OCR defaults, extraction behavior, document serialization, and canonical model fields were not changed.

## Environment

- Tesseract version: 5.5.2
- Installed Tesseract languages: 163 language/script packs available from `/opt/homebrew/share/tessdata/`
- Relevant installed languages: `eng`, `jpn`, `jpn_vert`
- Languages tested: `eng`, `eng+jpn`, `jpn`
- PSMs tested:
  - Full 11-page set: PSM 3
  - Stronger Japanese candidates subset: PSM 6 and PSM 11 for `eng+jpn` and `jpn`
- Helper command(s):
  - `.venv/bin/python experiments/review_real_ocr_text_quality.py --sample-id ... --language eng --psm 3 --output-dir data/notebook_outputs/focused_japanese_ocr_eval/eng_psm3`
  - `.venv/bin/python experiments/review_real_ocr_text_quality.py --sample-id ... --language eng+jpn --psm 3 --output-dir data/notebook_outputs/focused_japanese_ocr_eval/eng_jpn_psm3`
  - `.venv/bin/python experiments/review_real_ocr_text_quality.py --sample-id ... --language jpn --psm 3 --output-dir data/notebook_outputs/focused_japanese_ocr_eval/jpn_psm3`
  - Subset PSM 6/11 runs under `data/notebook_outputs/focused_japanese_ocr_eval/*_subset`
- Output directory: `data/notebook_outputs/focused_japanese_ocr_eval/`
- Generated contact sheets: `data/notebook_outputs/focused_japanese_ocr_eval/contact_sheets/`

Generated OCR outputs and contact sheets are ignored review artifacts and were not staged.

## Page Selection

Candidate pages were selected from contact-sheet review of DFD and Corpus 2 images, prioritizing visible Japanese, Japanese labels/calligraphy, martial arts term lists, and romanized Japanese terms. The manifests broadly mark many pages as Japanese/macron candidates, so visual inspection was necessary.

| Corpus | Sample ID | Path | Visible Japanese? | Visible Macrons? | Why Selected |
|---|---|---|---|---|---|
| DFD | `original_img_3336` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg` | yes | no confirmed macron diacritics | Large calligraphy block plus romanized terms: `katsujin no ken`, `satsujin no to`, `Yagyu Shinkage Ryu`. |
| DFD | `original_img_3344` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | yes | no confirmed macron diacritics | Diagram/labels and romanized martial arts/esoteric terms. |
| DFD | `original_img_3353` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3353.jpg` | yes | no confirmed macron diacritics | Figure page with visible Japanese labels and romanized school names. |
| DFD | `original_img_3334` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3334.jpg` | yes, small/uncertain | no confirmed macron diacritics | Sword/dragon figure with a small Japanese-like inscription and romanized terms. |
| DFD | `original_img_3330` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg` | uncertain | no confirmed macron diacritics | Romanized caption/term contrast page from prior review set. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_19_40_50` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.40.50.jpg` | yes | no confirmed macron diacritics | Bansenshukai/index page with large Japanese book-cover text. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_19_41_28` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.41.28.jpg` | yes | no confirmed macron diacritics | Bansenshukai page with Japanese blocks and English translation. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_19_42_44` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.42.44.jpg` | yes | no confirmed macron diacritics | Bansenshukai page with Japanese text block and English translation. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_34` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg` | yes | no confirmed macron diacritics | Martial arts term-list page with Japanese terms in parentheses and English prose. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_52_30` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.52.30.jpg` | yes | no confirmed macron diacritics | Takamatsu/autobiography page with Japanese title block and English prose. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_56_43` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.56.43.jpg` | yes, caption-level | no confirmed macron diacritics | Photo/caption page with Japanese caption-like content. |

## Summary Table

The table below summarizes the main PSM 3 comparison. PSM 6/11 were then run on the strongest Japanese candidates and are summarized in Cross-Config Findings.

| Sample ID | Lang | PSM | Japanese Recovered | Macron Terms Recovered | Output Quality | Main Issue |
|---|---|---:|---|---|---|---|
| `original_img_3336` | `eng` | 3 | no | no | partial | English prose and romanized terms readable; calligraphy ignored. |
| `original_img_3336` | `eng+jpn` | 3 | noisy | no | partial | Adds isolated Japanese-like glyphs but does not recover calligraphy terms. |
| `original_img_3336` | `jpn` | 3 | noisy | no | noisy | Adds punctuation/kana-like noise and degrades English. |
| `original_img_3344` | `eng` | 3 | no | no | partial | English OCR noisy due layout/diagram. |
| `original_img_3344` | `eng+jpn` | 3 | noisy | no | noisy | Isolated Japanese-like glyphs, no reliable target terms. |
| `original_img_3344` | `jpn` | 3 | noisy | no | noisy | More Japanese-like punctuation/fragments, English heavily damaged. |
| `original_img_3353` | `eng` | 3 | no | no | partial | English/romanized terms partial; labels not meaningfully recovered. |
| `original_img_3353` | `eng+jpn` | 3 | noisy | no | partial | Japanese-like fragments only. |
| `original_img_3353` | `jpn` | 3 | noisy | no | noisy | Adds fragments but not useful terms. |
| `original_img_3334` | `eng` | 3 | no | no | partial | Small inscription not recovered. |
| `original_img_3334` | `eng+jpn` | 3 | no | no | partial | No useful Japanese improvement. |
| `original_img_3334` | `jpn` | 3 | noisy | no | noisy | Japanese-like punctuation/fragments only. |
| `original_img_3330` | `eng` | 3 | no | no | partial | Romanized/caption page; no visible macron evidence. |
| `original_img_3330` | `eng+jpn` | 3 | no | no | partial | No useful Japanese improvement. |
| `original_img_3330` | `jpn` | 3 | noisy | no | noisy | Adds Japanese-like fragments. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `eng` | 3 | no | no | fail | Large Japanese cover text not recognized meaningfully. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `eng+jpn` | 3 | noisy | no | fail | Only scattered kana-like output. |
| `corpus2_new_doc_2026_04_28_19_40_50` | `jpn` | 3 | noisy | no | fail | Still does not recover meaningful cover text. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `eng` | 3 | no | no | partial | English translation partly readable; Japanese block becomes Latin noise. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `eng+jpn` | 3 | noisy | no | partial/noisy | Adds isolated kana/kanji-like fragments. |
| `corpus2_new_doc_2026_04_28_19_41_28` | `jpn` | 3 | noisy | no | noisy | Japanese fragments increase, English worsens. |
| `corpus2_new_doc_2026_04_28_19_42_44` | `eng` | 3 | no | no | partial | English translation partly readable. |
| `corpus2_new_doc_2026_04_28_19_42_44` | `eng+jpn` | 3 | noisy | no | partial/noisy | Japanese-like fragments only. |
| `corpus2_new_doc_2026_04_28_19_42_44` | `jpn` | 3 | noisy | no | noisy | Fragmentary Japanese-like output. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `eng` | 3 | no | no | partial | Romanized/English term list readable but Japanese parentheticals become Latin/noise. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `eng+jpn` | 3 | noisy | no | partial/noisy | Some Japanese-like fragments; no stable target compounds. |
| `corpus2_new_doc_2026_04_28_18_54_34` | `jpn` | 3 | partial | no | partial/noisy | Recovers isolated characters such as `術` and punctuation, but not reliable terms. |
| `corpus2_new_doc_2026_04_28_18_52_30` | `eng` | 3 | no | no | partial | English prose very noisy. |
| `corpus2_new_doc_2026_04_28_18_52_30` | `eng+jpn` | 3 | noisy | no | noisy | Adds isolated Japanese-like characters. |
| `corpus2_new_doc_2026_04_28_18_52_30` | `jpn` | 3 | noisy | no | noisy | Japanese-like fragments, English worsens. |
| `corpus2_new_doc_2026_04_28_18_56_43` | `eng` | 3 | no | no | partial | Caption/image page, sparse text. |
| `corpus2_new_doc_2026_04_28_18_56_43` | `eng+jpn` | 3 | noisy | no | noisy | Fragmentary Japanese-like output. |
| `corpus2_new_doc_2026_04_28_18_56_43` | `jpn` | 3 | noisy | no | noisy/fail | Sparse fragments only. |

## Per-Page Notes

### `original_img_3336`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg`

Visible source evidence:
- Japanese visible: yes, large calligraphy block.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: `katsujin no ken`, `satsujin no to`, `Yagyu Shinkage Ryu`; Japanese calligraphy present but not manually transcribed in this pass.
- Uncertainty: calligraphy is stylized and may not be suitable for full-page OCR.

OCR results:
- `eng`: English prose and romanized terms are the most readable.
- `eng+jpn`: similar English output plus isolated Japanese-like glyphs, for example stray `上` / kana-like fragments.
- `jpn`: degrades English substantially and produces scattered Japanese-like punctuation/fragments.

Recovered terms:
- Japanese: no meaningful target Japanese term recovered.
- Romanized/macron: romanized non-macron terms like `katsujin no ken` and `Yagyu Shinkage Ryu` are partially readable under `eng`/`eng+jpn`.
- Punctuation: `jpn` recovers Japanese punctuation-like marks, but not as reliable text.

Quality: partial.

Decision: usable for Japanese analysis? not yet.

### `original_img_3344`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg`

Visible source evidence:
- Japanese visible: yes, labels/diagram-level content.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: diagram labels and romanized/esoteric terms.
- Uncertainty: text is mixed with diagrams and page distortion.

OCR results:
- `eng`: noisy but partially readable English.
- `eng+jpn`: adds Japanese-like fragments and punctuation but does not recover clear Japanese labels.
- `jpn`: more Japanese-like fragments, heavy English degradation.

Recovered terms:
- Japanese: no reliable target terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: `jpn` PSM 6/11 sometimes emits `・`, `「`, or `」`.

Quality: noisy.

Decision: usable for Japanese analysis? no.

### `original_img_3353`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3353.jpg`

Visible source evidence:
- Japanese visible: yes, labels/vertical text.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: Japanese labels and romanized school names.
- Uncertainty: image/label geometry is not full-page prose.

OCR results:
- `eng`: romanized/English content is partially recovered.
- `eng+jpn`: Japanese-like fragments only.
- `jpn`: more fragments and punctuation, not useful Japanese text.

Recovered terms:
- Japanese: no reliable target terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: isolated `・` in `jpn`.

Quality: partial/noisy.

Decision: usable for Japanese analysis? no.

### `original_img_3334`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3334.jpg`

Visible source evidence:
- Japanese visible: yes, but small/uncertain inscription-level text.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: romanized terms and possible inscription.
- Uncertainty: inscription may be too small/stylized for full-page OCR.

OCR results:
- `eng`: partial English/romanized output.
- `eng+jpn`: no useful Japanese gain.
- `jpn`: noisy Japanese-like fragments only.

Recovered terms:
- Japanese: no reliable target terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: isolated punctuation under `jpn`.

Quality: partial/noisy.

Decision: usable for Japanese analysis? no.

### `original_img_3330`

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg`

Visible source evidence:
- Japanese visible: uncertain.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: romanized/caption text.
- Uncertainty: included mainly as contrast from prior review set.

OCR results:
- `eng`: partial English/caption output.
- `eng+jpn`: no useful Japanese gain.
- `jpn`: noisy Japanese-like fragments.

Recovered terms:
- Japanese: no reliable target terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: isolated punctuation under `jpn`.

Quality: partial.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_19_40_50`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.40.50.jpg`

Visible source evidence:
- Japanese visible: yes, large Japanese book-cover text.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: cover/title text.
- Uncertainty: page appears as book-cover/index imagery and may need cropping/rotation/region OCR.

OCR results:
- `eng`: fails, mostly Latin gibberish.
- `eng+jpn`: scattered kana-like fragments, no meaningful title.
- `jpn`: scattered kana-like fragments, no meaningful title.

Recovered terms:
- Japanese: no meaningful terms recovered.
- Romanized/macron: none.
- Punctuation: no useful punctuation result.

Quality: fail.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_19_41_28`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.41.28.jpg`

Visible source evidence:
- Japanese visible: yes, Japanese blocks plus English translation.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: Japanese text blocks and English translation.
- Uncertainty: mixed-language layout may require region-specific OCR rather than full-page language mixing.

OCR results:
- `eng`: English translation partially readable; Japanese block becomes Latin noise.
- `eng+jpn`: adds sparse kana/kanji-like fragments but no stable Japanese terms.
- `jpn`: increases Japanese-like fragments while damaging English.

Recovered terms:
- Japanese: fragments only.
- Romanized/macron: no macron terms recovered.
- Punctuation: occasional Japanese punctuation under `jpn`.

Quality: partial/noisy.

Decision: usable for Japanese analysis? not yet.

### `corpus2_new_doc_2026_04_28_19_42_44`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.42.44.jpg`

Visible source evidence:
- Japanese visible: yes, Japanese text block and English translation.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: Japanese block and English translation.
- Uncertainty: mixed block layout.

OCR results:
- `eng`: English translation partly readable.
- `eng+jpn`: Japanese-like fragments only.
- `jpn`: fragmentary Japanese-like output and degraded English.

Recovered terms:
- Japanese: no reliable target terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: not meaningful.

Quality: partial/noisy.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_18_54_34`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg`

Visible source evidence:
- Japanese visible: yes, term-list parentheticals and mixed Japanese/English martial arts list.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: Japanese martial arts terms in lists, romanized terms such as `Kenjutsu`, `Jujutsu`, `Yari`, `Naginata`.
- Uncertainty: dense small text and mixed punctuation.

OCR results:
- `eng`: English and romanized martial arts terms are partially readable; Japanese parentheticals become noise.
- `eng+jpn`: not materially better for Japanese terms.
- `jpn`: best Japanese-fragment run; recovers isolated characters such as `術`, `剣`, `刀`, and punctuation in some PSMs, but not stable compounds like `武道` or `柔術`.

Recovered terms:
- Japanese: partial fragments only; strongest useful hit was isolated `剣` / `術` under `jpn`.
- Romanized/macron: non-macron romanized terms are partially readable; no macron terms recovered.
- Punctuation: `・` appears frequently.

Quality: partial/noisy.

Decision: usable for Japanese analysis? not yet.

### `corpus2_new_doc_2026_04_28_18_52_30`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.52.30.jpg`

Visible source evidence:
- Japanese visible: yes, title/header block.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: Japanese title/header, English prose.
- Uncertainty: scan quality and mixed-language layout are poor.

OCR results:
- `eng`: English prose noisy but more useful than Japanese configs.
- `eng+jpn`: adds Japanese-like fragments but no reliable terms.
- `jpn`: fragments and punctuation; English deteriorates.

Recovered terms:
- Japanese: no reliable terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: occasional `・` or `」`.

Quality: noisy.

Decision: usable for Japanese analysis? no.

### `corpus2_new_doc_2026_04_28_18_56_43`

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.56.43.jpg`

Visible source evidence:
- Japanese visible: yes, caption-level or nearby text.
- Macrons visible: no confirmed macron diacritics.
- Terms expected: caption/label text.
- Uncertainty: sparse text and visual page.

OCR results:
- `eng`: sparse partial output.
- `eng+jpn`: sparse Japanese-like fragments.
- `jpn`: sparse fragments, not meaningful.

Recovered terms:
- Japanese: no reliable terms.
- Romanized/macron: no macron terms recovered.
- Punctuation: not useful.

Quality: noisy/fail.

Decision: usable for Japanese analysis? no.

## Cross-Config Findings

### `eng` vs `eng+jpn`

`eng` remains better for English prose and romanized non-macron terms. `eng+jpn` successfully activates Japanese recognition and often emits kana/kanji-like fragments, but on these full-page samples it did not reliably recover target Japanese compounds. It also introduces extra noise into already noisy English pages.

The most useful `eng+jpn` result is diagnostic: it proves the Japanese language pack runs and that Japanese-like glyphs flow through the current cleanup/serialization stack. It does not yet provide a stable Japanese text stream.

### `jpn`

`jpn` produces the most Japanese-like output, especially punctuation and isolated characters. On `corpus2_new_doc_2026_04_28_18_54_34`, it recovered isolated characters such as `術` and `剣` in some PSMs. However, it damages English prose and does not reliably recover full terms such as `武道`, `柔術`, `道場`, or `日本`.

This suggests the next useful OCR strategy is probably region/language selection rather than replacing full-page English OCR with full-page Japanese OCR.

### PSM 3 vs 6 vs 11

PSM 3 was adequate for the broad comparison. PSM 6 and PSM 11 did not change the conclusion:

- `eng+jpn` PSM 6/11 still produced mostly scattered Japanese-like fragments.
- `jpn` PSM 6/11 sometimes recovered more punctuation and isolated characters.
- PSM 11 was slightly useful on the term-list page, where `jpn` recovered isolated `剣` and punctuation, but it was still not a reliable Japanese text result.
- None of the tested PSMs recovered macronized romanized terms because no confirmed macron-bearing source text was found in this curated set.

## Failure Patterns

- Full-page mixed-language OCR is the wrong granularity for many sampled pages.
- `eng+jpn` is not enough by itself; it adds Japanese-capable recognition but does not reliably separate English prose, Japanese blocks, captions, and diagrams.
- `jpn` can recover isolated Japanese fragments, but English text becomes much worse.
- Dense term lists with small Japanese parentheticals need preprocessing or region-specific OCR.
- Large/stylized calligraphy and cover/title imagery are poorly handled by full-page Tesseract.
- No sampled page provided confirmed visible macron diacritics, so real macron OCR remains unproven.

## Decision Questions

### Do selected pages visibly contain Japanese/macrons?

The selected pages do visibly contain Japanese or Japanese-like source material. The strongest examples are `original_img_3336`, `corpus2_new_doc_2026_04_28_19_40_50`, `corpus2_new_doc_2026_04_28_19_41_28`, `corpus2_new_doc_2026_04_28_19_42_44`, and `corpus2_new_doc_2026_04_28_18_54_34`.

Visible macron diacritics were not confirmed in this set. The pages contain romanized Japanese terms, but mostly in non-macron form such as `Ryu`, `Yagyu`, `Kenjutsu`, and `Jujutsu`.

### Does Tesseract recover meaningful Japanese/macron terms?

Not reliably.

Tesseract with `jpn` can recover isolated Japanese characters on the better term-list sample, including fragments such as `術` and `剣`. It does not reliably recover complete target terms. `eng+jpn` mostly emits noisy Japanese-like glyphs rather than useful Japanese text.

Macronized terms were not recovered from real OCR because they were not confirmed in the selected source pages.

### Is the blocker OCR config, sample selection, preprocessing, or engine choice?

The immediate blockers are:

1. OCR config/language strategy: full-page `eng+jpn` is not enough; pages likely need region-specific language routing.
2. Preprocessing: Japanese text is often small, embedded in captions, in term-list parentheses, vertical, stylized, or part of figure imagery.
3. Sample selection for macrons: the sampled pages did not visibly prove macron-bearing source text.

Engine choice may become a blocker, but this pass does not prove that yet. Tesseract can emit some Japanese fragments when the `jpn` language pack is used, so the next falsification step should be more targeted region/preprocessing evaluation before abandoning Tesseract.

## Recommendation

- [ ] Real OCR stream is reliable enough to promote Japanese analysis into canonical fields
- [ ] More targeted sample selection is needed
- [x] OCR config/language strategy needs work
- [x] Image preprocessing for Japanese text needs work
- [ ] Tesseract is insufficient; evaluate another OCR engine for Japanese
- [x] Do not promote Japanese fields yet

Recommended next implementation/evaluation branch:

```text
Region-specific Japanese OCR evaluation:
  - manually crop or automatically select visible Japanese blocks/labels
  - compare eng, jpn, eng+jpn, and jpn_vert where appropriate
  - test light preprocessing on those regions only
  - keep English full-page OCR as the baseline text stream
```

Japanese analysis should not be promoted into first-class canonical fields until the OCR stream can produce reliable Japanese terms from confirmed Japanese source regions.
