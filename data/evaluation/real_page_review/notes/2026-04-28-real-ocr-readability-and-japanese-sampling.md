# Real OCR Readability and Japanese Sampling Review

Date: 2026-04-28  
Run date: 2026-04-29  
Temporary outputs: `data/notebook_outputs/ocr_text_readability_sampling/`

## Purpose

Validate `readable_text`, line ordering, and Japanese/macron OCR behavior after
canonical OCR box selection was fixed.

This was an experiment/review pass only. Runtime defaults, extraction behavior,
canonical model shape, database schema, and corpus data were unchanged.

## OCR Environment

- OCR engine: Tesseract
- OCR version: 5.5.2
- available languages: 163 installed Tesseract language/script packs, including
  `eng`, `jpn`, and `jpn_vert`
- language configs tested:
  - default project config: `eng`
  - review override: `eng+jpn`
- PSMs tested:
  - default multi-candidate full-page run: `11`, `3`, `6`
  - focused `eng+jpn` subset: explicit PSM `3`, `6`, and `11`
- notes:
  - EasyOCR remained disabled.
  - Auto/full-page runs still store selected PSM, but the default processor
    overwrites selected-result metadata and does not preserve the language string
    in auto-run summaries.
  - Explicit PSM review runs preserve `selected_language=eng+jpn`.

## Pages Reviewed

| Corpus | Sample ID | Path | Page Type | Why Selected |
|---|---|---|---|---|
| DFD | `original_img_3337` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg` | text-heavy | Text-heavy page from prior OCR quality review. |
| DFD | `original_img_3288` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg` | mixed English/Japanese candidate | Manifest expects Japanese/macrons; useful for language sampling. |
| DFD | `original_img_3344` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | diagram/labeled | Known labeled diagram page with surrounding text. |
| DFD | `original_img_3330` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg` | diagram/caption, romanized terms | Contains figure caption terms such as `ryu`/`Kamakura`; good PSM stress case. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_56_38` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.56.38.jpg` | text-heavy | Corpus 2 text-heavy sample with weak OCR. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg` | photo/visual | Photo/article page; useful for line ordering. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg` | broad/mixed | Known broad/mixed article/photo page. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_20_26_02` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 20.26.02.jpg` | noisy/odd layout | Tall/odd layout and low-confidence page. |

## Summary Table

| Corpus | Sample ID | Lang | PSM | Word Count | Line Count | Readable Text Quality | Line Order | Japanese Found | Macrons Found | Main Issue |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| DFD | `original_img_3337` | `eng` | selected 11 | 598 | 58 | usable-with-cleanup | acceptable | no | no | OCR errors remain, but duplication is fixed. |
| DFD | `original_img_3288` | `eng` | selected 3 | 56 | 10 | usable | good | no | no | English lecture list recognized; no Japanese/macrons observed. |
| DFD | `original_img_3344` | `eng` | selected 3 | 438 | 56 | problematic | partial | no | no | Source OCR is noisy; line grouping cannot fix recognition errors. |
| DFD | `original_img_3330` | `eng` | selected 3 | 90 | 10 | usable-with-cleanup | partial | no | no | Caption text useful; figure labels disrupt line order. |
| Corpus 2 | `16_56_38` | `eng` | selected 3 | 31 | 6 | problematic | partial | no | no | OCR quality weak. |
| Corpus 2 | `18_29_28` | `eng` | selected 3 | 360 | 48 | usable-with-cleanup | partial | no | no | Article/photo layout remains noisy. |
| Corpus 2 | `16_55_48` | `eng` | selected 3 | 110 | 22 | usable-with-cleanup | acceptable | no | no | Some OCR errors, but readable text is compact. |
| Corpus 2 | `20_26_02` | `eng` | selected 3 | 28 | 12 | fail | poor | no | no | Odd layout produces mostly noise. |
| DFD | `original_img_3288` | `eng+jpn` | selected 3 | 57 | 11 | usable-with-noise | good | no meaningful Japanese | no | Adds one stray Japanese quote-like mark, no useful Japanese terms. |
| DFD | `original_img_3344` | `eng+jpn` | selected 3 | 447 | 56 | problematic | partial | noisy glyphs | no | Japanese model introduces stray `。`/`本`-like noise, not useful terms. |
| Corpus 2 | `16_56_38` | `eng+jpn` | selected 3 | 37 | 6 | problematic | partial | noisy kana | no | Produces `に ここ ーー` noise, not reliable Japanese. |
| Corpus 2 | `18_29_28` | `eng+jpn` | selected 3 | 369 | 49 | usable-with-cleanup | partial | noisy punctuation/glyphs | no | Slightly different text; no meaningful Japanese terms. |

## Per-Page Notes

### original_img_3337 / eng / auto PSM

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg`

Text output:

- readable_text preview:
  - `three hours, you may fumble and stumble three, four,`
  - `two or six times. So now, if you're a stupid man, you walk away,`
  - `five, that. You hide from the fact. But if you're Smart,`
- obvious OCR errors: `ou'll`, dropped/changed words, punctuation noise

Structure:

- word count: 598
- line count: 58
- line ordering: acceptable for a text-heavy page
- duplicated text: no PSM duplication
- alternate candidates: 3 compact summaries, 1190 non-selected boxes summarized

Japanese/macron/punctuation:

- Japanese found: no
- macrons found: no
- punctuation preserved: mostly, with OCR quote artifacts

Decision: usable-with-cleanup.

### original_img_3288 / eng vs eng+jpn

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg`

Text output:

- `eng` readable_text preview:
  - `THE DRAEGER LECTURES`
  - `AT`
  - `THE UNIVERSITY OF HAWAII.`
- `eng+jpn` readable_text is similar and remains compact.

Structure:

- `eng`: 56 words, 10 lines
- `eng+jpn`: 57 words, 11 lines
- line ordering: good for centered title/list text
- duplicated text: no
- alternate candidates: compact summaries only

Japanese/macron/punctuation:

- Japanese found: no meaningful Japanese text
- macrons found: no
- punctuation preserved: yes for list punctuation; OCR artifacts remain

Decision: usable.

### original_img_3344 / eng vs eng+jpn

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg`

Text output:

- `eng` readable_text preview:
  - `; Ce ol Md thee or`
  - `Bhat ae “wall aa which hinges on esoterics which we calJ`
  - `is by Y it's very nature to be transmitted mouth to nouth`
- `eng+jpn` adds stray Japanese-like glyphs, for example `。 本`, but does not
  recover meaningful Japanese terms.

Structure:

- `eng`: 438 words, 56 lines
- `eng+jpn`: 447 words, 56 lines
- line ordering: partial; source OCR is noisy
- duplicated text: no PSM duplication
- alternate candidates: compact summaries only

Japanese/macron/punctuation:

- Japanese found: only noisy glyphs
- macrons found: no
- punctuation preserved: partial

Decision: problematic. OCR config/language alone does not fix this page.

### original_img_3330 / eng+jpn PSM comparison

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg`

Text output:

- PSM 3: 90 words, 10 lines, compact caption but partial line order
- PSM 6: 300 words, 59 lines, much noisier; many visual/label artifacts
- PSM 11: 138 words, 40 lines, sparse/noisy

Structure:

- line ordering: PSM 3 is best among sampled settings
- duplicated text: no
- alternate candidates: none in explicit single-PSM runs

Japanese/macron/punctuation:

- Japanese found: PSM 6/11 produce stray Japanese-like glyphs, not useful terms
- macrons found: no
- punctuation preserved: em dash/punctuation sometimes preserved

Decision: usable-with-cleanup under PSM 3; problematic under PSM 6/11.

### corpus2_new_doc_2026_04_28_16_56_38 / eng vs eng+jpn

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.56.38.jpg`

Text output:

- `eng` readable_text preview:
  - `Page Loft`
  - `Sot`
  - `Japanese Us`
- `eng+jpn` inserts noisy kana/long-vowel characters such as
  `に ここ ーー ニー ニニ ーー ニー`.

Structure:

- `eng`: 31 words, 6 lines
- `eng+jpn`: 37 words, 6 lines
- line ordering: partial
- duplicated text: no

Japanese/macron/punctuation:

- Japanese found: noisy kana, not useful content
- macrons found: no
- punctuation preserved: limited

Decision: problematic. OCR quality/config is the blocker.

### corpus2_new_doc_2026_04_28_18_29_28 / eng vs eng+jpn

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg`

Text output:

- `eng` readable_text preview:
  - `2`
  - `SATO SENSEI"`
  - `QAIS San. In Tokyo in`
- `eng+jpn` improves one heading fragment to `SATO KINBEI SENSEI"` but still
  introduces layout noise.

Structure:

- `eng`: 360 words, 48 lines
- `eng+jpn`: 369 words, 49 lines
- line ordering: partial; article/photo layout is still hard
- duplicated text: no

Japanese/macron/punctuation:

- Japanese found: no meaningful Japanese text
- macrons found: no
- punctuation preserved: partial

Decision: usable-with-cleanup.

### corpus2_new_doc_2026_04_28_16_55_48 / eng+jpn PSM comparison

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg`

Text output:

- PSM 3: 109 words, 22 lines, compact and closest to useful article text
- PSM 6: 239 words, 27 lines, more noisy, adds stray Japanese-like punctuation
- PSM 11: 167 words, 30 lines, more sparse/noisy

Structure:

- line ordering: PSM 3 is best among sampled settings
- duplicated text: no
- alternate candidates: none in explicit single-PSM runs

Japanese/macron/punctuation:

- Japanese found: PSM 6/11 introduce noisy glyphs, not useful Japanese terms
- macrons found: no
- punctuation preserved: partial

Decision: usable-with-cleanup under PSM 3.

### corpus2_new_doc_2026_04_28_20_26_02 / eng+jpn PSM comparison

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 20.26.02.jpg`

Text output:

- PSM 3: 29 words, 13 lines, mostly noise
- PSM 6: 47 words, 17 lines, still noise
- PSM 11: 50 words, 23 lines, still noise

Structure:

- line ordering: poor because recognition is poor
- duplicated text: no

Japanese/macron/punctuation:

- Japanese found: noisy glyphs only
- macrons found: no
- punctuation preserved: not meaningful

Decision: fail.

## Cross-Config Findings

Comparison:

```text
eng vs eng+jpn
```

- `eng+jpn` is available and runnable.
- It did not produce reliable Japanese terms on the reviewed pages.
- It sometimes inserted stray Japanese punctuation/kana/kanji-like glyphs into
  noisy English pages.
- It occasionally helped an English romanized name fragment, but not enough to
  treat `eng+jpn` as an obvious default.

Comparison:

```text
PSM 6 vs PSM 3 vs PSM 11
```

- PSM 3 was the most consistently useful setting on the focused subset.
- PSM 6 produced more boxes and more noisy lines on figure/caption and mixed
  article pages.
- PSM 11 remained useful for at least one text-heavy DFD page selected by the
  current multi-candidate scorer, but single-PSM subset checks did not show it
  as broadly better.

## Failure Patterns

- Line grouping is now sane in the sense that duplicate PSM words are gone.
- Reading order is acceptable on simple title/list and text-heavy pages.
- Reading order remains partial on figure/caption, photo/article, and odd layout
  pages.
- OCR quality itself is the blocker on some Corpus 2 pages.
- `eng+jpn` without better page/script segmentation creates noisy Japanese-like
  output rather than reliable Japanese terms.
- Real macron detection remains unproven because no sampled real OCR output
  produced macronized terms.

## Decision Questions

### Is readable_text good enough now?

For simple text-heavy or list pages, yes, with cleanup. The duplication bug is
fixed and `readable_text` is compact.

For diagram/caption and mixed article/photo pages, only partially. The remaining
problem is reading order and layout-aware grouping, not duplicate boxes.

### Are line regions ordered well enough for reconstruction?

Not broadly. They are adequate for simple pages, but not yet good enough for
page reconstruction on figure/caption pages or mixed layouts.

### Does eng+jpn produce Japanese-bearing output?

Technically yes, but the observed Japanese-bearing output is mostly noise. It
did not recover meaningful target terms such as:

```text
武道
柔術
術
道
日本
```

### Do macronized romanized terms appear and survive?

No real sampled OCR output contained macronized terms such as:

```text
Daitō
Daitō-ryū
koryū
budō
ō
ū
```

Synthetic cleanup tests still prove preservation once those characters exist,
but real OCR production of macrons remains unvalidated.

### Is the next blocker OCR config, line grouping, or serialization polish?

Primary blocker: line grouping / reading order for mixed layouts.

Secondary blocker: OCR config/language selection. `eng+jpn` should not become a
default based on this sample; it needs script/layout-aware selection or more
targeted Japanese-page sampling.

Serialization is acceptable enough for now.

## Recommended Next Implementation Pass

Choose one primary next pass:

- [ ] OCR config/language selection
- [x] line grouping / reading order improvement
- [ ] DocumentResult/data.json readability polish
- [ ] Japanese analysis promotion
- [ ] page reconstruction improvement
- [ ] more real OCR sampling needed

Recommended scope:

```text
Improve line grouping / reading order using selected OCR word boxes only.
```

Specific candidates:

- filter tiny artifact word boxes before line grouping
- avoid merging figure labels/captions with nearby body text too aggressively
- preserve current behavior for simple text pages
- add real-output fixture summaries from this review as test cases where
  practical

Do not promote Japanese analysis yet. Also do not switch defaults to `eng+jpn`
until a more targeted Japanese-bearing page sample proves it helps.
