# Real OCR Text Quality Review

Date: 2026-04-28  
Run date: 2026-04-29  
Temporary output: `data/notebook_outputs/ocr_text_quality_review/`

## Purpose

Validate the OCR cleanup and text hierarchy path on representative real pages.

Reviewed path:

```text
real OCR output
  -> OCRPostProcessor
  -> TextCleaner
  -> OCR adapters
  -> word TextRegions
  -> derived line TextRegions
  -> PageResult.metadata.readable_text
  -> DocumentResult.to_dict()
```

No OCR settings, extraction behavior, runtime defaults, database schema, or corpus
data were changed.

## OCR Environment

- OCR engine: Tesseract
- OCR version: 5.5.2
- language config: `eng`
- PSM / engine settings: processor tried full-page PSM `11`, `3`, and `6`; selected best result per page
- word boxes available: yes, from Tesseract word-level `image_to_data`
- line boxes available: no engine line boxes; line regions are derived by normalization
- EasyOCR: disabled by config
- Japanese OCR language: not enabled in this run

Selected PSM:

| Sample ID | Selected PSM | Best Word Boxes | All PSM Word Boxes |
|---|---:|---:|---|
| `original_img_3337` | 11 | 598 | 11:598, 3:595, 6:595 |
| `original_img_3288` | 3 | 56 | 11:61, 3:56, 6:65 |
| `original_img_3344` | 3 | 438 | 11:475, 3:438, 6:475 |
| `original_img_3330` | 3 | 90 | 11:97, 3:90, 6:248 |
| `corpus2_new_doc_2026_04_28_16_56_38` | 3 | 31 | 11:59, 3:31, 6:159 |
| `corpus2_new_doc_2026_04_28_18_29_28` | 3 | 360 | 11:354, 3:360, 6:443 |
| `corpus2_new_doc_2026_04_28_16_55_48` | 3 | 110 | 11:169, 3:110, 6:232 |
| `corpus2_new_doc_2026_04_28_20_26_02` | 3 | 28 | 11:49, 3:28, 6:50 |

## Pages Reviewed

| Corpus | Sample ID | Path | Page Type | Notes |
|---|---|---|---|---|
| DFD | `original_img_3337` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg` | text-heavy | From broader review; text-heavy page handled cleanly by image detector. |
| DFD | `original_img_3288` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg` | mixed English/Japanese candidate | Title/lecture list page; manifest expected Japanese/macrons but OCR produced English-only output. |
| DFD | `original_img_3344` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | diagram/labeled | Known labeled diagram page. |
| DFD | `original_img_3330` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg` | noisy/odd layout | Tall visual-strip/caption page. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_56_38` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.56.38.jpg` | text-heavy | Text-heavy page from generalization sample. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg` | photo/visual | Page where useful visual content was previously retained. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg` | broad/mixed | Known broad mixed text/photo case. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_20_26_02` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 20.26.02.jpg` | noisy/odd layout | Tall strip / odd layout case. |

## Summary Table

| Corpus | Sample ID | Raw Text Quality | Cleaned Text Quality | Readable Text Quality | Word Boxes | Line Regions | Japanese/Macrons | Main Issue |
|---|---|---|---|---|---:|---:|---|---|
| DFD | `original_img_3337` | usable-with-errors | usable-with-cleanup | fail | 2386 | 58 | not observed | OCR boxes duplicated across PSM candidates; line text repeats words heavily. |
| DFD | `original_img_3288` | usable | usable-with-cleanup | fail | 238 | 17 | not observed | Lecture-list text survives, but readable lines duplicate words and add noise. |
| DFD | `original_img_3344` | problematic | problematic | fail | 1826 | 60 | not observed | Low-quality OCR plus duplicated boxes make line regions unusable. |
| DFD | `original_img_3330` | usable-with-cleanup | usable-with-cleanup | fail | 525 | 23 | not observed | Caption text is captured, but layout/figure labels make derived lines noisy. |
| Corpus 2 | `16_56_38` | problematic | problematic | fail | 280 | 15 | not observed | OCR quality is poor and duplicated boxes inflate readable text. |
| Corpus 2 | `18_29_28` | usable-with-errors | usable-with-errors | fail | 1517 | 53 | not observed | Text is recognizable in places; derived lines repeat words and merge noise. |
| Corpus 2 | `16_55_48` | usable-with-cleanup | usable-with-cleanup | fail | 621 | 30 | not observed | Cleaned text is useful; derived line text is much noisier than cleaned text. |
| Corpus 2 | `20_26_02` | fail | fail | fail | 155 | 20 | not observed | Low confidence odd-layout page; OCR output mostly noise. |

## Per-Page Notes

### original_img_3337

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920001/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920001/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920001/text.txt`
- summary: `data/notebook_outputs/ocr_text_quality_review/summary.json`

Text quality:

- raw OCR: recognizable typewritten English but fragmented by blank lines
- cleaned OCR: useful paragraph text; line breaks now survive
- readable_text: unusable because words are repeated from multiple PSM box sets
- line breaks: cleaned text keeps meaningful structure better than derived lines
- obvious OCR errors: `ou'll`, `nyself`, punctuation artifacts

Structure:

- word regions: 2386, but best result had only 598 boxes
- line regions: 58 derived lines
- metadata: `readable_text`, `ocr_word_count`, `ocr_line_count`, and `ocr_text_boxes` are present
- boxes useful: yes for geometry, but duplicated

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: mostly, with OCR quote artifacts

Decision: usable-with-cleanup for text; fail for current derived `readable_text`.

### original_img_3288

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920002/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920002/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920002/text.txt`

Text quality:

- raw OCR: title and list content recognized
- cleaned OCR: useful one-page lecture list
- readable_text: duplicates title/list words and includes small artifact regions
- line breaks: raw title/list breaks partly collapse after cleanup, but not from whitespace flattening
- obvious OCR errors: `&=*=ESOTERIC`, trailing punctuation artifact

Structure:

- word regions: 238, while best result had 56 boxes
- line regions: 17 derived lines
- metadata: complete and visible in `data.json`
- boxes useful: likely useful after deduplication

Japanese/macron/punctuation:

- Japanese preserved: not observed in OCR output
- macrons preserved: not observed in OCR output
- punctuation preserved: ampersands and punctuation survive, but OCR noise remains

Decision: usable-with-cleanup for text; problematic for derived line hierarchy.

### original_img_3344

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920003/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920003/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920003/text.txt`

Text quality:

- raw OCR: noisy but contains recognizable English phrases
- cleaned OCR: cleanup improves spacing, but many OCR errors remain
- readable_text: unusable due repeated words and tiny artifact boxes
- line breaks: cleaned text is more readable than derived line text
- obvious OCR errors: `nouth`, `taliJ`, `comprehand`, quote artifacts

Structure:

- word regions: 1826, while best result had 438 boxes
- line regions: 60 derived lines
- metadata: present
- boxes useful: OCR boxes exist but need dedupe/filtering before line grouping

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: partially, with noisy quote/mark recognition

Decision: problematic.

### original_img_3330

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920004/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920004/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920004/text.txt`

Text quality:

- raw OCR: captures figure labels and caption text
- cleaned OCR: caption becomes a useful paragraph
- readable_text: noisy because line grouping sees figure labels and duplicate PSM boxes as text lines
- line breaks: cleaned caption structure is acceptable
- obvious OCR errors: `daw` for claw, missing spaces after quoted period

Structure:

- word regions: 525, while best result had 90 boxes
- line regions: 23 derived lines
- metadata: present
- boxes useful: useful for local labels, but not yet for readable page structure

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: yes, but quote spacing needs cleanup later

Decision: usable-with-cleanup for caption text; problematic for line hierarchy.

### corpus2_new_doc_2026_04_28_16_56_38

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.56.38.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920005/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920005/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920005/text.txt`

Text quality:

- raw OCR: low-quality text, but some words are recognizable
- cleaned OCR: preserves text and line breaks but cannot repair recognition errors
- readable_text: unreadable due duplicated boxes and poor OCR
- line breaks: present
- obvious OCR errors: `Page Loft`, `Jeeesn`, `vated pose`

Structure:

- word regions: 280, while best result had 31 boxes
- line regions: 15 derived lines
- metadata: present
- boxes useful: likely poor until OCR/PSM choice improves

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: limited

Decision: problematic.

### corpus2_new_doc_2026_04_28_18_29_28

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920006/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920006/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920006/text.txt`

Text quality:

- raw OCR: recognizes some article text and headings
- cleaned OCR: still useful in fragments
- readable_text: noisy and repetitive
- line breaks: cleaned text preserves line structure
- obvious OCR errors: `KINBET`, `ANNIVEZRIARS`, mixed photo/text artifacts

Structure:

- word regions: 1517, while best result had 360 boxes
- line regions: 53 derived lines
- metadata: present
- boxes useful: useful but need best-result selection/deduplication before line grouping

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: partial

Decision: usable-with-cleanup for fragments; fail for current readable hierarchy.

### corpus2_new_doc_2026_04_28_16_55_48

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920007/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920007/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920007/text.txt`

Text quality:

- raw OCR: recognizable article text
- cleaned OCR: useful, though OCR errors remain
- readable_text: noisy due duplicated boxes
- line breaks: cleaned text keeps a useful structure
- obvious OCR errors: `Kimra`, `sccomplshed`, `sade`

Structure:

- word regions: 621, while best result had 110 boxes
- line regions: 30 derived lines
- metadata: present
- boxes useful: yes after deduplication; current count is inflated

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: partial

Decision: usable-with-cleanup for cleaned text; fail for derived line readability.

### corpus2_new_doc_2026_04_28_20_26_02

Input path: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 20.26.02.jpg`

Artifacts inspected:

- data.json: `data/notebook_outputs/ocr_text_quality_review/doc_920008/data.json`
- page_1.html: `data/notebook_outputs/ocr_text_quality_review/doc_920008/page_1.html`
- text.txt: `data/notebook_outputs/ocr_text_quality_review/doc_920008/text.txt`

Text quality:

- raw OCR: mostly noise
- cleaned OCR: still mostly noise
- readable_text: not useful
- line breaks: present but not meaningful
- obvious OCR errors: most output is non-semantic text

Structure:

- word regions: 155, while best result had 28 boxes
- line regions: 20 derived lines
- metadata: present
- boxes useful: poor for readable text; may still help OCR-aware geometry

Japanese/macron/punctuation:

- Japanese preserved: not observed
- macrons preserved: not observed
- punctuation preserved: not meaningful

Decision: fail.

## Cross-Page Findings

- The full runtime OCR path completed for all 8 pages.
- `data.json`, `page_1.html`, and `text.txt` were produced for every page.
- Tesseract word boxes reached `PageResult.text_regions`.
- Derived line regions were created for every page.
- `PageResult.metadata["readable_text"]`, `ocr_word_count`, `ocr_line_count`,
  and `ocr_text_boxes` were present.
- The cleanup-chain line-break fix held on real outputs: cleaned text did not
  flatten into a single line.
- Real Japanese/macron preservation could not be evaluated from this run because
  the active OCR language config was `eng`, and no sampled OCR output contained
  Japanese characters or macronized terms.

## Failure Patterns

### Duplicate OCR Boxes from Multi-PSM Results

The dominant failure is duplicate OCR geometry.

The adapter currently sees boxes from:

```text
best_ocr_result
ocr_results[psm=11]
ocr_results[psm=3]
ocr_results[psm=6]
```

Because the best result is also present in `ocr_results`, and because all PSM
candidate boxes are promoted together, word counts are inflated. This directly
causes repeated words in derived line regions and makes `readable_text` much
less useful than cleaned text.

### Line Grouping Uses Too Much Geometry

The current line grouping works on synthetic fixtures but is not yet robust on
real pages with:

- duplicate word boxes
- overlapping boxes from different PSMs
- figure labels
- captions near illustrations
- photo/article layouts
- tiny artifact boxes

### OCR Quality Varies Sharply

Cleaned text is useful on several pages, especially:

- `original_img_3337`
- `original_img_3288`
- `original_img_3330`
- `corpus2_new_doc_2026_04_28_16_55_48`

But OCR is poor on:

- `corpus2_new_doc_2026_04_28_16_56_38`
- `corpus2_new_doc_2026_04_28_20_26_02`

### Japanese/Macrons Not Exercised by Active OCR Config

Synthetic tests protect Japanese/macron preservation once text exists, but this
real run did not produce Japanese/macron text. The active Tesseract language
configuration is English-only.

## Recommended Next Implementation Pass

Recommended next pass:

- [ ] improve OCR engine settings / PSM handling
- [x] improve line grouping / reading order
- [x] improve DocumentResult/data.json compact serialization
- [ ] promote Japanese analysis into canonical fields
- [ ] improve page reconstruction
- [ ] add annotation/review workflow for OCR text corrections

Specific next implementation:

```text
Fix OCR box selection before line grouping.
```

Proposed scope:

- Prefer `best_ocr_result.bounding_boxes` for canonical word regions.
- Do not aggregate all PSM candidates into `PageResult.text_regions` by default.
- Preserve alternate PSM box summaries only as diagnostic metadata if needed.
- Add tests proving `best_ocr_result` is not duplicated when it also appears in
  `ocr_results`.
- Re-run this same 8-page review and compare `readable_text` quality.

Do not promote Japanese analysis yet. The current blocker is that readable line
regions are polluted by duplicate OCR boxes, not that Japanese metadata lacks
first-class model fields.
