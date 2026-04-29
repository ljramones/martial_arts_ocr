# Real OCR Text Quality Review After Box Selection

Date: 2026-04-28  
Run date: 2026-04-29  
Temporary output: `data/notebook_outputs/ocr_text_quality_review/`

## Purpose

Validate that canonical OCR text regions now use the selected OCR result only,
while alternate Tesseract PSM candidates remain diagnostics.

## Setup

- OCR engine: Tesseract 5.5.2
- language config: `eng`
- PSM candidates: full-page PSM `11`, `3`, and `6`
- selected boxes: `best_ocr_result.bounding_boxes`
- alternate boxes: compact metadata only
- runtime defaults changed: no
- extraction behavior changed: no

## Summary Table

| Corpus | Sample ID | Canonical Words | Lines | Readable Text Result | Alternate Candidates | Alternate Word Boxes | Notes |
|---|---|---:|---:|---|---:|---:|---|
| DFD | `original_img_3337` | 598 | 58 | improved | 3 | 1190 | Repeated PSM words removed; line text is usable with OCR errors. |
| DFD | `original_img_3288` | 56 | 10 | improved | 3 | 126 | Title/list readable text is compact. |
| DFD | `original_img_3344` | 438 | 56 | improved but noisy | 3 | 950 | Repetition fixed; OCR quality remains weak. |
| DFD | `original_img_3330` | 90 | 10 | improved but layout-limited | 3 | 345 | Figure labels/caption remain hard, but duplication is gone. |
| Corpus 2 | `16_56_38` | 31 | 6 | improved but poor OCR | 3 | 218 | Canonical hierarchy is compact; source OCR is weak. |
| Corpus 2 | `18_29_28` | 360 | 48 | improved | 3 | 797 | Repetition fixed; article/photo layout still noisy. |
| Corpus 2 | `16_55_48` | 110 | 22 | improved | 3 | 401 | Readable text is close to cleaned text shape. |
| Corpus 2 | `20_26_02` | 28 | 12 | unchanged/poor OCR | 3 | 99 | Odd layout produces mostly noise, but no duplicate inflation. |

## Before / After

The previous run promoted boxes from `best_ocr_result` plus every PSM candidate.
This inflated canonical word counts and repeated words in line text.

After the fix:

- canonical word counts match the selected/best OCR result
- derived line counts are based only on selected boxes
- `readable_text` no longer repeats each word several times because of alternate PSM boxes
- alternate PSM candidates are preserved as compact diagnostics in
  `PageResult.metadata["ocr_alternative_candidates"]`

Examples:

```text
original_img_3337:
  before: 2386 canonical words, readable_len 13354
  after:   598 canonical words, readable_len 3354

original_img_3288:
  before: 238 canonical words, readable_len 1377
  after:   56 canonical words, readable_len 343

corpus2_new_doc_2026_04_28_16_55_48:
  before: 621 canonical words, readable_len 3252
  after:  110 canonical words, readable_len 623
```

## Metadata Check

Each page now includes:

```text
PageResult.metadata["readable_text"]
PageResult.metadata["ocr_word_count"]
PageResult.metadata["ocr_line_count"]
PageResult.metadata["ocr_text_boxes"]
PageResult.metadata["ocr_alternative_candidates"]
```

Alternate candidate summaries include:

```text
engine
psm
confidence
text_length
word_box_count
selected
```

Large duplicate alternate box arrays are not promoted into canonical
`text_regions`.

## Remaining Issues

- OCR quality is still page-dependent.
- Line grouping is simple and still struggles with captions, figure labels,
  article/photo layouts, and odd vertical/tall-strip pages.
- `original_img_3344`, `original_img_3330`, and Corpus 2 visual/article pages
  still need reading-order and layout-aware text grouping work.
- Real Japanese/macron preservation still needs a language-enabled OCR run; this
  validation used English-only Tesseract output.
- The postprocessor still emits noisy `ls` warnings for words containing the
  substring `ls`, such as `walls` and `accomplshed`; this is logging noise, not
  a text mutation in this pass.

## Decision

Canonical OCR box selection is fixed.

Recommended next implementation pass:

- improve line grouping / reading order using selected boxes only
- keep alternate PSM candidates as diagnostics
- run a language-enabled OCR sample before promoting Japanese analysis

Do not move to Japanese analysis promotion until line grouping and real
language-enabled samples are reviewed.
