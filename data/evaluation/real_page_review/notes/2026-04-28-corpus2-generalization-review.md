# Corpus 2 — Image Extraction Generalization Review

Date: 2026-04-28  
Manifest: `data/corpora/ad_hoc/corpus2/manifests/manifest.local.json`  
Corpus: `data/corpora/ad_hoc/corpus2/original/`  
Temporary review output: `data/notebook_outputs/corpus2_generalization_review/`

## Scope

This pass checks whether the DFD-tuned image-region detector generalizes to the second local corpus.

The review used a deterministic 18-page sample from the 138-image manifest. It ran the same classical image-region detector used by review-mode `ExtractionService` and saved temporary overlays/crops under ignored notebook output.

No detector thresholds, runtime behavior, OCR logic, or corpus files were changed.

## Pages Reviewed

| Index | Sample ID | Accepted | Rejected | Consolidation | Status | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | `corpus2_image12342` | 3 | 0 | 1 | usable-with-noise | Real illustrated regions detected, but a small text/heading block was also accepted. |
| 1 | `corpus2_new_doc_2026_04_28_16_55_48` | 2 | 0 | 4 | usable-with-noise | Photo regions retained, but a large mixed text/image crop and a text-heavy block were accepted. |
| 2 | `corpus2_new_doc_2026_04_28_16_56_38` | 0 | 1 | 0 | usable | Text-heavy page produced no accepted crops; rejected candidate was classified as `sparse_text_band`. |
| 3 | `corpus2_new_doc_2026_04_28_16_59_35` | 1 | 1 | 2 | problematic | Accepted crop is mostly text columns; likely plain-text false positive. |
| 4 | `corpus2_new_doc_2026_04_28_17_03_18` | 1 | 0 | 0 | problematic | Accepted crop is a text block, not a diagram/photo. |
| 5 | `corpus2_new_doc_2026_04_28_17_10_58` | 2 | 0 | 0 | problematic | Accepted heading/paragraph fragments while the page contains photographic material; old text false-positive pattern returns. |
| 6 | `corpus2_new_doc_2026_04_28_17_19_36` | 0 | 0 | 0 | usable | No accepted crops on a page with text and a small photo; possible missed photo region but no false positives. |
| 7 | `corpus2_new_doc_2026_04_28_18_16_34` | 0 | 0 | 0 | usable | Dense question/answer text page produced no accepted crops. |
| 8 | `corpus2_new_doc_2026_04_28_18_29_28` | 1 | 0 | 0 | usable | Photo/figure area retained; crop appears useful. |
| 9 | `corpus2_new_doc_2026_04_28_18_40_35` | 0 | 0 | 0 | usable-with-noise | Page appears to contain photo content, but no accepted crops; likely false negative. |
| 10 | `corpus2_new_doc_2026_04_28_18_54_00` | 0 | 0 | 0 | usable-with-noise | Photo grid page produced no accepted crops; likely false negative. |
| 11 | `corpus2_new_doc_2026_04_28_19_43_28` | 1 | 1 | 0 | problematic | Large text/manuscript block accepted; text-like density rejected only the top band. |
| 12 | `corpus2_new_doc_2026_04_28_19_55_36` | 0 | 1 | 0 | usable-with-noise | Text-like top band rejected; vertical manuscript area was not extracted. |
| 13 | `corpus2_new_doc_2026_04_28_20_05_24` | 2 | 1 | 0 | problematic | Accepted crops are text paragraph areas; top band rejected correctly. |
| 14 | `corpus2_new_doc_2026_04_28_20_17_15` | 1 | 1 | 2 | problematic | Large mixed text/manuscript crop accepted; top band rejected. |
| 15 | `corpus2_new_doc_2026_04_28_20_26_02` | 2 | 0 | 0 | usable-with-noise | Tall visual/manuscript strip captured, but crop grouping is broad and overlaps page text. |
| 16 | `corpus2_unknown_3` | 1 | 1 | 0 | problematic | Large text/manuscript area accepted; top band rejected. |
| 17 | `corpus2_unknown` | 1 | 0 | 0 | problematic | Large plain-text list accepted while bottom cartoon/illustration appears missed. |

## Summary

Pages reviewed: 18

High-level outcome:

- Corpus 2 does not behave like the DFD notes.
- Plain-text false positives return on multiple pages.
- The detector is useful on some photo/illustration pages, but not reliable enough for broad review-mode corpus processing without presets or strategy changes.
- The current DFD-tuned settings should not be generalized silently to Corpus 2.

## Plain-Text False Positives

Plain-text or mostly-text accepted crops were observed on:

- `corpus2_new_doc_2026_04_28_16_59_35`
- `corpus2_new_doc_2026_04_28_17_03_18`
- `corpus2_new_doc_2026_04_28_17_10_58`
- `corpus2_new_doc_2026_04_28_19_43_28`
- `corpus2_new_doc_2026_04_28_20_05_24`
- `corpus2_new_doc_2026_04_28_20_17_15`
- `corpus2_unknown_3`
- `corpus2_unknown`

This is the main regression compared with the later DFD reviews, where common accepted plain-text false positives had mostly disappeared.

## Diagrams / Photos Retained

Useful visual crops were observed on:

- `corpus2_image12342`
- `corpus2_new_doc_2026_04_28_16_55_48`
- `corpus2_new_doc_2026_04_28_18_29_28`
- `corpus2_new_doc_2026_04_28_20_26_02`

However, several useful crops are mixed with accepted text regions or broad page sections.

## Missed Visual Regions

Likely missed visual regions were observed on:

- `corpus2_new_doc_2026_04_28_17_19_36`
- `corpus2_new_doc_2026_04_28_18_40_35`
- `corpus2_new_doc_2026_04_28_18_54_00`
- `corpus2_unknown`

The photo grid on `18_54_00` is a notable false negative: no accepted crops were returned despite visible photo panels.

## Broad / Label-Heavy Crops

Broad or mixed text/image crops remain common:

- `corpus2_new_doc_2026_04_28_16_55_48`
- `corpus2_new_doc_2026_04_28_19_43_28`
- `corpus2_new_doc_2026_04_28_20_17_15`
- `corpus2_new_doc_2026_04_28_20_26_02`

This is similar to the DFD broad-crop issue, but Corpus 2 makes the text-vs-image ambiguity worse.

## Tall / Narrow Visual Regions

Tall/narrow content remains fragile. `corpus2_new_doc_2026_04_28_20_26_02` captured a useful tall strip, but other vertical manuscript or narrow page regions were either missed or accepted as broad mixed crops.

## Detector Preset Implications

Corpus 2 likely needs different detector behavior than the DFD notes:

- Stronger rejection for large plain-text blocks and headings.
- Better separation of photos/figures from adjacent article text.
- A strategy for photo-grid pages where current contour logic misses panels.
- Separate handling for manuscript/scroll images versus body text.

This points toward detector presets or pluggable strategy options rather than another one-size-fits-all threshold tweak.

## Runtime Integration Recommendation

Recommended status:

- [ ] Safe to integrate by default
- [ ] Safe to integrate broadly in review mode
- [x] Keep gated review-mode integration, but treat DFD and Corpus 2 separately
- [x] Add detector presets / strategy options before broader Corpus 2 runs
- [ ] Tune global thresholds immediately

The existing review-mode `ExtractionService` should remain available and disabled by default. It is useful for DFD-style pages, but Corpus 2 shows that current default detector settings do not generalize cleanly.

Recommended next coding pass: add a detector preset/options layer so review runs can choose between DFD-style scanned notes, photo/article pages, manuscript/scroll pages, and conservative text-heavy pages without changing global defaults.
