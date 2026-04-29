# Cross-Corpus Image Region Generalization Pass

Date: 2026-04-28  
DFD corpus: `data/corpora/donn_draeger/dfd_notes_master/original/`  
Corpus 2 manifest: `data/corpora/ad_hoc/corpus2/manifests/manifest.local.json`  
Temporary review output: `data/notebook_outputs/cross_corpus_generalization_pass/`

## Scope

This pass validates the new scoring-based review-mode detector across both DFD and Corpus 2.

The goal was one stronger detector configuration for similar scanned martial-arts/research pages, not per-corpus presets.

No runtime defaults were changed. Image extraction remains disabled by default.

## Detector Change Summary

The filter now computes and records candidate scores:

- `text_like_score`
- `figure_like_score`
- `photo_like_score`
- `sparse_symbol_score`
- `crop_quality_score`

Heuristic `figure` candidates are no longer blindly exempt from text filtering. Candidates are accepted only when visual evidence is strong enough or text-like evidence is weak enough.

## DFD Regression Check

| Sample | Accepted | Rejected | Outcome |
| --- | ---: | ---: | --- |
| `dfd_original_img_3335` | 1 | 1 | Large figure retained; text block rejected. |
| `dfd_original_img_3344` | 1 | 3 | Labeled diagram retained; text-like candidates rejected. |
| `dfd_original_img_3397` | 2 | 0 | Sparse symbols/arrows retained. |
| `dfd_original_img_3352` | 2 | 3 | Useful candidates retained; broad/right-side label-heavy crop reduced versus earlier passes. |
| `dfd_original_img_3330` | 2 | 3 | Tall/narrow visual strips retained more than before, but this remains an ambiguous case. |

DFD known-good cases stayed broadly good. The main behavior change is stricter rejection of text-like and broad mixed crops.

## Corpus 2 Check

| Sample | Accepted | Rejected | Outcome |
| --- | ---: | ---: | --- |
| `corpus2_new_doc_2026_04_28_16_59_35` | 0 | 4 | Prior text-column false positives rejected. |
| `corpus2_new_doc_2026_04_28_17_03_18` | 0 | 1 | Prior body-text crop rejected. |
| `corpus2_new_doc_2026_04_28_17_10_58` | 1 | 2 | Photo/text area retained; heading/text fragments rejected. |
| `corpus2_new_doc_2026_04_28_18_29_28` | 1 | 1 | Photo/article region retained, but crop is still broad and includes text. |
| `corpus2_new_doc_2026_04_28_18_54_00` | 1 | 0 | Photo-grid recall improved from none to one accepted panel/region, but still partial. |
| `corpus2_new_doc_2026_04_28_19_43_28` | 0 | 2 | Prior manuscript/text crop rejected. |
| `corpus2_new_doc_2026_04_28_20_05_24` | 0 | 3 | Prior text paragraph crops rejected. |
| `corpus2_new_doc_2026_04_28_20_17_15` | 0 | 4 | Prior broad text/manuscript crop rejected. |
| `corpus2_unknown_3` | 0 | 3 | Prior broad text/manuscript crop rejected. |
| `corpus2_unknown` | 1 | 1 | Bottom cartoon/visual crop retained; large text block rejected. |

## Improvements

- Corpus 2 plain-text false positives improved substantially.
- DFD known-good figure/diagram/symbol cases stayed intact.
- Corpus 2 photo-grid recall improved, though not completely.
- Rejection metadata is more useful: rejected candidates include scores and feature diagnostics.
- Accepted candidates include reason and scoring metadata for notebook/runtime review.

## Remaining Problems

- Corpus 2 photo grids are still under-detected.
- Some accepted crops are still broad or mixed text/image, especially `18_29_28`.
- Tall/narrow manuscript or visual strips remain ambiguous.
- The detector is better but still review-mode only.

## Integration Decision

Recommended status:

- [ ] Safe to enable by default
- [x] Keep disabled by default
- [x] Safe to continue broader review-mode testing
- [ ] Add per-corpus presets now
- [ ] Make YOLO the default

One detector configuration now appears more acceptable across both similar corpora than the previous DFD-tuned hard rules. It is not production-authoritative and should remain explicitly gated.

Recommended next step: run a broader review-mode validation on more Corpus 2 pages. If the same misses remain, improve photo-grid candidate generation rather than adding corpus-specific settings.
