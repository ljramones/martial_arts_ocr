# Donn Draeger DFD Notes — Candidate Consolidation Review

Date: 2026-04-28  
Manifest: data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json  
Notebook: notebooks/05_real_page_extraction_review.ipynb  
Temporary visual review sheet: /tmp/dfd_consolidation_review/contact_sheet.jpg

## Scope

This pass reviews post-filter candidate grouping/NMS consolidation. Runtime integration was not done, source data was not moved, and generated review overlays were kept outside the repo.

The main question is whether accepted candidates are less duplicated or fragmented after text-like filtering, without losing the recall fixes from the prior pass.

## Pages Reviewed

### original_img_3312
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg

- Accepted candidates:
  - `(853, 31, 1308, 737)` — merged top/middle right-edge figure/photo area
  - `(889, 763, 1315, 1200)` — lower right-edge figure/photo area
- Rejected candidates:
  - `(75, 86, 832, 1109)` — `text_like_components`
- Consolidation:
  - `adjacent_merge` produced `(853, 31, 1308, 737)`
- Outcome: fragmentation reduced from three right-edge candidates to two.
- Crop quality: useful; still split into upper and lower figure areas.
- Risk: future grouping could merge all right-edge figures, but current split may be acceptable because they are visually separated.

### original_img_3327
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3327.jpg

- Accepted candidates:
  - `(119, 641, 1145, 1251)` — broad bridge/structure region
- Rejected candidates:
  - `(77, 270, 1121, 628)` — `text_like_components`
  - `(77, 1364, 689, 1504)` — `text_like_components`
- Consolidation:
  - `adjacent_merge` produced the broad bridge crop.
  - `contained_suppression` removed an overlapping child crop.
- Outcome: duplicate/overlapping bridge candidates reduced to one.
- Crop quality: improved for preservation; crop is broad but semantically coherent.

### original_img_3352
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3352.jpg

- Accepted candidates:
  - `(16, 334, 1200, 712)` — broad top figure/annotation area
  - `(164, 1036, 284, 1372)` — narrow figure-like crop
  - `(101, 1377, 1095, 1519)` — bottom figure/caption band
- Rejected candidates:
  - `(100, 721, 1109, 855)` — `text_like_components`
- Consolidation:
  - `adjacent_merge` produced `(16, 334, 1200, 712)`
- Outcome: adjacent top candidates were consolidated.
- Crop quality: useful but broad; labels/annotations remain attached.
- Risk: this page still needs caption/label-aware handling before unattended runtime output.

### original_img_3335
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg

- Accepted candidates:
  - `(182, 469, 1050, 1439)` — large Fudo illustration
- Rejected candidates:
  - `(104, 74, 1138, 436)` — `text_like_components`
- Consolidation: none.
- Outcome: prior recall fix preserved.
- Crop quality: good.

### original_img_3344
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg

- Accepted candidates:
  - `(873, 114, 1190, 445)` — labeled diagram
- Rejected candidates:
  - `(94, 888, 214, 1061)` — `text_like_components`
  - `(62, 662, 205, 880)` — `text_like_components`
  - `(62, 97, 261, 243)` — `text_like_components`
- Consolidation: none.
- Outcome: prior labeled-diagram recall fix preserved.
- Crop quality: good.

### original_img_3397
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3397.jpg

- Accepted candidates:
  - `(24, 864, 575, 1582)` — sparse hand-drawn cluster
  - `(636, 1199, 887, 1445)` — smaller sparse symbol/arrow crop
- Rejected candidates: none.
- Consolidation: none.
- Outcome: prior sparse-symbol recall fix preserved.
- Crop quality: useful; no unwanted merge of distinct clusters.

### original_img_3292
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3292.jpg

- Accepted candidates:
  - `(141, 21, 720, 505)` — hand/outline diagram
  - `(189, 644, 738, 974)` — body-outline diagram
- Rejected candidates:
  - `(773, 719, 856, 934)` — `text_like_components`
  - `(896, 12, 1532, 1033)` — `text_like_components`
  - `(762, 26, 872, 343)` — `text_like_components`
- Consolidation: none.
- Outcome: separate diagrams are preserved, text false positives stay rejected.
- Crop quality: good.

### original_img_3340
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3340.jpg

- Accepted candidates:
  - `(708, 286, 1046, 511)` — hand/gesture diagram
- Rejected candidates:
  - `(652, 1340, 830, 1501)` — `text_like_components`
  - `(62, 1320, 335, 1494)` — `text_like_components`
- Consolidation: none.
- Outcome: known good page remains stable.
- Crop quality: good.

## Duplicate / Fragmentation Outcome

- `3327` is substantially improved: overlapping bridge crops are now one broad region.
- `3312` is improved: three right-edge figure/photo crops are reduced to two.
- `3352` is partially improved: adjacent upper figure/annotation candidates are consolidated, but broad label-heavy regions remain.

## Regression Check

- `3335` large illustration remains accepted.
- `3344` labeled diagram remains accepted.
- `3397` sparse clusters remain accepted.
- `3292` and `3340` retain known real diagrams.
- Old text-like false positives in the checked pages remain rejected.

## Runtime Integration Decision

Recommended status:

- [ ] Safe to integrate now
- [ ] Tune thresholds first
- [ ] Add detector strategy abstraction first
- [ ] Improve text cleanup first
- [x] More manual review needed

Runtime integration is more plausible after consolidation, but still not ready by default. The detector now produces useful regions on the reviewed target pages, but crop semantics are still not clean enough for unattended output because captions/labels can be merged into broad crops.

## Recommended Next Step

Run one more review over a larger subset of the 112-page corpus with the consolidated detector. If the same behavior holds, the next coding pass should add an opt-in `ExtractionService` that enriches `DocumentResult` behind a feature flag, keeping image extraction disabled by default until corpus-level review is complete.
