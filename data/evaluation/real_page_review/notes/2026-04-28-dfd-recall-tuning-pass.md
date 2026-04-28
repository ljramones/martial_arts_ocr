# Donn Draeger DFD Notes — Recall Tuning Review

Date: 2026-04-28  
Manifest: data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json  
Notebook: notebooks/05_real_page_extraction_review.ipynb  
Temporary visual review sheet: /tmp/dfd_recall_review/contact_sheet.jpg

## Scope

This pass reviews the same 19 pages after the recall tuning changes. Runtime integration was not done, thresholds were not changed during review, and generated crops/contact sheets were kept outside the repo.

Green boxes are accepted image/diagram candidates. Red boxes are rejected candidates with text-like rejection reasons.

## Pages Reviewed

### original_img_3288
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(666, 931, 1007, 1038)` — `title_text_like`
  - `(397, 917, 596, 1006)` — `title_text_like`
- Crop usefulness: no accepted crops.
- Notes: old title-text false positives remain rejected.
- Runtime integration signal: good for this text/title page class.

### original_img_3292
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3292.jpg

- Accepted candidates:
  - `(141, 21, 720, 505)` — useful hand/outline diagram
  - `(189, 644, 738, 974)` — useful body-outline diagram
- Rejected candidates:
  - `(773, 719, 856, 934)` — `text_like_components`
  - `(896, 12, 1532, 1033)` — `text_like_components`
  - `(762, 26, 872, 343)` — `text_like_components`
- Extra candidates: useful; page appears to contain two real diagram areas.
- Crop boundaries: acceptable.
- Runtime integration signal: good.

### original_img_3312
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg

- Accepted candidates:
  - `(889, 763, 1315, 1200)` — figure/photo-like region
  - `(855, 31, 1306, 390)` — figure/photo-like region
  - `(853, 397, 1308, 737)` — figure/photo-like region
- Rejected candidates:
  - `(75, 86, 832, 1109)` — `text_like_components`
- Extra candidates: useful; previous miss on right-edge figure/photo content is substantially improved.
- Crop boundaries: acceptable, though figures are split into separate stacked regions.
- Runtime integration signal: improved, but needs review for figure grouping.

### original_img_3327
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3327.jpg

- Accepted candidates:
  - `(119, 641, 1004, 917)` — large scene/bridge diagram
  - `(593, 937, 1145, 1251)` — bridge/structure diagram
  - `(84, 956, 556, 1262)` — overlapping/adjacent bridge crop
- Rejected candidates:
  - `(77, 270, 1121, 628)` — `text_like_components`
  - `(77, 1364, 689, 1504)` — `text_like_components`
- Extra candidates: partly useful but overlapping; NMS/grouping may need refinement.
- Crop boundaries: generally acceptable; duplicated structure crops should be consolidated later.
- Runtime integration signal: useful but noisy.

### original_img_3331
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3331.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(36, 27, 1111, 1503)` — `text_like_components`
- Notes: old body-text false positives remain rejected.
- Runtime integration signal: good for text-heavy page class.

### original_img_3340
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3340.jpg

- Accepted candidates:
  - `(708, 286, 1046, 511)` — hand/gesture diagram
- Rejected candidates:
  - `(652, 1340, 830, 1501)` — `text_like_components`
  - `(62, 1320, 335, 1494)` — `text_like_components`
- Crop usefulness: useful.
- Notes: known real diagram retained; old text false positive stays rejected.
- Runtime integration signal: good.

### original_img_3378
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3378.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(480, 1043, 935, 1164)` — `sparse_text_band`
- Notes: sparse text-like band is now rejected.
- Runtime integration signal: good.

### original_img_3380
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3380.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(512, 1298, 1043, 1484)` — `text_like_components`
- Notes: upside-down text false positive remains rejected.
- Runtime integration signal: good.

### original_img_3389
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3389.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(31, 21, 1120, 1117)` — `text_like_components`
- Notes: vertical/rotated text remains rejected.
- Runtime integration signal: good.

### original_img_3335
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg

- Accepted candidates:
  - `(182, 469, 1050, 1439)` — large Fudo/illustration region
- Rejected candidates:
  - `(104, 74, 1138, 436)` — `text_like_components`
- Fixed recall case: yes; large illustration is now recovered.
- Crop boundaries: useful, includes the main figure cleanly.
- Runtime integration signal: good.

### original_img_3337
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(50, 67, 1051, 685)` — `text_like_components`
- Notes: large text region is rejected.
- Runtime integration signal: good for text-heavy page class.

### original_img_3344
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg

- Accepted candidates:
  - `(873, 114, 1190, 445)` — labeled diagram
- Rejected candidates:
  - `(94, 888, 214, 1061)` — `text_like_components`
  - `(62, 662, 205, 880)` — `text_like_components`
  - `(62, 97, 261, 243)` — `text_like_components`
- Fixed recall case: yes; labeled diagram is now retained.
- Crop boundaries: useful.
- Runtime integration signal: good.

### original_img_3345
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3345.jpg

- Accepted candidates:
  - `(196, 773, 454, 957)` — small diagram
  - `(199, 517, 381, 748)` — small diagram
  - `(188, 245, 367, 435)` — small diagram
- Rejected candidates:
  - `(681, 154, 1133, 1103)` — `text_like_components`
  - `(33, 69, 168, 931)` — `text_like_components`
- Extra candidates: useful; the page contains multiple small diagrams.
- Crop boundaries: acceptable.
- Runtime integration signal: good.

### original_img_3352
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3352.jpg

- Accepted candidates:
  - `(517, 334, 1200, 712)` — labeled hand-drawn figure
  - `(101, 1377, 1095, 1519)` — bottom figure/caption band; useful but label-heavy
  - `(16, 475, 511, 667)` — smaller object/figure crop
  - `(164, 1036, 284, 1372)` — narrow figure-like crop
- Rejected candidates:
  - `(100, 721, 1109, 855)` — `text_like_components`
- Extra candidates: mostly useful, but label/caption-heavy.
- Crop boundaries: acceptable for review; future caption/label association needed.
- Runtime integration signal: useful but noisy.

### original_img_3353
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3353.jpg

- Accepted candidates:
  - `(543, 412, 1176, 796)` — large hand-drawn/labeled figure area
- Rejected candidates:
  - `(122, 806, 1073, 1513)` — `text_like_components`
  - `(126, 116, 763, 202)` — `text_like_components`
- Crop usefulness: useful and better focused than broad text-heavy candidates.
- Runtime integration signal: good.

### original_img_3356
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3356.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(37, 354, 1084, 953)` — `text_like_components`
  - `(74, 34, 1127, 346)` — `text_like_components`
  - `(718, 953, 1163, 1122)` — `text_like_components`
  - `(374, 1268, 772, 1432)` — `text_like_components`
- Notes: text-heavy page; no accepted image false positives.
- Runtime integration signal: good.

### original_img_3386
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3386.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(924, 784, 1134, 1122)` — `text_like_components`
- Notes: likely text fragment rejected.
- Runtime integration signal: good.

### original_img_3390
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3390.jpg

- Accepted candidates: 0
- Rejected candidates:
  - `(89, 90, 338, 777)` — `text_like_components`
- Notes: vertical text-like block rejected.
- Runtime integration signal: good.

### original_img_3397
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3397.jpg

- Accepted candidates:
  - `(24, 864, 575, 1582)` — sparse hand-drawn cluster
  - `(636, 1199, 887, 1445)` — small sparse symbol/arrow crop
- Rejected candidates: 0
- Fixed recall case: yes; sparse symbols/arrows are now detected.
- Crop boundaries: useful; one accepted crop is broad but captures the intended cluster.
- Runtime integration signal: good for this page, but sparse-symbol grouping should be watched.

## Fixed Recall Cases

- `3335`: large illustration/photo-like figure is now accepted and the top text block is rejected.
- `3344`: labeled diagram is now accepted while nearby text fragments are rejected.
- `3397`: sparse symbol/arrow clusters are now accepted.
- `3312`: right-edge figure/photo regions are now detected, correcting a previous miss.
- `3345`: multiple small diagrams are now accepted rather than only one.
- `3353`: the useful labeled figure crop remains accepted while broad text-heavy regions are rejected.

## Remaining False Positives

No old title/body/vertical text false positives reappeared in the reviewed pages. The most questionable accepted regions are not plain text; they are label/caption-heavy or overlapping figure areas.

Potentially noisy accepted candidates:

- `3327`: overlapping bridge/structure crops.
- `3352`: bottom band and narrow crop include labels/annotations and may be too broad.
- `3312`: stacked right-edge figure/photo regions are useful, but grouping may need refinement.

## Remaining False Negatives

No major known miss from the previous review remains obviously missed in this 19-page pass. The review did not cover the full 112-page corpus, so recall is not proven globally.

## Duplicate / Overlapping Candidates

The main remaining issue is candidate grouping rather than basic detection:

- `3327` returns multiple overlapping/adjacent structure crops.
- `3312` splits the right-side figure/photo content into three stacked regions.
- `3352` returns multiple useful but label-heavy figure/caption crops.

These are better than missed regions, but they are not yet clean enough for unattended runtime output.

## Crop Quality

Crop quality is generally good for review:

- Boundaries are acceptable for `3292`, `3335`, `3340`, `3344`, `3345`, and `3397`.
- Some crops include labels/captions, especially `3352` and `3353`; this may be acceptable for preservation, but later output should distinguish figure crop from caption text.
- Several crops are broad enough to need grouping/caption-aware postprocessing before runtime integration.

## Runtime Integration Decision

Recommended status:

- [ ] Safe to integrate now
- [ ] Tune thresholds first
- [x] Add candidate grouping / NMS refinement first
- [ ] Improve text cleanup first
- [ ] More manual review needed

Image extraction is much closer, but runtime integration is still premature. The next coding pass should refine grouping/NMS and crop consolidation, not redesign detection and not move to YOLO by default.

## Recommended Next Coding Pass

Add a candidate grouping/NMS refinement pass:

- merge or suppress overlapping diagram crops more intelligently after text filtering
- preserve separate small diagrams when they are clearly distinct
- avoid broad text/caption bands becoming accepted just because they are attached to figures
- add tests for stacked figure groups, multiple small diagrams, and overlapping diagram candidates
- rerun Notebook 05 on these same 19 pages after grouping changes
