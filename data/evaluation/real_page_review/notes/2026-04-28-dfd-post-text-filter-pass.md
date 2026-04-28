# Donn Draeger DFD Notes — Post Text-Filter Extraction Review

Date: 2026-04-28  
Manifest: data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json  
Notebook: notebooks/05_real_page_extraction_review.ipynb  
Review output: data/notebook_outputs/real_page_review/post_text_filter/

## Scope

This second pass reviews image-region detection after the text-like rejection changes. It compares the same 9 pages from the first pass and adds a second varied batch of 10 pages.

No runtime integration was done. No thresholds were changed during this review.

Legend:

- Accepted regions are final image/diagram candidates.
- Rejected candidates are detector candidates removed by the text-like filter.
- The primary question is whether typewritten text/title/vertical fragments are rejected without losing real diagrams.

## Same 9 Pages From First Pass

### original_img_3288
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(666, 931, 1007, 1038)` — `title_text_like`
  - `(397, 917, 596, 1006)` — `title_text_like`
- Real diagrams/images retained: n/a; none expected.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: improved; first-pass title false positives are fixed.

### original_img_3292
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3292.jpg

- Accepted image regions: 1
  - `(189, 644, 738, 974)` — body-outline diagram
- Rejected candidates:
  - `(941, 78, 1001, 405)` — `text_like_components`
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful diagram crop.
- Integration signal: improved; this is a good representative success case.

### original_img_3312
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(91, 439, 191, 756)` — `text_like_components`
  - `(745, 325, 833, 681)` — `text_like_components`
- Real diagrams/images retained: no clear accepted figure; page has visible figure/photo content on the right edge that was not captured.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: not observed in candidates.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: false positives fixed, but figure/photo recall is still weak.

### original_img_3327
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3327.jpg

- Accepted image regions: 1
  - `(593, 937, 1145, 1251)` — bridge/structure diagram
- Rejected candidates:
  - `(682, 1365, 916, 1478)` — `text_like_components`
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful diagram crop.
- Integration signal: improved; another good success case.

### original_img_3331
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3331.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(554, 176, 860, 306)` — `text_like_components`
  - `(35, 1360, 292, 1474)` — `text_like_components`
- Real diagrams/images retained: n/a; no obvious standalone diagram.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: improved; first-pass false positives are fixed.

### original_img_3340
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3340.jpg

- Accepted image regions: 1
  - `(708, 286, 1046, 511)` — hand/gesture diagram
- Rejected candidates:
  - `(652, 1340, 830, 1501)` — `text_like_components`
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful diagram crop.
- Integration signal: improved; text false positive fixed while real diagram remains.

### original_img_3378
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3378.jpg

- Accepted image regions: 0
- Rejected candidates: 0
- Real diagrams/images retained: n/a; sparse/mostly blank page.
- Text/title/vertical fragments rejected: n/a.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: still good for sparse page class.

### original_img_3380
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3380.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(512, 1298, 1043, 1484)` — `text_like_components`
- Real diagrams/images retained: n/a; no obvious diagram.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: improved; upside-down text false positive is fixed.

### original_img_3389
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3389.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(31, 381, 145, 685)` — `text_like_components`
  - `(251, 581, 388, 762)` — `text_like_components`
- Real diagrams/images retained: n/a; no clear diagram in reviewed content.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: improved; vertical text false positives are fixed.

## New Varied Batch

### original_img_3335
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg

- Accepted image regions: 0
- Rejected candidates: 0
- Real diagrams/images retained: no; a large illustration/photo-like figure is visible but was not detected.
- Text/title/vertical fragments rejected: n/a.
- Over-rejection: none; no candidate was produced.
- Under-rejection: none.
- Crop quality: no accepted crops.
- Integration signal: risky; large illustration recall is weak.

### original_img_3337
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3337.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(64, 90, 241, 288)` — `text_like_components`
- Real diagrams/images retained: n/a; no obvious standalone figure in reviewed view.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: acceptable for text-heavy page.

### original_img_3344
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(873, 114, 1190, 445)` — `text_like_components`
  - `(62, 662, 205, 880)` — `text_like_components`
- Real diagrams/images retained: no; a labeled diagram/figure area appears to have been rejected.
- Text/title/vertical fragments rejected: yes for the text fragment.
- Over-rejection: yes; labeled diagram-like content was rejected as text-like.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: risky; labeled diagrams can be over-filtered.

### original_img_3345
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3345.jpg

- Accepted image regions: 1
  - `(196, 773, 454, 957)` — small weapon/object diagram
- Rejected candidates:
  - `(33, 69, 159, 485)` — `text_like_components`
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful but partial; this page may include more small images than the single accepted crop.
- Integration signal: partially good; small diagram retention works here.

### original_img_3352
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3352.jpg

- Accepted image regions: 2
  - `(517, 334, 1200, 712)` — hand-drawn/labeled figure area
  - `(224, 480, 511, 667)` — object/detail crop
- Rejected candidates: 0
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: n/a.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful, though the larger crop includes some labels/nearby marks.
- Integration signal: good for hand-drawn/labeled figure class.

### original_img_3353
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3353.jpg

- Accepted image regions: 1
  - `(543, 412, 1176, 796)` — large hand-drawn/labeled figure area
- Rejected candidates:
  - `(485, 125, 763, 201)` — `text_like_components`
- Real diagrams/images retained: yes.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: useful but includes labels/caption-like annotations.
- Integration signal: good, with future caption/label association needed.

### original_img_3356
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3356.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(195, 1234, 428, 1438)` — `text_like_components`
  - `(33, 1067, 287, 1153)` — `text_like_components`
- Real diagrams/images retained: n/a; appears text-heavy.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: acceptable for text-heavy page.

### original_img_3386
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3386.jpg

- Accepted image regions: 0
- Rejected candidates: 0
- Real diagrams/images retained: n/a; text-heavy page.
- Text/title/vertical fragments rejected: n/a; no candidates.
- Over-rejection: none observed.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: acceptable for text-heavy page.

### original_img_3390
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3390.jpg

- Accepted image regions: 0
- Rejected candidates:
  - `(242, 90, 338, 248)` — `text_like_components`
- Real diagrams/images retained: n/a; visible candidate appears text-like.
- Text/title/vertical fragments rejected: yes.
- Over-rejection: none obvious from reviewed view.
- Under-rejection: none observed.
- Crop quality: no accepted crops.
- Integration signal: acceptable, but needs another pass if page contains small symbols outside candidate areas.

### original_img_3397
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3397.jpg

- Accepted image regions: 0
- Rejected candidates: 0
- Real diagrams/images retained: no; page contains small symbols, arrows, and hand-drawn shapes, but no candidates were produced.
- Text/title/vertical fragments rejected: n/a.
- Over-rejection: none; no candidate was produced.
- Under-rejection: none.
- Crop quality: no accepted crops.
- Integration signal: risky; small symbols/arrows and sparse diagram clusters are missed.

## Same-9-Page Comparison Outcome

The same-9 comparison is substantially improved:

- First-pass title/text false positives on `3288`, `3331`, `3380`, and `3389` are now rejected.
- The real diagrams retained in the first comparison remain retained on `3292`, `3327`, and `3340`.
- `3378` remains clean with no candidates.
- The reviewed false-positive problem is mostly fixed for these 9 pages.

## New-Batch Outcome

The second batch shows that false positives are much lower, but recall is still uneven:

- Good retained diagrams: `3345`, `3352`, `3353`.
- Good text-heavy no-region pages: `3337`, `3356`, `3386`, likely `3390`.
- Missed large illustration/photo: `3335`.
- Missed small symbols/arrows/diagram clusters: `3397`.
- Over-rejected labeled diagram: `3344`.

## False Positives Remaining

No obvious accepted typewritten text false positives were observed in this 19-page pass. The text-like filter is doing its intended job on titles, body fragments, and vertical/rotated text columns.

## False Negatives / Over-Rejections

- `3335`: large illustration/photo-like figure not detected.
- `3397`: small symbols/arrows and sparse hand-drawn shapes not detected.
- `3344`: labeled diagram/figure area rejected as `text_like_components`, likely because labels and diagram strokes look too text-like.
- Some retained diagram crops on `3352` and `3353` include labels/annotation areas; this is acceptable for review but will need caption/label handling later.

## Runtime Integration Decision

Image extraction is improved but still premature for runtime integration.

The current detector is now much safer against text false positives, but it is not yet reliable enough across the corpus because it misses large illustrations and sparse symbol/arrow pages, and can over-reject labeled diagrams.

Recommended status:

- [ ] Safe to integrate now
- [x] Tune recall / detector coverage first
- [x] Review more diagram-heavy pages
- [ ] Improve text cleanup first
- [ ] Wire into `DocumentResult`

## Recommended Next Coding Pass

Add a second detector-quality pass focused on recall, not text rejection:

- improve large illustration/photo-like region detection without reintroducing text false positives
- add support for sparse symbol/arrow/hand-drawn shape clusters
- add a way to preserve labeled diagrams where labels are embedded inside real figure regions
- consider a pluggable detector strategy interface if the current contour detector cannot balance line drawings, photos, symbols, and text rejection cleanly

Do not integrate into `WorkflowOrchestrator` until the detector handles large illustrations and sparse diagram clusters more consistently.
