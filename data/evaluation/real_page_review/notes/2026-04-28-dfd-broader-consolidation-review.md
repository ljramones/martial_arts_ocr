# Donn Draeger DFD Notes — Broader Consolidation Review

Date: 2026-04-28  
Manifest: data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json  
Notebook: notebooks/05_real_page_extraction_review.ipynb  
Temporary visual review sheet: /tmp/dfd_broader_consolidation_review/contact_sheet.jpg

## Summary

Reviewed 34 pages total: the existing 19-page review set plus 15 additional pages from the Donn Draeger corpus.

Overall result: current image-region extraction is good enough to consider gated runtime integration for review workflows, but not good enough to enable by default. Plain-text false positives remain well controlled, real diagrams/images are usually retained, and consolidation reduces duplicate/fragmented crops. Remaining problems are broad label-heavy crops and missed very tall/narrow visual strips.

Recommended integration status: safe to integrate behind a disabled-by-default option, or enabled only by explicit review/workbench configuration.

## Pages Reviewed

### original_img_3288

- Status: usable
- Accepted regions: none
- Rejected patterns: title text rejected as `title_text_like`.
- Notes: no image regions expected; old false positives remain fixed.

### original_img_3292

- Status: usable
- Accepted regions: `(141, 21, 720, 505)`, `(189, 644, 738, 974)`
- Rejected patterns: vertical/body text rejected as `text_like_components`.
- Notes: two useful diagram crops; no duplicate issue.

### original_img_3312

- Status: usable-with-noise
- Accepted regions: `(853, 31, 1308, 737)`, `(889, 763, 1315, 1200)`
- Rejected patterns: main text column rejected as `text_like_components`.
- Consolidation: one `adjacent_merge`.
- Notes: right-edge figure/photo content is recovered, but split into two stacked regions.

### original_img_3327

- Status: usable-with-noise
- Accepted regions: `(119, 641, 1145, 1251)`
- Rejected patterns: top and bottom text rejected.
- Consolidation: `adjacent_merge`, then `contained_suppression`.
- Notes: duplicate bridge crops reduced to one broad crop. Useful, but broad.

### original_img_3331

- Status: usable
- Accepted regions: none
- Rejected patterns: large body text rejected.
- Notes: no accepted plain-text false positives.

### original_img_3340

- Status: usable
- Accepted regions: `(708, 286, 1046, 511)`
- Rejected patterns: lower body text rejected.
- Notes: known hand/gesture diagram remains stable.

### original_img_3378

- Status: usable
- Accepted regions: none
- Rejected patterns: sparse text band rejected.
- Notes: sparse page remains clean.

### original_img_3380

- Status: usable
- Accepted regions: none
- Rejected patterns: upside-down text rejected.
- Notes: no accepted text false positive.

### original_img_3389

- Status: usable
- Accepted regions: none
- Rejected patterns: vertical/rotated text rejected.
- Notes: no accepted text false positive.

### original_img_3335

- Status: usable
- Accepted regions: `(182, 469, 1050, 1439)`
- Rejected patterns: top body text rejected.
- Notes: large figure recall fix remains good.

### original_img_3337

- Status: usable
- Accepted regions: none
- Rejected patterns: large text block rejected.
- Notes: text-heavy page handled cleanly.

### original_img_3344

- Status: usable
- Accepted regions: `(873, 114, 1190, 445)`
- Rejected patterns: nearby text fragments rejected.
- Notes: labeled diagram recall fix remains good.

### original_img_3345

- Status: usable
- Accepted regions: `(188, 245, 367, 435)`, `(199, 517, 381, 748)`, `(196, 773, 454, 957)`
- Rejected patterns: vertical text and larger body text rejected.
- Notes: multiple small diagrams preserved separately; acceptable.

### original_img_3352

- Status: usable-with-noise
- Accepted regions: `(16, 334, 1200, 712)`, `(164, 1036, 284, 1372)`, `(101, 1377, 1095, 1519)`
- Rejected patterns: middle body text rejected.
- Consolidation: one `adjacent_merge`.
- Notes: useful figure regions, but crops include labels/caption areas and remain broad.

### original_img_3353

- Status: usable
- Accepted regions: `(543, 412, 1176, 796)`
- Rejected patterns: text-heavy bands rejected.
- Notes: useful labeled figure crop retained.

### original_img_3356

- Status: usable
- Accepted regions: none
- Rejected patterns: several body text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3386

- Status: usable
- Accepted regions: none
- Rejected patterns: one text fragment rejected.
- Notes: text-heavy page handled cleanly.

### original_img_3390

- Status: usable
- Accepted regions: none
- Rejected patterns: vertical text-like block rejected.
- Notes: no accepted text false positive.

### original_img_3397

- Status: usable
- Accepted regions: `(24, 864, 575, 1582)`, `(636, 1199, 887, 1445)`
- Rejected patterns: none.
- Notes: sparse symbols/arrows remain detected; separate clusters are not over-merged.

### original_img_3289

- Status: usable
- Accepted regions: none
- Rejected patterns: title/body text rejected.
- Notes: title-page variant handled cleanly.

### original_img_3295

- Status: usable
- Accepted regions: none
- Rejected patterns: two large text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3300

- Status: usable
- Accepted regions: none
- Rejected patterns: two large text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3305

- Status: usable
- Accepted regions: none
- Rejected patterns: small text fragment rejected.
- Notes: no obvious diagram missed in reviewed view.

### original_img_3318

- Status: usable
- Accepted regions: none
- Rejected patterns: vertical/body text rejected.
- Notes: no accepted text false positive.

### original_img_3324

- Status: usable
- Accepted regions: none
- Rejected patterns: several vertical/body text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3330

- Status: problematic
- Accepted regions: `(664, 176, 861, 1083)`
- Rejected patterns: several tall narrow candidates rejected as `bad_aspect_ratio` or `rotated_text_like`.
- Notes: page contains several tall visual strips; only one was accepted. This is a remaining recall problem for very tall/narrow figure panels.

### original_img_3338

- Status: usable
- Accepted regions: none
- Rejected patterns: text fragments rejected.
- Notes: no accepted text false positive.

### original_img_3342

- Status: usable
- Accepted regions: none
- Rejected patterns: large text block rejected.
- Notes: text-heavy rotated page handled cleanly.

### original_img_3348

- Status: usable
- Accepted regions: none
- Rejected patterns: large text block rejected.
- Notes: no accepted text false positive.

### original_img_3358

- Status: usable
- Accepted regions: none
- Rejected patterns: top and lower text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3365

- Status: usable
- Accepted regions: none
- Rejected patterns: two large rotated text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3370

- Status: usable
- Accepted regions: none
- Rejected patterns: two large rotated text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3384

- Status: usable
- Accepted regions: none
- Rejected patterns: multiple text blocks rejected.
- Notes: no accepted text false positive.

### original_img_3394

- Status: usable-with-noise
- Accepted regions: `(391, 871, 819, 1070)`
- Rejected patterns: body text and caption/text bands rejected.
- Notes: small central diagram accepted; crop is useful, but surrounding text/caption layout remains complex.

## Usable Pages

Most reviewed pages are usable for candidate image-region extraction:

- Text-heavy pages generally produce no accepted image regions and reject text candidates correctly.
- Known diagram pages keep useful regions.
- Multiple small diagrams can remain separate when appropriate, as on `3345`.

## Usable with Noise

Pages with useful output but remaining noise or broad crops:

- `3312`: useful right-edge figures, still split into two regions.
- `3327`: bridge content consolidated into one broad crop.
- `3352`: useful figure crops, but label/caption-heavy.
- `3394`: small diagram accepted; surrounding text/caption complexity remains.

## Problematic Pages

- `3330`: very tall/narrow visual strips are partially missed because most candidates are rejected as bad aspect ratio or rotated text.

This does not invalidate gated integration, but it argues against enabling the detector by default.

## Failure Patterns

- Very tall/narrow figure panels are still hard to distinguish from vertical text.
- Label/caption-heavy crops remain broad because the detector currently preserves the figure area rather than separating captions.
- Some figure groups are semantically valid but not ideal for final scholarly layout.

## Text False Positives

No common accepted plain-text false positives were observed in this 34-page review. Title text, body text, vertical text, rotated text, and sparse text bands were generally rejected.

## Missed Visual Regions

- `3330` is the clearest miss: several tall visual strips were rejected while one similar strip was accepted.
- Other reviewed known visual pages retained at least the main diagram/image content.

## Duplicate / Broad Crop Issues

- Duplicates are mostly controlled after post-filter consolidation.
- Broad crops remain on `3327` and `3352`; they are acceptable for preservation/review but not perfect extraction.
- `3312` remains split into upper and lower right-edge figure/photo regions, which may be acceptable because the content is visually stacked.

## Runtime Integration Recommendation

Recommended status: safe to integrate behind a disabled-by-default option.

The detector is useful enough to enrich `DocumentResult` in a review workflow, provided it is explicitly enabled and failures remain non-fatal. It should not be enabled by default for all processing yet.

Suggested integration constraints:

- Add an opt-in config such as `ENABLE_IMAGE_REGION_EXTRACTION=false`.
- Keep extraction failures non-fatal.
- Store accepted image regions as candidate regions, not authoritative layout truth.
- Preserve rejection/consolidation diagnostics in metadata when review mode is enabled.
- Continue Notebook 05 corpus review before enabling broadly.

## Next Coding Recommendation

Add a conservative `ExtractionService` behind a disabled-by-default option. The service should enrich `DocumentResult` with candidate `ImageRegion` objects and optional diagnostics, but it should not replace OCR text flow or page reconstruction decisions yet.
