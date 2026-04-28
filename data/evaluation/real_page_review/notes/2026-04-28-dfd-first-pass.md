# Donn Draeger DFD Notes — Extraction Review First Pass

Date: 2026-04-28  
Manifest: data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json  
Notebook: notebooks/05_real_page_extraction_review.ipynb

## Scope

This first pass reviews a small, varied subset of the Donn Draeger DFD notes corpus before runtime integration.

The goal is to identify real failure patterns in:

- image/diagram region detection
- crop quality
- text cleanup
- Japanese/macron preservation
- line-break preservation
- layout / reading order

This is not a tuning pass yet. Findings should drive the next code pass.

Reviewed pages were selected from the generated 112-page manifest after visual contact-sheet inspection. Review outputs were written under `data/notebook_outputs/real_page_review/`. The generated manifest does not include `sample_text` or OCR output, so Notebook 05 skipped real-page text cleanup for these pages.

## Pages Reviewed

### Page / Sample ID: original_img_3288
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg

Image-region extraction:
- Result: fail
- False positives: two large title text fragments were detected as diagram regions.
- False negatives: no obvious diagram content expected on this sparse title/list page.
- Crop quality: crops are readable but are text, not image/diagram content.
- Bounding box notes: boxes tightly surround bold typewritten text, indicating text filtering is not strong enough for titles/headings.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated; no OCR/sample text in manifest.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: fail
- Notes: false image regions would pollute any future `ImageRegion` payload.

Overall:
- Safe for runtime integration? no
- Required fixes: reject bold title/typewritten text as image regions.

---

### Page / Sample ID: original_img_3292
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3292.jpg

Image-region extraction:
- Result: partial
- False positives: one vertical text fragment was detected as a diagram.
- False negatives: the body-outline diagram was detected.
- Crop quality: diagram crop is useful; false-positive crop is only text.
- Bounding box notes: diagram bbox is usable, but detector also accepts narrow vertical text.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: partial
- Notes: page contains rotated content; future reading-order logic must account for orientation.

Overall:
- Safe for runtime integration? maybe, after false-positive filtering
- Required fixes: filter narrow text-like vertical blocks without losing real diagrams.

---

### Page / Sample ID: original_img_3312
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg

Image-region extraction:
- Result: fail
- False positives: two vertical chunks of body text were detected as diagrams.
- False negatives: visible human-figure/photo content was not captured as useful full image regions.
- Crop quality: crops are clear but incorrect.
- Bounding box notes: narrow text columns were preferred over larger visual figure areas.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: fail
- Notes: detector output would misrepresent this page as image-heavy text snippets.

Overall:
- Safe for runtime integration? no
- Required fixes: improve figure/photo detection recall and reject text-column crops.

---

### Page / Sample ID: original_img_3327
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3327.jpg

Image-region extraction:
- Result: partial
- False positives: one crop includes a text phrase near the figure area.
- False negatives: major line-drawing regions were detected.
- Crop quality: the bridge/structure crop is useful; the second crop mixes text with nearby image content.
- Bounding box notes: bbox placement is close enough for one diagram, but association with captions/text is weak.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: partial
- Notes: diagram/caption proximity will need explicit handling later.

Overall:
- Safe for runtime integration? maybe, but not yet
- Required fixes: improve crop separation around captions and nearby body text.

---

### Page / Sample ID: original_img_3331
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3331.jpg

Image-region extraction:
- Result: fail
- False positives: both detected regions are text fragments.
- False negatives: no obvious standalone diagram was present in this page view.
- Crop quality: text crops are clear but semantically wrong.
- Bounding box notes: detector is over-sensitive to dense typewritten text.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: fail
- Notes: false image regions would contaminate reading-order and artifact output.

Overall:
- Safe for runtime integration? no
- Required fixes: tune text-like rejection before integration.

---

### Page / Sample ID: original_img_3340
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3340.jpg

Image-region extraction:
- Result: partial
- False positives: one body-text fragment was detected as a diagram.
- False negatives: hand/gesture diagram was detected.
- Crop quality: hand diagram crop is useful; text crop is false positive.
- Bounding box notes: real diagram bbox is good enough for review; false positive indicates minimum area/text filtering issue.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: partial
- Notes: image detection is useful but noisy.

Overall:
- Safe for runtime integration? maybe, after filtering
- Required fixes: reject body-text crops while preserving line drawings.

---

### Page / Sample ID: original_img_3378
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3378.jpg

Image-region extraction:
- Result: pass
- False positives: none.
- False negatives: none obvious; page appears mostly blank/notebook cover with very little content.
- Crop quality: no crops generated.
- Bounding box notes: no regions.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: pass
- Notes: correctly avoided extracting noise from a sparse page.

Overall:
- Safe for runtime integration? yes for this page class
- Required fixes: none from this page.

---

### Page / Sample ID: original_img_3380
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3380.jpg

Image-region extraction:
- Result: fail
- False positives: one upside-down typewritten text fragment was detected as a diagram.
- False negatives: no obvious diagram expected.
- Crop quality: crop is readable but incorrect.
- Bounding box notes: orientation/noise conditions allow text to pass as diagram.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: fail
- Notes: rotated/sparse pages need orientation-aware filtering.

Overall:
- Safe for runtime integration? no
- Required fixes: orientation-aware text-like rejection.

---

### Page / Sample ID: original_img_3389
Path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3389.jpg

Image-region extraction:
- Result: fail
- False positives: two vertical body-text fragments were detected as diagrams.
- False negatives: no clear diagram was captured from the visible content.
- Crop quality: crops are text only.
- Bounding box notes: vertical text columns are recurring false positives.

Text cleanup:
- Result: not evaluated
- Japanese preserved: not evaluated.
- Macrons preserved: not evaluated.
- Line breaks preserved: not evaluated.
- Damaged terms: not evaluated.

Layout / reading order:
- Result: fail
- Notes: rotated text columns are likely a major failure class.

Overall:
- Safe for runtime integration? no
- Required fixes: stronger text-column rejection and orientation handling.

---

## Cross-Page Findings

### Image / Diagram Extraction

Reviewed 9 pages. The current detector can find some real line drawings, especially isolated diagrams on `original_img_3292`, `original_img_3327`, and `original_img_3340`. However, the dominant failure is false-positive detection of typewritten text fragments as `diagram` regions.

### Text Cleanup

Real-page text cleanup was not evaluated because the generated manifest does not include `sample_text` or OCR output, and Notebook 05 does not run OCR. The next review should add OCR-like snippets for selected pages or run a controlled OCR pass separately.

### Japanese / Macron Preservation

Not evaluated from the reviewed real pages. Existing utility tests cover synthetic preservation, but this pass did not produce real OCR text to verify Japanese/macron survival.

### Layout / Reading Order

Not safe yet. False image regions would interrupt any future reading-order model and would pollute `DocumentResult.image_regions`.

### Crop Quality

When the detected region is a real diagram, crop quality is generally usable. The main crop problem is semantic, not mechanical: many crops are clear, tightly bounded text fragments rather than images.

## Failure Patterns

- Dense typewritten body text is frequently classified as a diagram/image region.
- Bold headings and short title text can be classified as diagram regions.
- Narrow vertical or rotated text blocks are accepted as diagrams.
- Some real figure/photo content is missed while nearby text snippets are detected.
- Sparse/no-content pages can be handled correctly, as shown by `original_img_3378`.
- Text cleanup cannot be assessed with the current local manifest because no OCR/sample text is present.

## Recommended Next Coding Pass

Tune image-region detection before runtime integration:

- strengthen text-like rejection for typewritten text fragments
- add orientation-aware filtering for rotated/vertical text columns
- adjust detector thresholds so line drawings are preserved but text blocks are rejected
- consider separate detector strategies for line drawings/photos versus text-column rejection if threshold tuning is not enough

A secondary pass should extend the review harness to include OCR/sample text for selected pages so text cleanup, Japanese preservation, macrons, and line breaks can be evaluated on real page content.

## Integration Decision

Recommended status:

- [ ] Safe to integrate now
- [x] Tune thresholds first
- [ ] Add detector strategy abstraction first
- [ ] Improve text cleanup first
- [x] More manual review needed
