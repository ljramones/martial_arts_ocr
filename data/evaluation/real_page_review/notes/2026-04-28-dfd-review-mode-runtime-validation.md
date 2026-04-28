# Donn Draeger DFD Notes — Review-Mode Runtime Validation

Date: 2026-04-28  
Corpus: `data/corpora/donn_draeger/dfd_notes_master/original/`  
Runtime output root: `data/runtime/processed/`  
Mode: `WorkflowOrchestrator` with `ExtractionService(enable_image_regions=True)`  

## Scope

This pass validates the runtime review-mode path after gated image-region extraction integration.

The goal is not OCR quality validation. OCR was stubbed with a lightweight canonical `DocumentResult` so the pass could isolate:

- `WorkflowOrchestrator` artifact writing
- `ExtractionService` enrichment
- `data.json` image-region serialization
- crop path generation
- disabled-mode behavior

No runtime defaults were changed.

## Pages Processed

### original_img_3335

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg`

Artifacts:
- data.json: present
- page_data.json: present
- page_1.html: present
- text.txt: present
- crops folder: `data/runtime/processed/doc_93335/image_regions/`

Image regions:
- count: 1
- crop paths valid: yes
- crop quality: large recovered figure candidate; runtime crop path valid
- notes: bbox `x=182, y=469, width=868, height=970`; one text-like candidate was rejected

Runtime behavior:
- extraction errors: none
- OCR/pipeline status: completed
- disabled-mode comparison: representative disabled run on this sample produced 0 image regions

Integration assessment: pass

### original_img_3344

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg`

Artifacts:
- data.json: present
- page_data.json: present
- page_1.html: present
- text.txt: present
- crops folder: `data/runtime/processed/doc_93344/image_regions/`

Image regions:
- count: 1
- crop paths valid: yes
- crop quality: labeled diagram candidate retained
- notes: bbox `x=873, y=114, width=317, height=331`; three text-like candidates rejected

Runtime behavior:
- extraction errors: none
- OCR/pipeline status: completed
- disabled-mode comparison: covered by representative disabled run

Integration assessment: pass

### original_img_3397

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3397.jpg`

Artifacts:
- data.json: present
- page_data.json: present
- page_1.html: present
- text.txt: present
- crops folder: `data/runtime/processed/doc_93397/image_regions/`

Image regions:
- count: 2
- crop paths valid: yes
- crop quality: sparse symbols/arrows represented as saved candidate crops
- notes: bboxes `x=24, y=864, width=551, height=718` and `x=636, y=1199, width=251, height=246`

Runtime behavior:
- extraction errors: none
- OCR/pipeline status: completed
- disabled-mode comparison: covered by representative disabled run

Integration assessment: pass

### original_img_3352

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3352.jpg`

Artifacts:
- data.json: present
- page_data.json: present
- page_1.html: present
- text.txt: present
- crops folder: `data/runtime/processed/doc_93352/image_regions/`

Image regions:
- count: 3
- crop paths valid: yes
- crop quality: runtime crops valid; known broad/label-heavy case remains review-only
- notes: one adjacent merge recorded; accepted bboxes include a broad upper crop `x=16, y=334, width=1184, height=378`

Runtime behavior:
- extraction errors: none
- OCR/pipeline status: completed
- disabled-mode comparison: covered by representative disabled run

Integration assessment: partial because crop quality remains review-mode only, but runtime behavior is correct

### original_img_3330

Input path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg`

Artifacts:
- data.json: present
- page_data.json: present
- page_1.html: present
- text.txt: present
- crops folder: `data/runtime/processed/doc_93330/image_regions/`

Image regions:
- count: 1
- crop paths valid: yes
- crop quality: one tall/narrow candidate retained; other tall strips rejected as text-like/aspect-ratio risks
- notes: accepted bbox `x=664, y=176, width=197, height=907`; this remains the fragile tall/narrow-strip case

Runtime behavior:
- extraction errors: none
- OCR/pipeline status: completed
- disabled-mode comparison: covered by representative disabled run

Integration assessment: partial because the detector remains conservative on tall/narrow strips, but runtime behavior is correct

## Disabled-Mode Comparison

Representative disabled run:

- sample: `original_img_3335`
- output: `data/runtime/processed/doc_98335/`
- artifacts: `data.json`, `page_data.json`, `page_1.html`, and `text.txt` all present
- image regions: 0
- extraction metadata: absent
- text output: unchanged fake OCR text
- pipeline status: completed

This confirms disabled mode remains unchanged for normal processing.

## Summary

Pages processed: 5

Enabled-mode behavior:
- `data.json` was produced for every selected page.
- `pages[].image_regions[]` was present when extraction produced candidates.
- `image_regions[].image_path` pointed to existing crop files.
- Crops were saved under `data/runtime/processed/doc_<id>/image_regions/`.
- `page_data.json`, `page_1.html`, and `text.txt` were preserved.
- No extraction failures occurred.

Disabled-mode behavior:
- Extraction disabled produced no image regions.
- Extraction metadata was absent.
- Standard artifacts were still produced.

## Runtime Bugs Found

No runtime integration bugs were found in this pass.

The remaining issues are detector-quality issues already identified in review:

- broad/label-heavy crops on pages like 3352
- conservative handling of tall/narrow visual strips on pages like 3330

## Integration Assessment

Review-mode integration is usable for broader corpus testing.

It should remain disabled by default. The runtime path is functioning, but detector output should still be treated as candidate image regions for review workflows, not authoritative production layout.

Recommended next step: run this review-mode pipeline on a larger DFD subset and then on a second corpus to check whether the detector generalizes beyond DFD notes.
