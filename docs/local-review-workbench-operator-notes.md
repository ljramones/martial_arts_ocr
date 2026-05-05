# Local Review Workbench Operator Notes

## Purpose

These notes describe how to use the first local research review workbench slice.

This is a local research prototype. It is not production OCR software. Region overlays and future OCR results are review aids, not ground truth.

## Start Flask

Use the normal local Flask workflow for this repository. For example:

```bash
.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:5000/review
```

## Create Or Load A Project

In the workbench:

1. Enter a local folder path containing scanned page images.
2. Optionally enter a project ID.
3. Click `Create / Load`.

Supported page image extensions:

```text
jpg
jpeg
png
tif
tiff
bmp
```

The browser does not directly read arbitrary folders. The folder path is interpreted by the local Flask server.

## Project Storage

Review projects are stored under:

```text
data/runtime/review_projects/<project_id>/project_state.json
```

`data/runtime/` is ignored and should not be committed.

## Region Editing

The workbench supports manual region review:

- select a page;
- choose `Select / Move`, `Draw Image Region`, `Draw Text Region`, or `Draw Japanese Region`;
- drag a rectangle on the page to create a reviewed manual region;
- use `Draw Ignore Region` to mark page areas that should be skipped;
- keep drawing to create multiple regions in sequence;
- add a default manual region if preferred;
- click a region overlay or region list item;
- drag the region to move it;
- drag corner handles to resize it;
- edit bbox numbers for precise changes;
- change region type;
- add notes;
- save the region.

The state model preserves:

```text
detected_type
reviewed_type
effective_type
detected_bbox
reviewed_bbox
effective_bbox
```

Reviewer edits update reviewed/effective fields. They do not overwrite detected fields.

The right panel includes a region inventory grouped by image, text, Japanese, ignored, and other regions. Use it as the page checklist: click any item to select the corresponding overlay box.

Quick type buttons set common review labels such as `image`, `diagram`, `photo`, English text, Japanese horizontal, Japanese vertical, or `ignore`. Advanced duplicate/nudge controls remain available in the collapsible section, but the primary review flow is drawing and correcting meaningful regions.

Numeric bbox fields are for fine adjustment after drawing or selecting a box. They are not the primary way to create regions. When no region is selected, use the page overlay: select a drawing tool and drag directly on the scanned page.

Review actions also record lightweight feedback metadata:

```text
accepted
resized
rejected
manually_added
ignored
```

This feedback is local review data. It is intended for future detector evaluation/training exports, not live learning.

## Overlay Alignment

Workbench bboxes are stored in natural image coordinates:

```text
[x, y, width, height]
```

The browser maps those coordinates to the rendered image using the image's displayed size and offset inside the viewer. The overlay should align to the scanned page image, not to the dark viewer container.

If overlays appear misaligned:

- reload the page after server changes;
- check whether the image is still loading;
- check whether the browser is applying EXIF orientation differently from the backend;
- do not save reviewed bboxes until the overlay alignment is correct.

## Run Orientation

After selecting a page, click `Run Orient` before running recognition.

The workbench uses the existing neural-network orientation subsystem through the review-layer orientation service. It does not replace the NN detector with heuristics, and it does not modify the original source image.

Page orientation state is saved in `project_state.json`:

```text
detected_rotation_degrees
detected_confidence
reviewed_rotation_degrees
effective_rotation_degrees
status
source
```

`detected_rotation_degrees` is the NN's current-orientation class for the source image. `effective_rotation_degrees` is the clockwise correction applied to display/process the page upright.

The mapping is:

```text
detected current 0   -> apply correction 0
detected current 90  -> apply correction 270
detected current 180 -> apply correction 180
detected current 270 -> apply correction 90
```

Use the reviewed correction dropdown to override the correction applied when needed. The effective-oriented image is then used for display and for `Run Recognition`.

If orientation changes after regions already exist, the workbench marks regions stale. Rerun recognition or manually review all boxes before using downstream OCR.

## Run Recognition

After selecting a page, click `Run Recognition` to import machine-detected regions for that page.

Run orientation first. Recognition uses the effective-oriented page, not the raw sideways/upside-down source image.

This uses the existing review-mode region/image detection path as advisory input. It does not run OCR, translation, or Japanese analysis.

Review-mode recognition also includes a conservative multi-figure row proposal pass. It is intended to suggest sibling figure/photo/diagram panels that may otherwise be missed in rows of related figures. These proposals are still advisory: keep, resize, retype, ignore, or delete them during review.

Imported machine regions use:

```text
source=machine_detection
status=detected
detected_type
detected_bbox
effective_type
effective_bbox
metadata.detector
```

Rerun behavior is conservative:

- unreviewed `source=machine_detection` regions are replaced;
- manual regions are preserved;
- reviewed regions are preserved;
- ignored regions are preserved.

Selected-region audit fields show detector metadata when available, including confidence, mixed-region flags, needs-review flags, layout fusion metadata, and region role.

## Run Selected-Region OCR

After a region has a reviewed/effective type and bbox, select it and click `Run OCR`.

Selected-region OCR:

- runs only on the selected region;
- crops from the effective-oriented page;
- uses the selected region's `effective_bbox`;
- routes from the selected region's `effective_type`;
- stores an OCR attempt in `project_state.json`;
- shows the latest OCR output in the right panel.

Initial routing:

```text
english_text                  -> eng, PSM 6
romanized_japanese_text       -> eng, PSM 6
caption_label                 -> eng, PSM 7
modern_japanese_horizontal    -> jpn, PSM 6, upscale_2x
modern_japanese_vertical      -> jpn_vert, PSM 5, upscale_2x
mixed_english_japanese        -> eng+jpn, PSM 6
```

`image`, `diagram`, `photo`, `ignore`, and unknown regions are skipped in this slice. OCR attempts are review artifacts; they do not mutate source images, OCR text elsewhere, canonical Japanese fields, or runtime defaults.

Use `Run Variants` when the default selected-region OCR is poor. This runs a small review-mode matrix for the selected region only, stores each output as a separate OCR attempt, and selects the highest-scored attempt as the latest region OCR result. Variants may include alternate PSMs and preprocessing such as grayscale, threshold, upscale, or contrast/sharpen depending on region type.

Variant results are comparison evidence, not truth. Review the actual output before accepting or using it downstream.

## Review OCR Attempts

The OCR panel separates machine output from reviewed text:

- `OCR output` is the raw/cleaned OCR attempt output and is read-only.
- `Reviewed text` is the local reviewer correction field.
- `Accept OCR` marks the attempt accepted and stores the current reviewed text.
- `Save Reviewed Text` marks the attempt edited and stores the corrected text.
- `Reject OCR` marks the attempt rejected and leaves the raw OCR evidence intact.

OCR attempt records keep raw fields and review fields separate:

```text
text
cleaned_text
reviewed_text
review_status
source_text_mutated=false
```

Use reviewed text for cases where the scan is legible to a human but OCR misses a dirty or typewritten line. For example, if OCR misses `[Question.]`, enter the corrected line in `Reviewed text` and save it as edited. This does not mutate the OCR attempt's raw text, page image, region bbox, or canonical document fields.

## Export Reviewed Page State

Click `Export Page` after reviewing regions and OCR attempts for the selected page.

The workbench writes a timestamped local export under:

```text
data/runtime/review_projects/<project_id>/exports/<timestamp>/
```

Initial export files:

```text
project_state_snapshot.json
page_<page_id>_review.json
page_<page_id>_review.md
page_<page_id>_text.txt
crops/
  region_<region_id>.png
```

Export behavior:

- `page_<page_id>_review.json` preserves region geometry, source, review status, OCR route, raw OCR, reviewed text, and `source_text_mutated=false`.
- `page_<page_id>_review.md` is a human-readable review artifact.
- `page_<page_id>_text.txt` concatenates reviewed text-region output by page order.
- Plain text export prefers `reviewed_text` when present, then cleaned/raw OCR text.
- Region crops use the effective-oriented page image and each region's effective bbox.
- Ignored regions remain in JSON metadata but are omitted from text export and crop export.

Generated exports live under `data/runtime/` and should not be committed.

## Export v2: Multi-Page Bundle And HTML

Use `Export Selection` when you want a multi-page research artifact.

Page selection modes:

- `Current page`: exports the selected page.
- `Selected pages`: exports the highlighted page IDs in the selected-pages list.
- `Page range`: exports all pages between the chosen start and end page IDs, using project page order.
- `All pages`: exports every page in the project.

Formats:

- `Review bundle`: multi-page audit/recovery artifacts.
- `HTML`: clean research reconstruction with reviewed text and image crops.
- `DOCX later` and `PDF later` are intentionally disabled in this slice.

Export v2 writes under:

```text
data/runtime/review_projects/<project_id>/exports/<timestamp>/
```

Initial v2 structure:

```text
export_manifest.json
project_state_snapshot.json
document_export_model.json
review_bundle/
  pages/
    page_<page_id>_review.json
    page_<page_id>_review.md
    page_<page_id>_text.txt
  crops/
    page_<page_id>_region_<region_id>.png
html/
  document.html
  assets/
    page_<page_id>_region_<region_id>.png
```

The existing `Export Page` button and page-level endpoint remain available for the simpler current-page bundle. `Export Selection` uses the project-level Export v2 endpoint and can produce the review bundle and HTML in one run.

Export v2 still follows the same non-destructive rules: reviewed text is preferred for display/plain text, raw OCR is preserved in JSON/Markdown, source text is not mutated, ignored regions are skipped from text and crops by default, and generated exports are local runtime artifacts.

## Duplicate and Nudge Regions

If recognition finds one region in a repeated row but misses nearby siblings, prefer `Draw Image Region` for the missing boxes. The selected-region panel also has advanced duplicate/nudge controls:

- `Duplicate` creates a same-position manual copy.
- `Duplicate Left` / `Duplicate Right` creates same-sized sibling boxes offset by one region width.
- `Duplicate Up` / `Duplicate Down` creates same-sized sibling boxes offset by one region height.
- Nudge buttons move the selected region by 10 pixels.
- Arrow keys move the selected region by 1 pixel when focus is not inside an input.
- Shift + arrow moves by 10 pixels.

Duplicated regions are reviewer-created, not machine detections:

```text
source=reviewer_manual_duplicate
status=reviewed
detected_bbox=null
metadata.duplicated_from_region_id=<source region>
```

Save the duplicated region after final bbox/type adjustment.

Manually drawn regions are stored as reviewer-created evidence:

```text
source=reviewer_manual
detected_bbox=null
reviewed_bbox=<drawn bbox>
effective_bbox=<drawn bbox>
metadata.feedback_type=missed_positive
metadata.manually_added=true
```

Those records can later be exported as missed-positive detector feedback.

## Recognition Diagnostics

After `Run Recognition`, the page state includes compact recognition diagnostics under:

```text
pages[].recognition_diagnostics
```

The diagnostics summarize the candidate lifecycle:

```text
raw candidates
accepted candidates
rejected candidates
suppressed / merged candidates
imported workbench regions
```

Use the `Recognition Diagnostics` panel to inspect whether a missing figure was:

- never proposed as a raw candidate;
- rejected as text-like;
- suppressed by top-k or consolidation;
- merged into another region;
- accepted but not imported.

The diagnostics include per-detector summaries, including the multi-figure row proposal pass when enabled.

This panel is diagnostic only. It does not change detector thresholds or imported-region behavior.

## Region Types

Current region type options:

```text
ignore
image
diagram
photo
english_text
romanized_japanese_text
modern_japanese_horizontal
modern_japanese_vertical
mixed_english_japanese
caption_label
unknown_needs_review
```

Use `unknown_needs_review` when the region is unclear.

Use `ignore` for irrelevant or false-positive regions. Ignored regions remain visible in project state unless deleted.

## What Not To Commit

Do not commit:

```text
data/runtime/
data/notebook_outputs/
generated review projects
generated crops
private corpus images
```

Commit only source code, templates/static assets, tests, and docs.

## Current Limits

- No translation button yet.
- No PDF/DOCX export.
- No multi-user coordination.
- No polished annotation UI.
- No OCR-all-regions action yet; OCR is selected-region only.

The next slice should focus on either OCR-all-reviewed-regions or export ergonomics after page-level exports have been exercised on real pages.
