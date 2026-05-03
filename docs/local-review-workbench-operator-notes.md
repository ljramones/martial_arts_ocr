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
- add a manual region;
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

- No OCR button yet.
- No translation button yet.
- No PDF/DOCX export.
- No multi-user coordination.
- No polished annotation UI.

The next slice should add reviewed-region OCR controls without changing OCR defaults or mutating source text.
