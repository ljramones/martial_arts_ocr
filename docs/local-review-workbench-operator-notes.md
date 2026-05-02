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

The first slice supports manual region review:

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
- No automatic region recognition button yet.
- No PDF/DOCX export.
- No multi-user coordination.
- No polished annotation UI.

The next slice should wire existing review-mode region detection into detected regions, then keep reviewer edits as overrides.
