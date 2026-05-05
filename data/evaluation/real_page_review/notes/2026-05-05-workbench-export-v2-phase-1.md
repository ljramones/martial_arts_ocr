# Workbench Export v2 Phase 1

## Purpose

Implement the first Export v2 slice for the local research review workbench: multi-page review bundles and readable HTML exports from reviewed project state.

## What Changed

- Added a project-level Export v2 endpoint beside the existing page-level export endpoint.
- Added page selection modes for current page, selected pages, page range, and all pages.
- Added a shared document/page export model written as `document_export_model.json`.
- Added `export_manifest.json` with selected pages, requested formats, export version, and `source_text_mutated=false`.
- Added multi-page `review_bundle` output with per-page JSON, Markdown, text, and crops.
- Added HTML export with simple readable page sections, reviewed text, and crop assets.
- Added workbench UI controls for page selection and format selection.

## Export Formats

Implemented:

- `review_bundle`
- `html`

Deferred:

- `docx`
- `pdf`
- translation export
- pixel-perfect scanned-page layout

## Page Selection

Supported modes:

- `current`: selected page only.
- `selected`: explicitly selected page IDs.
- `range`: page IDs between start and end in project order.
- `all`: every page in the project.

The backend rejects empty selections, unknown page IDs, and unsupported modes.

## Text And OCR Behavior

Text export and HTML display use:

1. selected/latest OCR attempt `reviewed_text`;
2. selected/latest OCR attempt `cleaned_text`;
3. selected/latest OCR attempt raw text;
4. empty text.

Raw OCR is preserved in JSON/Markdown artifacts. Reviewed text remains separate. `source_text_mutated=false` is preserved.

## Crop Behavior

Crops use the effective-oriented page image and each region's `effective_bbox`.

Ignored regions remain in JSON metadata but are skipped from text export and crop export by default.

## Validation

Automated tests use generated temporary images and cover:

- multi-page review bundle export;
- HTML export;
- reviewed text preference;
- raw OCR preservation;
- ignored-region skip behavior;
- image/diagram crop writing;
- selected and range page modes;
- unsupported format rejection;
- page-level endpoint compatibility.

## Limitations

- HTML is a clean research reconstruction, not a page facsimile.
- Region ordering is still page order, then `y`, then `x`.
- DOCX/PDF are intentionally disabled until the shared export model is exercised on real reviewed pages.
- Export artifacts live under ignored runtime paths and may contain private source-derived crops.
