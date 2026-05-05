# Workbench Export v2 Design

## Purpose

Export v2 turns reviewed local workbench state into durable research artifacts across one or more pages.

It should support:

- single-page export;
- multi-page export;
- readable HTML;
- the current audit/review bundle;
- later DOCX and PDF generation from the same shared export model.

This is still a local research workbench feature, not a polished publishing system. Exported documents should be useful for research review and sharing with trusted collaborators, while preserving enough audit data to recover raw OCR, reviewed text, region geometry, and uncertainty.

## Current Export Behavior

Current export is page-level only.

It writes a timestamped local export under:

```text
data/runtime/review_projects/<project_id>/exports/<timestamp>/
```

Current files:

```text
project_state_snapshot.json
page_<page_id>_review.json
page_<page_id>_review.md
page_<page_id>_text.txt
crops/
  region_<region_id>.png
```

Current behavior:

- `reviewed_text` is preferred for text export when present.
- `cleaned_text` / raw OCR text remain preserved.
- `source_text_mutated=false` is preserved.
- crops use the effective-oriented page image and each region's effective bbox.
- ignored regions remain in JSON metadata but are skipped for text and crop output.

This current export should remain as the audit/recovery bundle. Export v2 extends it rather than replacing it.

## Goals

- Export current page, selected pages, page range, or all pages.
- Export one or more formats in a single run.
- Preserve auditability across all formats.
- Keep raw OCR and reviewed text separate.
- Produce readable HTML before DOCX/PDF.
- Support future DOCX/PDF through the same export model rather than format-specific reconstruction logic.
- Keep generated exports under ignored runtime paths.

## Non-Goals

- No pixel-perfect facsimile of the scanned page.
- No silent source text mutation.
- No automatic OCR correction.
- No translation export unless translation is already reviewed/stored in a later workflow.
- No DOCX/PDF implementation in the first Export v2 slice.
- No dependency additions in this design pass.
- No polished publishing layout yet.
- No mutation of project state during export.

## Page Selection

Export v2 should support these selection modes:

- `current_page`: selected page in the UI.
- `selected_pages`: explicit user-selected page IDs from the page list.
- `page_range`: start/end page indexes or page IDs.
- `all_pages`: every page in the project.

Validation rules:

- empty selection is rejected;
- invalid page IDs are rejected;
- range endpoints must exist;
- range order may be normalized if both endpoints are valid, or rejected if the UI chooses stricter behavior;
- final export ordering follows project page order.

The resolved page ID list should be stored in the export manifest so the artifact remains understandable after the runtime project changes.

## Format Selection

Format IDs:

```text
review_bundle
html
docx
pdf
```

First implementation slice:

```text
review_bundle
html
```

Deferred:

```text
docx
pdf
```

The backend should reject unsupported formats until implemented. The UI can show DOCX/PDF as disabled with "coming later."

## Shared Export Model

All export formats should use a neutral in-memory export model before writing files.

Conceptual shape:

```json
{
  "project_id": "example",
  "pages": [
    {
      "page_id": "page_025",
      "source_path": "...",
      "orientation": {},
      "blocks": [
        {
          "block_id": "page_025_r_001",
          "type": "text",
          "region_id": "r_001",
          "bbox": [180, 70, 940, 180],
          "text": "[Question.]\\nAaaa yes.",
          "raw_ocr": "(Question. ]\\n\\nAaaa yes.",
          "review_status": "edited",
          "asset_path": null
        },
        {
          "block_id": "page_025_r_003",
          "type": "image",
          "region_id": "r_003",
          "bbox": [463, 853, 340, 455],
          "text": "",
          "raw_ocr": "",
          "review_status": "manually_added",
          "asset_path": "assets/page_025_region_r_003.png"
        }
      ]
    }
  ],
  "metadata": {
    "source_text_mutated": false
  }
}
```

The model should centralize:

- page selection resolution;
- text preference;
- block ordering;
- asset naming;
- warning/uncertainty flags.

## PageExportModel

Page-level fields:

```text
page_id
page_index
source_path
filename
orientation
review_status
blocks[]
assets[]
warnings[]
```

`warnings[]` can include:

```text
regions_unreviewed
ocr_unreviewed
reading_order_uncertain
no_text_regions
regions_stale
```

Page export models should be derived from current project state and should not write back to `project_state.json`.

## DocumentExportModel

Document-level fields:

```text
project_id
source_folder
pages[]
formats_requested[]
created_at
export_version
source_text_mutated=false
page_selection
warnings[]
```

The document model should support single-page exports as a one-page document and multi-page exports as a sequence of page models.

## Text Selection Rules

For each text-like region:

```text
1. selected/latest OCR attempt reviewed_text
2. selected/latest OCR attempt cleaned_text
3. selected/latest OCR attempt raw text
4. region notes or empty
```

Rules:

- skip ignored regions for plain text and HTML body output by default;
- preserve raw OCR in JSON/Markdown review bundle artifacts;
- HTML uses the selected display text, preferably reviewed text;
- rejected OCR attempts should not contribute display text unless explicitly included in a future audit view;
- `source_text_mutated=false` remains explicit in manifest/model metadata.

Text-like region types:

```text
english_text
romanized_japanese_text
modern_japanese_horizontal
modern_japanese_vertical
mixed_english_japanese
caption_label
```

## Region Ordering Rules

Initial ordering:

```text
project page order
then region y
then region x
```

This is not full layout reconstruction. It is a pragmatic reading-order approximation for reviewed research exports.

When available, the export model should surface uncertainty:

```text
reading_order_uncertain=true
```

Future work can incorporate explicit reviewer ordering, columns, caption/image grouping, and richer page reconstruction.

## Crop / Asset Rules

Asset rules:

- crops use the effective-oriented page image;
- crops use each region's `effective_bbox`;
- ignored regions are not cropped by default;
- asset filenames include page ID and region ID;
- assets are copied/written per export so exports remain self-contained.

Example asset path:

```text
assets/page_025_region_r001.png
```

For review bundles, crops may live under:

```text
review_bundle/crops/page_025_region_r001.png
```

For HTML, assets may live under:

```text
html/assets/page_025_region_r001.png
```

The first implementation may generate separate copies for review bundle and HTML for simplicity. A later implementation can deduplicate assets if needed.

## Export Folder Structure

Target structure:

```text
exports/<timestamp>/
  export_manifest.json
  project_state_snapshot.json

  review_bundle/
    pages/
      page_001_review.json
      page_001_review.md
      page_001_text.txt
      page_002_review.json
      page_002_review.md
      page_002_text.txt
    crops/
      page_001_region_r001.png
      page_002_region_r003.png

  html/
    document.html
    assets/
      page_001_region_r001.png
      page_002_region_r003.png

  docx/
    document.docx

  pdf/
    document.pdf
```

For the first implementation slice, create only:

```text
exports/<timestamp>/
  export_manifest.json
  project_state_snapshot.json
  review_bundle/
  html/
```

The existing page-level export endpoint can remain for backward compatibility.

## Export Manifest

Manifest shape:

```json
{
  "project_id": "example",
  "created_at": "2026-05-05T00:00:00",
  "page_selection": {
    "mode": "current",
    "page_ids": ["page_025"]
  },
  "formats": ["review_bundle", "html"],
  "source_text_mutated": false,
  "export_version": 2
}
```

Additional future fields:

```text
export_id
source_folder
warnings
artifact_paths
format_versions
```

## HTML Export Design

HTML should be a clean research reconstruction, not a pixel-perfect copy of the scanned page.

Recommended structure:

```text
document heading
project/source metadata
per-page sections
reviewed text blocks
image/diagram/photo crop blocks
caption/label text near page order
warnings such as reading_order_uncertain
```

HTML can use simple embedded CSS and relative asset paths. No complex layout framework is needed.

Potential per-block rendering:

- text block: heading with region ID/type, then reviewed/display text;
- image block: crop image plus region metadata;
- caption block: styled as caption text;
- uncertain/unreviewed block: visible warning marker;
- raw OCR/details: deferred, or collapsed later.

## Review Bundle Export Design

The current page export bundle should become multi-page capable.

It should preserve:

- project snapshot;
- page review JSON;
- page review Markdown;
- page text;
- crops;
- raw OCR;
- reviewed text;
- region source/status/feedback metadata;
- `source_text_mutated=false`.

Review bundle is the audit/recovery format. HTML/DOCX/PDF are readable derived formats.

## DOCX Export Design Later

DOCX should be generated from `DocumentExportModel`, not directly from project state.

Expected output:

```text
document title/source metadata
page headings
reviewed text blocks
image crops
captions/labels
basic warnings/notes
```

Do not add a DOCX dependency until the shared export model and HTML output have been validated on real pages.

## PDF Export Design Later

PDF should also be generated from `DocumentExportModel`.

The first PDF implementation should be a clean research reconstruction, not a scanned-page facsimile. A later PDF path could use HTML-to-PDF or another renderer, but dependency choice should wait until HTML export behavior is stable.

## UI Design

Use an Export modal or compact panel rather than adding many controls to the main page editor.

Proposed UI:

```text
Export

Pages:
  ( ) Current page
  ( ) Selected pages
  ( ) Range: [page_001] to [page_005]
  ( ) All pages

Formats:
  [x] Review bundle
  [x] HTML
  [ ] DOCX (coming later)
  [ ] PDF (coming later)

[Export]
```

First implementation slice:

- enable `Current page`, `Range`, and `All pages` if straightforward;
- enable `review_bundle` and `html`;
- keep DOCX/PDF disabled.

The current `Export Page` button can remain as a simple shortcut for current-page review bundle export.

## Backend/API Design

Add project-level endpoint:

```text
POST /api/review/projects/<project_id>/export
```

Request body:

```json
{
  "page_selection": {
    "mode": "current",
    "page_ids": ["page_025"],
    "range": null
  },
  "formats": ["review_bundle", "html"]
}
```

Response:

```json
{
  "export_id": "20260505_103000",
  "export_path": "data/runtime/review_projects/example/exports/20260505_103000",
  "formats": {
    "review_bundle": "review_bundle/",
    "html": "html/document.html"
  },
  "page_ids": ["page_025"]
}
```

Current page-level endpoint can remain:

```text
POST /api/review/projects/<project_id>/pages/<page_id>/export
```

It can eventually call the same Export v2 service with:

```text
mode=current
page_ids=[page_id]
formats=[review_bundle]
```

## Safety / Non-Destructive Rules

- Do not mutate source images.
- Do not mutate raw OCR text.
- Keep `reviewed_text` separate.
- Preserve `source_text_mutated=false`.
- Do not treat unreviewed OCR as final truth.
- Generated exports live under ignored runtime paths.
- Exports may include private source-derived crops, so outputs are not committed.
- Export should be repeatable and auditable from `project_state_snapshot.json`.

## Test Strategy

Future tests should cover:

- page selection resolution;
- invalid/empty page selection rejection;
- export model creation;
- reviewed text preference;
- raw OCR preservation;
- `source_text_mutated=false` preservation;
- multi-page review bundle output;
- HTML output contains expected text and images;
- ignored regions skipped from text/html unless configured;
- manifest written;
- generated temp images only;
- page-level endpoint compatibility.

## Implementation Phases

Phase 1:

- shared export model;
- page selection resolution;
- multi-page `review_bundle`;
- HTML export;
- export modal/panel;
- tests.

Phase 2:

- DOCX export from `DocumentExportModel`.

Phase 3:

- PDF export from `DocumentExportModel`.

Phase 4:

- richer layout/styling/options;
- explicit reviewer reading order;
- selected-page checkbox UI;
- optional raw OCR/details in HTML;
- export packaging.

## First Implementation Slice

Implement multi-page review bundle and HTML export only.

Explicitly defer:

- DOCX;
- PDF;
- translation export;
- pixel-perfect scanned-page layout;
- polished publication styling.

This slice should prove the shared model, page selection, manifest, and HTML output on reviewed real pages before adding publishing formats.

Implementation status:

- Phase 1 implemented project-level Export v2 beside the existing page-level endpoint.
- Supported page selections: current page, selected pages, page range, and all pages.
- Supported formats: `review_bundle` and `html`.
- Generated artifacts include `export_manifest.json`, `project_state_snapshot.json`, `document_export_model.json`, multi-page review bundle pages/crops, and `html/document.html` with assets.
- `docx` and `pdf` remain unsupported and should be rejected by the backend until later phases.

## Future Backlog

- DOCX export.
- PDF export.
- ZIP packaging.
- explicit selected-pages UI in the page list.
- reviewer-controlled reading order.
- include/exclude toggles for raw OCR, diagnostics, notes, ignored regions, and crops.
- project-level export history.
- HTML print stylesheet.
- optional side-by-side scanned crop/reviewed text layout.
