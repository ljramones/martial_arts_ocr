# Local Research Review Workbench Design

## Purpose

Design a local web-based research review workbench for scanned martial arts documents.

The workbench should let a researcher load pages, inspect machine-proposed regions, correct region labels and boundaries, run OCR/translation selectively, and export work-in-progress at any stage.

This is not a polished commercial product. It is a local research workbench for the user and trusted collaborators.

## Implementation Status

MVP slice 1 implemented:

- local file-backed project state under `data/runtime/review_projects/`;
- `/review` local workbench page;
- local folder/page listing through server-side paths;
- scanned page image viewer;
- editable SVG region overlays;
- manual region creation;
- region type dropdown;
- bbox drag/resize and numeric bbox editing;
- delete/ignore region controls;
- `project_state.json` save/load behavior;
- detected fields preserved separately from reviewer overrides.

Not implemented in slice 1:

- automatic region recognition button;
- OCR execution;
- translation;
- DOCX/PDF export;
- canonical Japanese field promotion;
- database-backed review projects.

## Research Prototype Assumptions

- This is for local research use.
- It is not polished production software.
- OCR and layout output are advisory.
- Human correction is part of the workflow.
- Recognition results are suggestions, not truth.
- Every recognition result must be inspectable and overrideable.
- Local files and explicit exports are acceptable.
- Runtime defaults should not be silently changed.
- Review state should be auditable: machine output and reviewer overrides must both be preserved.
- The UI should be honest about uncertainty and should show caveats rather than hiding them.

## End-to-End Workflow

1. Researcher opens a local folder or single page.
2. Backend creates or opens a local review project.
3. UI lists pages and displays the selected scanned page.
4. Researcher runs region recognition for the page.
5. Detected regions appear as overlay boxes on the page.
6. Researcher selects a region.
7. Researcher changes region type, moves/resizes bbox, adds notes, or marks the region ignored.
8. Researcher adds missing manual regions or deletes/ignores bad detections.
9. Workbench saves `project_state.json` and page-level review state.
10. Researcher runs OCR for selected reviewed regions.
11. OCR uses effective region type and effective bbox, not raw detector output.
12. Researcher reviews OCR attempts, selects best output, edits notes, or marks output as needing review.
13. Researcher optionally requests translation for selected Japanese regions.
14. Researcher exports current state at any stage.

## User Stories

- As a researcher, I can open a local folder and see its scanned pages without uploading to a remote service.
- As a researcher, I can inspect region detections overlaid on the source image.
- As a researcher, I can correct a detected image region into a vertical Japanese text region.
- As a researcher, I can drag a box edge so OCR runs on the true region boundary.
- As a researcher, I can add a missing caption, label, or Japanese parenthetical region manually.
- As a researcher, I can mark a noisy or irrelevant region as ignored without deleting detector evidence.
- As a researcher, I can run OCR only after reviewing the region type and bbox.
- As a researcher, I can compare OCR attempts and keep all attempts inspectable.
- As a researcher, I can request translation for a selected Japanese OCR result.
- As a researcher, I can export partial work even when the page is not fully reviewed.

## Page / Project State Model

Use file-backed project state first. A database can remain available for existing document processing, but the workbench MVP should not require a schema migration.

Example project structure:

```text
data/runtime/review_projects/<project_id>/
  project_state.json
  pages/
    page_001.json
    page_002.json
  crops/
    page_001/
  exports/
  logs/
```

`project_state.json` should include:

```json
{
  "project_id": "review_20260502_001",
  "schema_version": "research_review_project.v1",
  "source": {
    "input_kind": "folder",
    "source_path": "/local/path/to/pages",
    "allowed_root": "/local/path"
  },
  "pages": [
    {
      "page_id": "page_001",
      "source_image": "/local/path/to/pages/page001.jpg",
      "state_path": "pages/page_001.json",
      "status": "regions_reviewed"
    }
  ],
  "created_at": "2026-05-02T00:00:00",
  "updated_at": "2026-05-02T00:00:00"
}
```

Each page state should include:

```json
{
  "page_id": "page_001",
  "source_image": "/local/path/to/pages/page001.jpg",
  "width": 1240,
  "height": 1754,
  "regions": [],
  "ocr_attempts": [],
  "translation_attempts": [],
  "exports": [],
  "notes": ""
}
```

## Region Model

The region model must preserve both machine detection and reviewer correction.

```json
{
  "region_id": "r_001",
  "detected_type": "image",
  "reviewed_type": "modern_japanese_vertical",
  "effective_type": "modern_japanese_vertical",

  "detected_bbox": [120, 40, 90, 600],
  "reviewed_bbox": [118, 36, 96, 612],
  "effective_bbox": [118, 36, 96, 612],

  "status": "reviewed",
  "source": "reviewer_override",
  "confidence": 0.64,
  "needs_review": false,
  "ignored": false,
  "notes": "Vertical Japanese sidebar; adjust crop before OCR."
}
```

Field meanings:

- `detected_type`: original machine label.
- `reviewed_type`: reviewer override, nullable.
- `effective_type`: `reviewed_type` if present, otherwise `detected_type`.
- `detected_bbox`: original machine bbox `[x, y, width, height]`.
- `reviewed_bbox`: reviewer-adjusted bbox, nullable.
- `effective_bbox`: `reviewed_bbox` if present, otherwise `detected_bbox`.
- `status`: `detected`, `reviewed`, `ignored`, `manual`, `ocr_ready`, `ocr_reviewed`.
- `source`: `detector`, `reviewer_override`, `manual`, `imported`.
- `notes`: free-form researcher note.

Do not overwrite detected fields when reviewer fields change.

## Region Types

Initial region types:

- `ignore`
- `image`
- `diagram`
- `photo`
- `english_text`
- `romanized_japanese_text`
- `modern_japanese_horizontal`
- `modern_japanese_vertical`
- `mixed_english_japanese`
- `caption_label`
- `unknown_needs_review`

Experimental/deferred:

- `stylized_calligraphy`
- `classical_japanese`
- `kuzushiji`
- `table`
- `seal`

Classical Japanese, kuzushiji, densho, and calligraphy should be marked experimental or needs-review. Do not imply current OCR can handle them reliably.

## Boundary Editing / BBox Override Model

BBox editing is MVP scope.

The viewer must support:

- move selected region;
- resize selected region with handles;
- add a new manual region;
- delete or ignore a region;
- save reviewed bbox;
- preserve detected bbox for audit.

The first implementation does not need pixel-perfect drawing tools. It does need reliable basic edits:

- drag box body to move;
- drag corner/edge handles to resize;
- numeric bbox fields for precise correction;
- reset reviewed bbox to detected bbox;
- snap-to-image bounds so bboxes cannot leave the page image;
- mark region as reviewed after bbox/type changes.

Coordinate system:

- store bboxes in original image pixel coordinates;
- draw overlays by scaling image coordinates into viewport coordinates;
- never store viewport/screen coordinates as canonical bbox values.

## Recognition and Override Flow

Region recognition flow:

```text
source page
  -> review-mode extraction
  -> detected regions
  -> overlay boxes
  -> reviewer type/bbox/status edits
  -> effective regions
```

Rules:

- Detector output is advisory.
- Reviewer override is never hidden.
- Ignoring a region should set status/ignored, not erase provenance.
- Manual regions should have no `detected_bbox` and should set `source=manual`.
- All edits should update `updated_at` and optionally append an audit event.

Recommended audit event shape:

```json
{
  "event_id": "evt_001",
  "timestamp": "2026-05-02T00:00:00",
  "actor": "local_user",
  "action": "bbox_updated",
  "region_id": "r_001",
  "before": [120, 40, 90, 600],
  "after": [118, 36, 96, 612]
}
```

## OCR Routing Flow

OCR must use the effective region type and effective bbox.

Routing examples:

| Effective Type | OCR Behavior |
|---|---|
| `english_text` | crop + English OCR |
| `romanized_japanese_text` | crop + English OCR + macron candidate scan |
| `modern_japanese_horizontal` | crop + `jpn`, PSM 6 |
| `modern_japanese_vertical` | crop + `jpn_vert`, PSM 5 |
| `mixed_english_japanese` | run comparison routes; keep goal explicit |
| `caption_label` | crop + review-selected OCR route |
| `image` / `diagram` / `photo` | preserve crop, skip OCR by default |
| `ignore` | skip |
| `unknown_needs_review` | require reviewer selection before OCR |

OCR attempt record:

```json
{
  "attempt_id": "ocr_001",
  "region_id": "r_001",
  "input_bbox": [118, 36, 96, 612],
  "input_type": "modern_japanese_vertical",
  "route": {
    "engine": "tesseract",
    "language": "jpn_vert",
    "psm": 5,
    "preprocess_profile": "none"
  },
  "ocr_output": "...",
  "terms_recovered": ["忍者", "伊賀"],
  "quality_judgment": "partial",
  "selected": false,
  "needs_review": true,
  "created_at": "2026-05-02T00:00:00"
}
```

Keep all attempts unless the researcher explicitly deletes local experiment output.

## Translation Flow

Translation should be request-based, not automatic.

Flow:

```text
selected Japanese region
  -> reviewed OCR output selected
  -> request translation
  -> translation attempt stored
  -> reviewer accepts/edits/rejects translation
```

Translation attempt record:

```json
{
  "translation_id": "tr_001",
  "region_id": "r_001",
  "ocr_attempt_id": "ocr_001",
  "source_text": "...",
  "translation_output": "...",
  "engine": "manual_or_configured_provider",
  "status": "needs_review",
  "reviewed_translation": null,
  "notes": ""
}
```

No translation should mutate OCR text. Translation output is another review artifact.

## Export Flow

Export should be available at any stage.

Initial exports:

- `project_state.json`
- page `*.json`
- `data.json`
- `text.txt`
- `page_1.html`
- `crops/`
- region OCR review JSON/CSV/Markdown where applicable
- macron candidate review summaries where applicable

Future exports:

- DOCX
- PDF
- Markdown
- reviewed translation bundle
- glossary feedback bundle
- accepted/rejected OCR correction reports

Export rules:

- Preserve original OCR and detected regions.
- Include reviewer overrides separately.
- Make incomplete/review-needed status visible.
- Do not claim reviewed output is authoritative unless reviewer explicitly marks it accepted.

## UI Layout

Recommended MVP layout:

```text
+--------------------------------------------------------------+
| Top bar: project path | save | export | status                |
+-------------------+------------------------------------------+
| Page list         | Page viewer                              |
| - page_001        | - scanned image                          |
| - page_002        | - overlay boxes                          |
| - page_003        | - selected bbox handles                  |
+-------------------+----------------------+-------------------+
| Region list       | Selected region panel                    |
| - r_001 reviewed  | type dropdown                            |
| - r_002 ignored   | bbox numeric editor                      |
| - r_003 detected  | notes/status                             |
|                   | run OCR / translation / export buttons   |
+-------------------+------------------------------------------+
```

Core UI behaviors:

- click overlay to select region;
- click region list item to select overlay;
- color boxes by effective type/status;
- show detected bbox as faint outline when reviewed bbox differs;
- show OCR/translation attempts in the selected region panel;
- display clear warnings for `needs_review`, `reading_order_uncertain`, and experimental region types.

## Backend/API Design

Use local Flask endpoints. Do not assume cloud storage or browser filesystem access.

Because browsers cannot directly access arbitrary folders, folder selection should be server-side:

- configured allowed roots;
- folder path entry;
- project import endpoint;
- backend page listing;
- no remote/cloud assumptions.

Suggested endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/workbench/projects` | Create/open project from allowed local path. |
| `GET` | `/workbench/projects/<project_id>` | Get project state. |
| `GET` | `/workbench/projects/<project_id>/pages` | List pages. |
| `GET` | `/workbench/projects/<project_id>/pages/<page_id>` | Get page state. |
| `GET` | `/workbench/projects/<project_id>/pages/<page_id>/image` | Serve source page image. |
| `POST` | `/workbench/projects/<project_id>/pages/<page_id>/recognize-regions` | Run opt-in region recognition. |
| `PATCH` | `/workbench/projects/<project_id>/pages/<page_id>/regions/<region_id>` | Update type, bbox, status, notes. |
| `POST` | `/workbench/projects/<project_id>/pages/<page_id>/regions` | Add manual region. |
| `POST` | `/workbench/projects/<project_id>/pages/<page_id>/regions/<region_id>/ocr` | Run OCR on effective region. |
| `POST` | `/workbench/projects/<project_id>/pages/<page_id>/regions/<region_id>/translate` | Request translation for reviewed OCR text. |
| `POST` | `/workbench/projects/<project_id>/exports` | Export current state. |

Allowed-root checks must happen before serving or processing local paths.

## Storage Model

Use plain files first:

```text
data/runtime/review_projects/<project_id>/
  project_state.json
  pages/
    page_001.json
  crops/
    page_001/
      r_001.png
  ocr/
    page_001/
  translations/
  exports/
```

Storage principles:

- `data/runtime/` remains ignored.
- Project state is readable JSON.
- Large generated crops stay local.
- Local decisions and exports are explicit.
- No schema migration is required for MVP.

If the existing database is used later, it should mirror or index state, not become the only source of truth before the file format stabilizes.

## MVP Scope

MVP should include:

1. Open local folder / page.
2. Display scanned page.
3. Run region recognition.
4. Show overlay boxes.
5. Select region.
6. Change region type.
7. Move/resize bbox.
8. Add/delete/ignore region.
9. Save `project_state.json`.
10. Export current state.

OCR and translation can be Phase 2, but the state model must anticipate them.

## Deferred Features

- Full OCR routing UI.
- Translation provider integration.
- DOCX/PDF export.
- Multi-user collaboration.
- Polished review dashboard.
- Training/fine-tuning loops.
- Automatic Japanese region detection.
- Automatic macron normalization.
- Canonical Japanese model fields.
- Remote/cloud storage.
- Rich annotation history UI.
- Pixel-perfect vector annotation tools.

## Safety / Non-Destructive Rules

- Never silently change runtime defaults.
- Never treat detector output as truth.
- Never overwrite detected bbox/type with reviewed bbox/type.
- Never mutate original OCR text.
- Never apply macron normalization automatically.
- Never auto-translate without an explicit request.
- Never commit `data/runtime/`, generated crops, private images, or local review projects.
- Always let the researcher inspect source image, crop, OCR text, selected route, and review status.
- Always keep an export path even when review is incomplete.

## Relevant Existing Code

- `src/martial_arts_ocr/app/flask_app.py`: current Flask app, upload/status routes, app factory, dependency attachment.
- `src/martial_arts_ocr/app/routes.py`: placeholder for future route migration.
- `src/martial_arts_ocr/pipeline/orchestrator.py`: canonical processing and artifact writing path.
- `src/martial_arts_ocr/pipeline/document_models.py`: `DocumentResult`, `PageResult`, `TextRegion`, `ImageRegion`, line/word aliases.
- `src/martial_arts_ocr/pipeline/extraction_service.py`: opt-in image-region extraction enrichment.
- `src/martial_arts_ocr/reconstruction/page_reconstructor.py`: current debug/review HTML generation.
- `experiments/run_review_mode_extraction.py`: no-OCR review-mode extraction runner.
- `experiments/review_japanese_region_ocr.py`: experiment-only Japanese region OCR routing and review exports.
- `experiments/review_macron_candidates.py`: review-only macron candidate workflow.
- `docs/review-mode-extraction-guide.md`: review-mode extraction usage and output inspection.
- `docs/extraction-architecture-freeze-2026-04-28.md`: extraction boundary.
- `docs/ocr-output-state-2026-04-28.md`: OCR/text state.
- `docs/document-output-state-2026-04-28.md`: artifact contract.
- `docs/japanese-ocr-experimental-state-2026-04-28.md`: Japanese OCR routing state.
- `docs/macron-candidate-workflow-state-2026-04-28.md`: macron review workflow state.

## Recommended Implementation Phases

### Phase 1: Local Page Viewer and Editable Overlay

- Add workbench routes behind local-only Flask UI.
- Open page/folder from allowed local roots.
- Display page image.
- Run region recognition.
- Draw overlay boxes.
- Select region.
- Change region type.
- Move/resize bbox.
- Add/delete/ignore region.
- Save file-backed state.
- Export `project_state.json`.

### Phase 2: Region OCR Review

- Run OCR on selected effective regions.
- Use effective type and effective bbox.
- Store OCR attempts.
- Show OCR output and route metadata.
- Export region OCR JSON/CSV/Markdown.

### Phase 3: Japanese Region and Macron Review

- Add UI hooks for Japanese OCR profiles.
- Show expected/recovered/missing terms when present.
- Run macron candidate scan on selected OCR text.
- Store accept/reject/defer/edit decisions locally.

### Phase 4: Translation Review

- Request translation for selected reviewed Japanese OCR output.
- Store translation attempts.
- Let researcher edit/accept/reject translation.
- Export reviewed translation bundle.

### Phase 5: Better Export and Review Convenience

- Markdown/DOCX/PDF exports.
- Source-image crop bundles.
- Glossary feedback export.
- Better duplicate grouping and review queues.

## First Implementation Pass

Build the local page viewer and editable overlay prototype.

Target behavior:

```text
folder/page list
page image
region overlays
select region
drag/resize bbox
type dropdown
save project_state.json
export current state
```

Do not implement OCR in the first pass unless the region editing state is already stable. BBox correction is the foundation for every downstream OCR, translation, and export step.
