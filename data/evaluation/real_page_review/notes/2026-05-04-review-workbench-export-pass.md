# Review Workbench Export Pass

## Purpose

Add a small reviewed-page export artifact after the single-page workbench validation proved that orientation, region review, selected-region OCR, OCR variants, reviewed text edits, and persistence work on a real page.

This pass intentionally does not add PDF/DOCX export, translation, OCR-all, new OCR behavior, detector changes, orientation changes, or database schema changes.

## Export Artifacts

The page-level export writes a timestamped local artifact under:

```text
data/runtime/review_projects/<project_id>/exports/<timestamp>/
```

Files:

```text
project_state_snapshot.json
page_<page_id>_review.json
page_<page_id>_review.md
page_<page_id>_text.txt
crops/
  region_<region_id>.png
```

Generated export directories remain ignored runtime output and should not be committed.

## Reviewed Text Behavior

For text-like regions, the plain text export uses:

```text
reviewed_text if present
else cleaned_text
else raw OCR text
else empty
```

Rejected OCR attempts do not contribute text to the plain text export.

The JSON and Markdown review artifacts preserve raw OCR and reviewed text separately:

```text
raw_text
cleaned_text
reviewed_text
review_status
source_text_mutated=false
```

This keeps the `[Question.]` style correction auditable: export can use reviewed text without pretending OCR produced it.

## Crop Behavior

Region crops are written for non-ignored regions using:

```text
effective-oriented page image
effective_bbox
```

Ignored regions remain in JSON metadata but are skipped for text and crop export in this MVP.

## UI Behavior

The workbench now includes an `Export Page` button. The UI shows the generated export directory and a compact summary after export completion.

## Tests

Added generated-image tests for:

- JSON/Markdown/text artifact creation;
- reviewed text preference;
- raw OCR preservation;
- `source_text_mutated=false` preservation;
- text-region ordering by y/x;
- ignored-region omission from text/crops;
- effective-oriented image crop sizing;
- export without OCR attempts.

## Limitations

- No ZIP/download packaging.
- No PDF/DOCX export.
- No full-project export.
- No OCR-all-reviewed-regions action.
- No polished publication layout.

## Recommended Next Use

Exercise this export on `IMG_3312.jpg` and at least one additional reviewed page. The next branch should be chosen from export friction: either improve export ergonomics or add OCR-all-reviewed-text-regions if repeated selected-region OCR becomes the main bottleneck.
