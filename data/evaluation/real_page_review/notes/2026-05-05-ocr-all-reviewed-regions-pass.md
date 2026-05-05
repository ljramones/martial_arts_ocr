# OCR All Reviewed Regions Pass

## Purpose

The small-batch workbench review showed that selected-region OCR works, but clicking OCR region by region is too slow across multiple pages. This pass adds a conservative page-level batch action for OCRing reviewed text regions after the reviewer has already defined region type and bbox.

## Added Behavior

- Added a page-level `OCR Reviewed Text Regions` action in the workbench UI.
- Added a page API endpoint:
  `POST /api/review/projects/<project_id>/pages/<page_id>/ocr-reviewed-regions`
- Reused the selected-region OCR service and attempt storage path.
- Stored each result as a normal page-level OCR attempt and linked it back to the region through `ocr_attempt_ids` and `last_ocr_attempt_id`.
- Returned a summary of attempts created, skipped regions, and per-region errors.

## Eligible Region Types

Only reviewed/effective text-like regions are eligible:

```text
english_text
romanized_japanese_text
modern_japanese_horizontal
modern_japanese_vertical
mixed_english_japanese
caption_label
```

## Skip Behavior

The batch action skips:

- image, diagram, and photo regions;
- ignored regions;
- `unknown_needs_review` regions;
- regions without an `effective_bbox`;
- regions with accepted or edited OCR attempts;
- regions with existing `reviewed_text`.

This keeps reviewer corrections authoritative and avoids accidental churn in already-reviewed OCR.

## Reviewed Text Preservation

The action does not overwrite `reviewed_text`. Raw OCR text, cleaned OCR text, and reviewed text remain separate. OCR attempts continue to carry:

```text
source_text_mutated=false
```

## Error Handling

Per-region OCR errors are returned in the page-level summary and do not stop remaining eligible regions from being attempted.

## Deferred

- OCR all regions across a whole project;
- force rerun behavior;
- OCR image/diagram/photo regions;
- translation;
- changes to export semantics;
- changes to OCR, detection, or orientation defaults.

## Verification

Automated tests cover eligible text regions, skipped non-text/ignored/unknown regions, preservation of reviewed text, per-region error handling, and route/UI exposure. Full verification was run as part of this pass.
