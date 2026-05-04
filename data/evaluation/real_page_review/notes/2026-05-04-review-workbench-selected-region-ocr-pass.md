# Review Workbench Selected-Region OCR Pass

## Purpose

Add the first OCR slice to the local research review workbench while preserving the workflow rule:

```text
review region type and bbox first
then run OCR on that selected region
```

## Scope

This pass adds selected-region OCR only. It does not add OCR-all, translation, export polish, canonical Japanese fields, or automatic text correction.

## Behavior

For a selected region:

1. The workbench uses the page's effective-oriented image.
2. The selected region's `effective_bbox` defines the crop.
3. The selected region's `effective_type` chooses the OCR route.
4. The OCR result is stored as a page-level `ocr_attempt`.
5. The selected region records `ocr_attempt_ids` and `last_ocr_attempt_id`.
6. The latest output is shown in the right panel.

## Initial Routing

| Region Type | Language | PSM | Preprocess | Notes |
|---|---|---:|---|---|
| `english_text` | `eng` | 6 | none | Default text block route. |
| `romanized_japanese_text` | `eng` | 6 | none | Macron correction remains review-candidate based. |
| `caption_label` | `eng` | 7 | none | Short label/caption route. |
| `modern_japanese_horizontal` | `jpn` | 6 | `upscale_2x` | Matches region OCR evidence. |
| `modern_japanese_vertical` | `jpn_vert` | 5 | `upscale_2x` | Matches vertical Japanese evidence. |
| `mixed_english_japanese` | `eng+jpn` | 6 | none | Review route for blended text. |

Non-text/image region types are skipped by this slice.

## State Shape

OCR attempts are stored in `project_state.json`:

```json
{
  "attempt_id": "ocr_001",
  "region_id": "r_001",
  "region_type": "english_text",
  "bbox": [10, 12, 80, 24],
  "orientation_degrees": 0,
  "text": "...",
  "cleaned_text": "...",
  "confidence": 0.88,
  "route": {
    "engine": "tesseract",
    "language": "eng",
    "psm": 6,
    "preprocess_profile": "none"
  },
  "status": "ok",
  "source_text_mutated": false
}
```

## Non-Goals

- No OCR-all-regions button.
- No translation.
- No PDF/DOCX export.
- No automatic macron correction.
- No canonical Japanese field promotion.
- No runtime OCR default changes.

## Decision

Selected-region OCR is now the next workbench layer after manual region review. OCR attempts remain review artifacts and can be rerun or superseded without mutating source text or canonical fields.
