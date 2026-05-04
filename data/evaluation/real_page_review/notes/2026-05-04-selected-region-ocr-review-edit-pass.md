# Selected-Region OCR Review/Edit Pass

## Purpose

Add the review layer needed after selected-region OCR. Old typewritten scans can be legible to a human while Tesseract still misreads short dirty lines, so the workbench must preserve raw OCR output and store reviewer corrections separately.

## Background

The selected-region OCR variants pass improved the real `IMG_3312.jpg` case enough to recover `Aaaa yes.`, but no variant reliably recovered the visible first line `[Question.]`. That failure is an OCR recognition limitation, not an orientation, bbox, or UI state issue.

## Implementation

- Added OCR attempt review state updates.
- Added a review API endpoint for OCR attempts.
- Added selected-region OCR review controls:
  - `Accept OCR`
  - `Save Reviewed Text`
  - `Reject OCR`
- Added a reviewed text editor beside the read-only OCR output.

## State Model

OCR attempts preserve raw and reviewed text separately:

```json
{
  "attempt_id": "ocr_001",
  "text": "Le opmgageet",
  "cleaned_text": "Le opmgageet",
  "reviewed_text": "[Question.]\nAaaa yes.",
  "review_status": "edited",
  "source_text_mutated": false
}
```

`text` and `cleaned_text` remain OCR evidence. Reviewer corrections are stored in `reviewed_text`; they do not mutate source OCR text, source images, region bboxes, or canonical document fields.

## Review Status Values

- `unreviewed`: OCR attempt exists but has not been reviewed.
- `accepted`: reviewer accepted the attempt text.
- `edited`: reviewer saved corrected text.
- `rejected`: reviewer rejected the attempt.

## Non-Goals

- No new OCR engines.
- No additional OCR variants.
- No OCR-all-regions action.
- No export behavior change.
- No automatic text correction.
- No canonical field promotion.

## Recommended Next Use

Use the review text editor for cases where OCR is close but misses locally obvious text, such as dirty typewritten first lines. Later export code can prefer `reviewed_text` when present while still carrying raw OCR evidence for audit.
