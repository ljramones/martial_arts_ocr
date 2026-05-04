# Single-Page Workbench End-to-End Review

## Purpose

Validate the local research review workbench on a real scanned page from page load through orientation, region review, selected-region OCR, OCR correction, and persistence.

This pass used the existing workbench API/state paths in a headless operator run. No workbench code, OCR behavior, detector behavior, export behavior, runtime defaults, private images, or generated runtime project state were committed.

## Target Page

- Path: `/Users/larrymitchell/ML/martial_arts_ocr/data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg`
- Source orientation: current orientation detected as `270` degrees
- Expected correction: `90` degrees clockwise
- Project ID: `img3312_e2e_20260504`
- Runtime project state path: `data/runtime/review_projects/img3312_e2e_20260504/project_state.json`

## Workflow Performed

| Step | Result | Notes |
|---|---|---|
| Create/load project | Passed | Loaded source folder and selected `IMG_3312.jpg` as `page_025`. |
| Run Orient | Passed | NN orientation detected current `270`; workbench applied correction `90`. |
| Confirm upright effective page | Passed by state | Effective dimensions changed from `1600 x 1200` to `1200 x 1600`. |
| Run Recognition | Partial | Recognition imported one machine `diagram` region and rejected one broad candidate. |
| Draw/review regions | Passed | Added one English text block, three diagram regions, and one caption/label region manually. |
| Run selected-region OCR | Passed | Default OCR attempt stored as `ocr_001`. |
| Run OCR variants | Passed | Seven variant attempts stored; best-ranked attempt was `ocr_008`, `eng / PSM 6 / upscale_2x`. |
| Edit reviewed text | Passed | `ocr_008` marked `edited`; reviewed text was saved separately. |
| Reload project/page | Passed | Orientation, regions, OCR attempts, reviewed text, and `source_text_mutated=false` persisted. |

## Orientation

- Detected current orientation: `270`
- Correction applied: `90` clockwise
- Manual override used: no
- Result: passed; effective page dimensions are `1200 x 1600`
- Confidence: `0.4679487347602844`
- Source: `orientation_cnn`, model used `convnext`

## Region Review

| Region | Type | Source | BBox | Reviewed? | Notes |
|---|---|---|---|---|---|
| `det_001` | `diagram` | `machine_detection` | `[463, 853, 340, 455]` | unreviewed | Recognition found the central figure region. |
| `r_001` | `english_text` | `reviewer_manual` | `[180, 70, 940, 180]` | manually added | Opening English question block containing the known OCR issue. |
| `r_002` | `diagram` | `reviewer_manual` | `[95, 835, 330, 455]` | manually added | Left figure/image region. |
| `r_003` | `diagram` | `reviewer_manual` | `[463, 853, 340, 455]` | manually added | Central figure/image region, comparable to machine detection. |
| `r_004` | `diagram` | `reviewer_manual` | `[805, 835, 300, 455]` | manually added | Right figure/image region. |
| `r_005` | `caption_label` | `reviewer_manual` | `[210, 1368, 610, 70]` | manually added | Caption/label strip below figures. |

## OCR Review

| Region | Route | Variant Used | Raw OCR Issue | Reviewed Text Change | Status |
|---|---|---|---|---|---|
| `r_001` | `eng / PSM 6` | default | Read the first line as `(Question. ]`; recovered `Aaaa yes`. | Edited reviewed text to start with `[Question.]` and `Aaaa yes. I'm glad you're frank.` | default attempt `unreviewed` |
| `r_001` | `eng / PSM 6 / upscale_2x` | `ocr_008` | Best variant still read the first line as `(Question. ]`; recovered `Aaaa yes`. | Saved corrected opening line in `reviewed_text`; raw OCR preserved. | `edited` |

Default OCR preview:

```text
(Question. ]

Aaaa yes. I'm glad you're frank. Okay. You state your
-ion so everybody can be brought up to date.
```

Best variant preview:

```text
(Question. ]

Aaaa yes. I'm glad you're frank. Okay. You state your
-jon so everybody can be brought up to date.
```

Reviewed text starts with:

```text
[Question.]
Aaaa yes. I'm glad you're frank.
```

## Persistence Check

After reload, confirm:

- orientation persisted: yes
- regions persisted: yes, six regions total
- reviewed bboxes persisted: yes
- OCR attempts persisted: yes, eight attempts total
- reviewed_text persisted: yes, on `ocr_008`
- source_text_mutated=false: yes

## Operator Findings

### What Worked

- Orientation flow worked on the known rotated page.
- Effective-oriented OCR used the corrected page geometry.
- Manual region creation and region state persistence worked.
- Selected-region OCR and variant OCR both stored attempts.
- The OCR review/edit layer solved the real `[Question.]` problem in the correct way: by storing reviewer correction separately from raw OCR.
- Reload confirmed the key audit invariant: raw OCR is preserved and `source_text_mutated=false`.

### What Was Tedious

- A complete page still requires several manual region boxes.
- Recognition found the central figure, but side figures still required manual boxes.
- The source folder contains many images, so selecting a single target page from a full-folder project is heavier than a one-page import flow.

### What Was Confusing

- Recognition and manual region boxes can overlap; the state preserves both, but the operator must decide which region is the reviewed one.
- OCR variants create many attempts, but the UI does not yet provide a compact attempt list for comparing or selecting attempts.
- The reviewed text editor can correct the opening line, but there is not yet an export view that shows how reviewed text will be used downstream.

### Missing Workflow Support

- Export reviewed page state is now the main missing workflow.
- OCR attempt list/comparison UI would make variant review easier.
- A one-page import flow would reduce friction for targeted validation pages.

## Next Implementation Recommendation

Choose one primary next branch:

- [x] Export reviewed page state
- [ ] OCR all reviewed text regions
- [ ] Improve OCR attempt/review UI ergonomics
- [ ] Improve region inventory/navigation
- [ ] Add keyboard shortcuts / faster region editing
- [ ] Improve page reconstruction/export

## Recommendation Rationale

The workbench can now create reviewed regions and reviewed OCR text for a real page. The next bottleneck is not more OCR logic; it is getting the reviewed state out in a useful, inspectable artifact. A reviewed-page export should prefer `reviewed_text` when present, preserve raw OCR attempts for audit, include reviewed region geometry/type, and clearly mark incomplete/unreviewed regions.
