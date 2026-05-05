# Workbench OCR-All Batch Review

## Purpose

Validate the OCR-all reviewed text regions workflow on a small real batch.

This pass used an ignored runtime copy of the previous small-batch project so that OCR-all could create new attempts without destroying prior reviewed evidence. The known edited `IMG_3312` opening text was preserved to validate skip/protection behavior.

## Batch Summary

| Page | Page Type | Text Regions Eligible | OCR Attempts Created | Skipped | OCR Errors | Corrections Needed | Export Result | Main Friction |
|---|---|---:|---:|---:|---:|---:|---|---|
| `page_025` / `IMG_3312.jpg` | mixed text + figures + caption | 2 | 1 | 6 | 0 | 1 new caption attempt needs review; existing edited text preserved | pass | OCR-all correctly skipped protected text and non-text regions. |
| `page_026` / `IMG_3313.jpg` | mostly English text | 1 | 1 | 0 | 0 | 1 | pass | Broad region OCR remains noisy. |
| `page_027` / `IMG_3314.jpg` | noisy English text | 1 | 1 | 0 | 0 | 1 | pass | Broad/noisy OCR needs review. |
| `page_028` / `IMG_3315.jpg` | English text / awkward scan | 1 | 1 | 0 | 0 | 1 | pass | OCR was created automatically, but review remains manual. |
| `page_029` / `IMG_3316.jpg` | English text with Japanese/history terms | 1 | 1 | 0 | 0 | 1 | pass | Draft OCR useful, still needs human review. |
| `page_030` / `IMG_3317.jpg` | awkward/noisy page | 1 | 1 | 0 | 0 | 1 | pass | OCR quality poor; tighter regions or variants likely needed. |

## Pages Reviewed

### `page_025` / `IMG_3312.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg`
- Page type: mixed text, figure row, caption/label strip
- Orientation: detected current 270 degrees; correction 90 degrees clockwise
- Regions reviewed:
  - text-like regions: `r_001` (`english_text`), `r_005` (`caption_label`)
  - image/diagram regions: `det_001`, `r_002`, `r_003`, `r_004`
  - ignored/unknown regions: `r_006`
- OCR-all result:
  - eligible regions: 2
  - attempts created: 1 (`ocr_009` for `r_005`)
  - skipped regions: 6
  - skip reasons: `reviewed_ocr_exists` for `r_001`; `non_text_region` for diagram regions; `ignored_region` for `r_006`
  - OCR errors: 0
- OCR review:
  - good attempts: existing `r_001` reviewed text remained good enough for validation
  - attempts needing correction: generated caption OCR is noisy and needs review
  - reviewed_text edits: existing `[Question.]` correction preserved; no new edit made in this pass
  - accepted/edited/rejected counts: 1 edited existing attempt, 1 unreviewed new attempt
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: yes, exported text and HTML start with `[Question.]`
  - raw OCR preserved: yes, raw OCR remains visible in review artifacts
- Operator notes:
  - what worked: protected reviewed text was skipped; caption OCR was created; image/ignored regions were skipped
  - what was tedious: reviewing generated attempts still requires per-attempt navigation
  - what was confusing: none in the OCR-all summary
  - what was missing: a page-level OCR attempt review queue

### `page_026` / `IMG_3313.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3313.jpg`
- Page type: mostly English text
- Orientation: detected current 270 degrees; correction 90 degrees clockwise
- Regions reviewed:
  - text-like regions: 1 broad `english_text` region
  - image/diagram regions: 0
  - ignored/unknown regions: 0
- OCR-all result:
  - eligible regions: 1
  - attempts created: 1 (`ocr_001`)
  - skipped regions: 0
  - skip reasons: none
  - OCR errors: 0
- OCR review:
  - good attempts: usable only as rough draft text
  - attempts needing correction: 1
  - reviewed_text edits: none in this pass
  - accepted/edited/rejected counts: 1 unreviewed
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: no reviewed text present, so OCR text was used
  - raw OCR preserved: yes
- Operator notes:
  - what worked: OCR-all removed the separate OCR click
  - what was tedious: reviewing a long noisy attempt still requires manual navigation/editing
  - what was confusing: none
  - what was missing: faster attempt review and tighter text-block workflow

### `page_027` / `IMG_3314.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3314.jpg`
- Page type: noisy English text
- Orientation: detected current 270 degrees; correction 90 degrees clockwise
- Regions reviewed:
  - text-like regions: 1 broad `english_text` region
  - image/diagram regions: 0
  - ignored/unknown regions: 0
- OCR-all result:
  - eligible regions: 1
  - attempts created: 1 (`ocr_001`)
  - skipped regions: 0
  - skip reasons: none
  - OCR errors: 0
- OCR review:
  - good attempts: rough draft only
  - attempts needing correction: 1
  - reviewed_text edits: none in this pass
  - accepted/edited/rejected counts: 1 unreviewed
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: no reviewed text present
  - raw OCR preserved: yes
- Operator notes:
  - what worked: batch OCR created the attempt without selected-region clicking
  - what was tedious: OCR quality still requires human cleanup
  - what was confusing: none
  - what was missing: review queue and possibly variant batching later

### `page_028` / `IMG_3315.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3315.jpg`
- Page type: English text / awkward scan
- Orientation: detected current 270 degrees; correction 90 degrees clockwise
- Regions reviewed:
  - text-like regions: 1 broad `english_text` region
  - image/diagram regions: 0
  - ignored/unknown regions: 0
- OCR-all result:
  - eligible regions: 1
  - attempts created: 1 (`ocr_001`)
  - skipped regions: 0
  - skip reasons: none
  - OCR errors: 0
- OCR review:
  - good attempts: partial draft
  - attempts needing correction: 1
  - reviewed_text edits: none in this pass
  - accepted/edited/rejected counts: 1 unreviewed
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: no reviewed text present
  - raw OCR preserved: yes
- Operator notes:
  - what worked: OCR-all converted reviewed region state into OCR attempts
  - what was tedious: manual review of long noisy OCR remains
  - what was confusing: none
  - what was missing: next/previous attempt controls

### `page_029` / `IMG_3316.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3316.jpg`
- Page type: English text with Japanese/history vocabulary
- Orientation: detected current 270 degrees; correction 90 degrees clockwise
- Regions reviewed:
  - text-like regions: 1 broad `english_text` region
  - image/diagram regions: 0
  - ignored/unknown regions: 0
- OCR-all result:
  - eligible regions: 1
  - attempts created: 1 (`ocr_001`)
  - skipped regions: 0
  - skip reasons: none
  - OCR errors: 0
- OCR review:
  - good attempts: usable draft, still needs review
  - attempts needing correction: 1
  - reviewed_text edits: none in this pass
  - accepted/edited/rejected counts: 1 unreviewed
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: no reviewed text present
  - raw OCR preserved: yes
- Operator notes:
  - what worked: batch OCR and export held up
  - what was tedious: specialized vocabulary still needs human review
  - what was confusing: none
  - what was missing: a way to step through unreviewed OCR attempts quickly

### `page_030` / `IMG_3317.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3317.jpg`
- Page type: awkward/noisy page
- Orientation: detected current 0 degrees; correction 0 degrees
- Regions reviewed:
  - text-like regions: 1 broad `english_text` region
  - image/diagram regions: 0
  - ignored/unknown regions: 0
- OCR-all result:
  - eligible regions: 1
  - attempts created: 1 (`ocr_001`)
  - skipped regions: 0
  - skip reasons: none
  - OCR errors: 0
- OCR review:
  - good attempts: poor OCR, draft only
  - attempts needing correction: 1
  - reviewed_text edits: none in this pass
  - accepted/edited/rejected counts: 1 unreviewed
- Export:
  - review_bundle: pass
  - HTML: pass
  - reviewed_text preferred: no reviewed text present
  - raw OCR preserved: yes
- Operator notes:
  - what worked: OCR-all did not fail on the noisy page
  - what was tedious: output quality means significant review burden
  - what was confusing: none
  - what was missing: tighter region workflow or variant batching may help later

## Cross-Page Findings

- OCR-all successfully created 6 new OCR attempts across 6 pages.
- The known edited `IMG_3312` text region was skipped rather than overwritten.
- Image, diagram, ignored, and unknown/non-text regions were skipped correctly.
- No per-region OCR errors occurred.
- Export v2 still worked after OCR-all, producing review bundle and HTML for the selected 6 pages.
- The biggest workflow burden moved from "run OCR for each region" to "review each generated OCR attempt."

## Reviewed Text Protection

Confirm:

- regions with accepted/edited OCR were skipped: yes. `page_025` / `r_001` was skipped with `reviewed_ocr_exists`.
- existing reviewed_text was preserved: yes. Exported text and HTML still start with `[Question.]`.
- source_text_mutated=false remained true: yes for preserved and newly created OCR attempts, and in the export manifest.

## OCR-All Usefulness

- Did OCR-all reduce manual clicking? Yes. Six OCR attempts were created through six page-level calls instead of selecting each text region and pressing `Run OCR`.
- Were skip reasons understandable? Yes. `reviewed_ocr_exists`, `non_text_region`, and `ignored_region` matched expected state.
- Did any region get OCRed that should not have? No. The generated attempts were for `caption_label` or `english_text` regions.
- Did any eligible region fail to OCR unexpectedly? No.

## Export After OCR-All

- reviewed_text preferred: yes. `page_025_text.txt` and HTML used the corrected `[Question.]` text.
- raw OCR preserved: yes. Raw OCR remained present in review Markdown/JSON and HTML details.
- ignored regions handled correctly: ignored region was retained in model/review state but skipped from text/crop output as expected.
- HTML useful: yes. The HTML contained the corrected text, new OCR outputs, page sections, source metadata, and crop assets.

## Next Biggest Workflow Friction

The next bottleneck is OCR attempt review, not OCR execution. OCR-all creates attempts efficiently, but the user still needs a faster way to move through unreviewed attempts, compare raw/cleaned/reviewed text, mark accepted/edited/rejected, and jump to the associated region/page.

## Recommendation

Choose one primary next branch:

- [x] OCR attempt navigation/review queue
- [ ] DOCX export
- [ ] page-to-page workflow ergonomics
- [ ] HTML/export polish
- [ ] more batch review

## Rationale

OCR-all solved the previous per-region OCR-clicking bottleneck. Export remains functional and HTML is usable enough. The remaining operator cost is reviewing generated attempts one at a time through the selected-region panel. A simple OCR review queue should list unreviewed attempts across the current page or selected pages, support next/previous navigation, show raw OCR beside reviewed text, and keep the existing non-destructive review controls.
