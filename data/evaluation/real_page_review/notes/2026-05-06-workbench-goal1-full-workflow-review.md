# Workbench Goal 1 Full Workflow Review

## Purpose

Evaluate whether the current local research workbench is useful enough for Goal 1: English / romanized Japanese scanned-page reconstruction with reviewed text, images, and exportable output.

This pass used the existing ignored runtime project `ocr_all_batch_20260505` and fresh ignored exports generated with the current code. It validates the full loop from reviewed page state through OCR review/export artifacts, not a new editorial transcription of every page.

## Batch Summary

| Page | Page Type | Orientation | Region Review | OCR-All | OCR Queue | HTML | DOCX | Main Friction |
|---|---|---|---|---|---|---|---|---|
| `page_025` / `IMG_3312.jpg` | mixed text + figures + caption | detected 270, corrected 90 | 1 machine region kept, 6 reviewer regions | 1 new caption attempt, protected edited text skipped | useful; 1 edited, 1 pending | pass | pass with appendix off/on | Manual figure/caption setup still takes time; one pending caption attempt remains. |
| `page_026` / `IMG_3313.jpg` | mostly English text | detected 270, corrected 90 | 1 manual text region | 1 attempt | pending | pass | pass | Broad-region OCR is noisy and needs review. |
| `page_027` / `IMG_3314.jpg` | noisy English text | detected 270, corrected 90 | 1 manual text region | 1 attempt | pending | pass | pass | OCR quality depends heavily on tighter crops. |
| `page_028` / `IMG_3315.jpg` | English text / awkward scan | detected 270, corrected 90 | 1 manual text region | 1 attempt | pending | pass | pass | OCR cleanup remains manual. |
| `page_029` / `IMG_3316.jpg` | English text with Japanese/history vocabulary | detected 270, corrected 90 | 1 manual text region | 1 attempt | pending | pass | pass | Specialized vocabulary still needs human review. |
| `page_030` / `IMG_3317.jpg` | awkward/noisy page | detected 0, corrected 0 | 1 manual text region | 1 attempt | pending | pass | pass | OCR is poor; page likely needs tighter regions or variants. |

## Pages Reviewed

### `page_025` / `IMG_3312.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3312.jpg`
- Page type: mixed text, figure row, caption/label strip
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions useful: 1 central diagram candidate retained
  - manual regions drawn: 6 reviewer regions
  - regions adjusted: manually drawn/reviewed during earlier batch passes
  - image/diagram regions: 4
  - text/caption regions: 2
  - ignored/unknown regions: 1 ignored region
- OCR:
  - OCR-all attempts created: 1 new caption attempt in the OCR-all pass
  - skipped regions: edited text region, image/diagram regions, ignored region
  - variants used: earlier selected-region variant pass for the opening text
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1 caption attempt remains unreviewed
  - accepted: 0
  - edited: 1 existing opening text attempt
  - rejected: 0
  - queue usefulness: useful for locating the remaining unreviewed OCR attempt; queue does not solve OCR quality by itself
- reviewed_text:
  - corrections needed: yes, the known opening line was corrected to `[Question.]` / `Aaaa yes.`
  - correction effort: low for the specific line, but requires human reading
- Export:
  - review_bundle: pass; JSON/Markdown/text and crops present
  - HTML: pass; corrected reviewed text and raw OCR details visible
  - DOCX raw OCR appendix off: pass; corrected text appears, full raw OCR omitted from main body
  - DOCX raw OCR appendix on, if tested: pass; appendix appears with raw OCR evidence
  - crops: 4 image/diagram crops embedded/referenced
- Operator notes:
  - what worked: orientation, reviewer regions, OCR-all skip protection, reviewed_text, HTML, DOCX, crops
  - what was tedious: manual side-figure/caption setup and cleaning unreviewed OCR
  - what was confusing: page still carries warning/needs_review state for unreviewed regions, which is honest but visually noisy
  - what was missing: faster region setup for text/caption blocks would help

### `page_026` / `IMG_3313.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3313.jpg`
- Page type: mostly English text
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions useful: 0
  - manual regions drawn: 1 broad English text region
  - regions adjusted: not in this validation pass
  - image/diagram regions: 0
  - text/caption regions: 1
  - ignored/unknown regions: 0
- OCR:
  - OCR-all attempts created: 1
  - skipped regions: 0
  - variants used: no
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1
  - accepted: 0
  - edited: 0
  - rejected: 0
  - queue usefulness: useful for surfacing that OCR still needs review
- reviewed_text:
  - corrections needed: yes, broad OCR is noisy
  - correction effort: moderate to high
- Export:
  - review_bundle: pass
  - HTML: pass
  - DOCX raw OCR appendix off: pass
  - DOCX raw OCR appendix on, if tested: pass at document level
  - crops: text crop present in review bundle; no reader-facing image crop expected
- Operator notes:
  - what worked: orientation, OCR-all, export
  - what was tedious: broad text-region OCR review
  - what was confusing: OCR quality looks worse when region is too broad
  - what was missing: text-block segmentation or faster manual split workflow

### `page_027` / `IMG_3314.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3314.jpg`
- Page type: noisy English text
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions useful: 0
  - manual regions drawn: 1 broad English text region
  - regions adjusted: not in this validation pass
  - image/diagram regions: 0
  - text/caption regions: 1
  - ignored/unknown regions: 0
- OCR:
  - OCR-all attempts created: 1
  - skipped regions: 0
  - variants used: no
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1
  - accepted: 0
  - edited: 0
  - rejected: 0
  - queue usefulness: useful for review navigation, but the attempt still needs human correction
- reviewed_text:
  - corrections needed: yes
  - correction effort: high because the crop/noise produces rough OCR
- Export:
  - review_bundle: pass
  - HTML: pass
  - DOCX raw OCR appendix off: pass
  - DOCX raw OCR appendix on, if tested: pass at document level
  - crops: review bundle text crop present
- Operator notes:
  - what worked: full pipeline completed
  - what was tedious: OCR cleanup
  - what was confusing: none beyond expected OCR quality limitations
  - what was missing: better text region setup and variant use across pages

### `page_028` / `IMG_3315.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3315.jpg`
- Page type: English text / awkward scan
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions useful: 0
  - manual regions drawn: 1 broad English text region
  - regions adjusted: not in this validation pass
  - image/diagram regions: 0
  - text/caption regions: 1
  - ignored/unknown regions: 0
- OCR:
  - OCR-all attempts created: 1
  - skipped regions: 0
  - variants used: no
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1
  - accepted: 0
  - edited: 0
  - rejected: 0
  - queue usefulness: useful for finding the next attempt
- reviewed_text:
  - corrections needed: yes
  - correction effort: moderate
- Export:
  - review_bundle: pass
  - HTML: pass
  - DOCX raw OCR appendix off: pass
  - DOCX raw OCR appendix on, if tested: pass at document level
  - crops: review bundle text crop present
- Operator notes:
  - what worked: page-to-export flow
  - what was tedious: reviewing long uncorrected OCR
  - what was confusing: none
  - what was missing: faster page-to-page review ergonomics

### `page_029` / `IMG_3316.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3316.jpg`
- Page type: English text with Japanese/history vocabulary
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions useful: 0
  - manual regions drawn: 1 broad English text region
  - regions adjusted: not in this validation pass
  - image/diagram regions: 0
  - text/caption regions: 1
  - ignored/unknown regions: 0
- OCR:
  - OCR-all attempts created: 1
  - skipped regions: 0
  - variants used: no
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1
  - accepted: 0
  - edited: 0
  - rejected: 0
  - queue usefulness: useful, but terminology still requires human review
- reviewed_text:
  - corrections needed: yes
  - correction effort: moderate
- Export:
  - review_bundle: pass
  - HTML: pass
  - DOCX raw OCR appendix off: pass
  - DOCX raw OCR appendix on, if tested: pass at document level
  - crops: review bundle text crop present
- Operator notes:
  - what worked: draft document output
  - what was tedious: specialized vocabulary cleanup
  - what was confusing: none
  - what was missing: romanized Japanese/macron review support in the workbench may be useful later

### `page_030` / `IMG_3317.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3317.jpg`
- Page type: awkward/noisy page
- Orientation:
  - detected: 0
  - correction: 0
  - manual override: no
  - result: page served without correction
- Region review:
  - machine regions useful: 0
  - manual regions drawn: 1 broad English text region
  - regions adjusted: not in this validation pass
  - image/diagram regions: 0
  - text/caption regions: 1
  - ignored/unknown regions: 0
- OCR:
  - OCR-all attempts created: 1
  - skipped regions: 0
  - variants used: no
  - OCR errors: 0
- OCR review queue:
  - pending attempts: 1
  - accepted: 0
  - edited: 0
  - rejected: 0
  - queue usefulness: useful for finding the pending attempt; output quality remains the issue
- reviewed_text:
  - corrections needed: yes
  - correction effort: high
- Export:
  - review_bundle: pass
  - HTML: pass
  - DOCX raw OCR appendix off: pass
  - DOCX raw OCR appendix on, if tested: pass at document level
  - crops: review bundle text crop present
- Operator notes:
  - what worked: pipeline did not fail on noisy page
  - what was tedious: OCR correction burden
  - what was confusing: none
  - what was missing: tighter regions or variants before review may be necessary

## Cross-Page Findings

- The current workbench is useful enough for Goal 1 prototype use.
- Orientation worked for this batch: five pages required 90-degree correction and one page required no correction.
- Manual region drawing is still essential; recognition helped on the mixed image page but did not remove manual text-region work.
- OCR-all removed the prior per-region OCR execution bottleneck.
- The OCR review queue makes pending attempts visible and navigable, but full correction remains human work.
- HTML and DOCX exports now have clear roles and are useful enough for reviewed research artifacts.
- The main remaining friction is not export; it is efficient page/region setup and OCR cleanup on noisy broad text regions.

## Region Review Speed and Friction

Region review is acceptable for small batches, especially because direct drawing works. It is not yet fast for long folders because text-heavy pages still need manual text block drawing and the broad-region strategy produces noisier OCR than tighter blocks.

The most useful future region improvement would be faster text-block setup: either better text-region proposals, split/merge tools, or keyboard/page navigation that makes manual region marking less repetitive.

## OCR-All Usefulness

OCR-all is useful and should remain part of the main workflow. It created page-level attempts without requiring repeated selected-region OCR clicks and respected skip/protection rules:

- edited/reviewed text was not overwritten;
- image/diagram regions were skipped;
- ignored regions were skipped;
- no per-region OCR errors occurred in this batch.

## OCR Queue Usefulness

The current page-level OCR queue is useful for pending attempt review. It makes the post-OCR-all state inspectable and supports the correct accept/edit/reject workflow.

The limitation is scope: this batch still has pending attempts spread across pages, so a project-wide or selected-pages review queue may become useful once more pages have reviewed regions and OCR attempts.

## reviewed_text Correction Flow

The correction flow is correct and non-destructive. `page_025` demonstrates the important behavior:

- raw OCR did not perfectly recover the opening line;
- reviewer correction stored `[Question.]` and `Aaaa yes.`;
- export preferred `reviewed_text`;
- raw OCR remained preserved in review bundle/HTML/DOCX appendix when requested;
- `source_text_mutated=false` remained true.

The biggest burden is not the data model; it is the time required to correct noisy OCR on broad text regions.

## HTML Export Usefulness

HTML is useful as a readable research artifact. It includes:

- document heading and page table of contents;
- page sections in correct order;
- corrected reviewed text;
- crop assets for image/diagram regions;
- raw OCR details;
- warnings such as `reading_order_uncertain` and `needs_review`;
- `source_text_mutated=false`.

HTML remains the best review/read artifact when raw OCR evidence should be visible but not always expanded.

## DOCX Export Usefulness

DOCX is useful as a clean editable document artifact.

The fresh default export path was:

```text
data/runtime/review_projects/ocr_all_batch_20260505/exports/20260506_183146/docx/document.docx
```

The fresh raw-OCR-appendix export path was:

```text
data/runtime/review_projects/ocr_all_batch_20260505/exports/20260506_183148/docx/document.docx
```

Both were generated from the same six selected pages and matched the review bundle/HTML page selection.

### DOCX with Raw OCR Appendix Off

Result: pass.

The default DOCX converted successfully with macOS `textutil`. It included reviewed text such as:

```text
[Question.]
Aaaa yes. I'm glad you're frank.
```

It included the note:

```text
DOCX main body omits full raw OCR by default for readability.
```

It did not include the raw OCR appendix. This is the right default for a readable document.

### DOCX with Raw OCR Appendix On

Result: pass.

The appendix export converted successfully with `textutil` and included:

```text
Appendix: Raw OCR Evidence
```

The appendix contained raw OCR evidence for page/region records, including the imperfect raw OCR for the `IMG_3312` opening text. This is useful when the DOCX itself must carry evidence, but for ordinary research reading the review bundle is enough.

## Biggest Remaining Blocker for Goal 1

The biggest remaining blocker is page/region workflow speed, especially text-region setup and page-to-page movement through pending OCR attempts.

Export is no longer the blocker. OCR execution is no longer the blocker. The main operator cost is:

```text
draw or refine text regions
  -> review noisy OCR
  -> move to next page/attempt
```

## Goal 1 Readiness

Choose one:

- [x] Current workbench is useful enough for Goal 1 prototype use
- [ ] Useful but needs one more workflow feature first
- [ ] Not yet useful enough; major blocker remains

## Recommended Next Branch

Choose one primary next branch:

- [x] page/project navigation ergonomics
- [ ] project-wide OCR review queue
- [ ] DOCX polish
- [ ] PDF export
- [ ] HTML/export polish
- [ ] OCR/recognition improvement
- [ ] more batch review

## Rationale

Goal 1 is prototype-ready: the workbench can produce reviewed text, image crops, HTML, DOCX, and audit artifacts from real Donn Draeger pages while preserving raw OCR and reviewer corrections separately.

The next highest-value feature should reduce operator movement cost across pages and attempts. A project-wide OCR queue may be useful later, but the broader friction is navigation: quickly moving through pages, seeing which pages have pending OCR/region work, and resuming incomplete review without hunting through the page list.
