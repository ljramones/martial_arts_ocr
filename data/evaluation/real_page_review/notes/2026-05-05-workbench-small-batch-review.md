# Workbench Small-Batch Review

## Purpose

Validate the local research review workbench on a small real batch before adding more features.

This review used runtime-only project state and generated exports. It was a workflow validation pass, not a full editorial transcription of every page.

## Batch Summary

| Page | Page Type | Orientation | Regions | OCR | Export | Main Friction |
|---|---|---|---|---|---|---|
| `page_025` / `IMG_3312.jpg` | mixed text + figures + caption | detected 270, corrected 90 | 1 machine region kept, 6 manual regions | 1 text region OCRed, variants used, reviewed_text edited | pass | Manual side-figure/caption regions still needed; selected OCR is useful but per-region. |
| `page_026` / `IMG_3313.jpg` | mostly English text | detected 270, corrected 90 | 1 manual text region | 1 default OCR attempt, accepted for workflow smoke | pass | Broad text region OCR was noisy; real review needs better text block segmentation. |
| `page_027` / `IMG_3314.jpg` | noisy English text | detected 270, corrected 90 | 1 manual text region | 1 default OCR attempt, accepted for workflow smoke | pass | OCR quality degraded on broad/noisy crop. |
| `page_028` / `IMG_3315.jpg` | English text / awkward scan | detected 270, corrected 90 | 1 manual text region | 1 default OCR attempt, accepted for workflow smoke | pass | Manual text region plus OCR click repetition is tedious. |
| `page_029` / `IMG_3316.jpg` | English text with Japanese/history terms | detected 270, corrected 90 | 1 manual text region | 1 default OCR attempt, accepted for workflow smoke | pass | OCR usable enough for draft text, but needs human review. |
| `page_030` / `IMG_3317.jpg` | awkward/noisy page | detected 0, corrected 0 | 1 manual text region | 1 default OCR attempt, accepted for workflow smoke | pass | OCR output was poor; page likely needs tighter regions or variants. |

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
  - machine regions kept: 1
  - manual regions drawn: 6
  - regions resized: 0 during this scripted pass
  - image/diagram regions: 4
  - text regions: 1
  - ignored regions: 1
- OCR review:
  - regions OCRed: 1
  - variants used: yes
  - reviewed_text corrections: corrected the known opening line to start with `[Question.]` / `Aaaa yes.`
  - quality: useful draft OCR, still needs human correction
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: figure crops exported and visible
- Operator notes:
  - what worked: orientation, manual regions, OCR variants, reviewed_text, HTML/export bundle
  - what was tedious: side figure/caption regions still required manual work
  - what was confusing: duplicate/unreviewed machine region remains visible unless cleaned up
  - what was missing: a faster way to OCR all reviewed text regions

### `page_026` / `IMG_3313.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3313.jpg`
- Page type: mostly English text
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions kept: 0
  - manual regions drawn: 1 broad English text region
  - regions resized: 0
  - image/diagram regions: 0
  - text regions: 1
  - ignored regions: 0
- OCR review:
  - regions OCRed: 1
  - variants used: no
  - reviewed_text corrections: accepted OCR output for workflow smoke, not full editorial review
  - quality: noisy because the crop was broad and not carefully segmented
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: text crop included in review bundle; no HTML image asset
- Operator notes:
  - what worked: page entered the reviewed/exported flow
  - what was tedious: manual text region setup and per-region OCR
  - what was confusing: broad-region OCR makes output look worse than a tighter human-selected block might
  - what was missing: text-block proposals or batch OCR after drawing regions

### `page_027` / `IMG_3314.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3314.jpg`
- Page type: noisy English text
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions kept: 0
  - manual regions drawn: 1 broad English text region
  - regions resized: 0
  - image/diagram regions: 0
  - text regions: 1
  - ignored regions: 0
- OCR review:
  - regions OCRed: 1
  - variants used: no
  - reviewed_text corrections: accepted OCR output for workflow smoke, not full editorial review
  - quality: noisy
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: text crop included in review bundle
- Operator notes:
  - what worked: orientation and export pipeline
  - what was tedious: selected OCR repeated page by page
  - what was confusing: OCR quality varies sharply with crop quality
  - what was missing: faster text-region creation and OCR-all reviewed text regions

### `page_028` / `IMG_3315.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3315.jpg`
- Page type: English text / awkward scan
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions kept: 0
  - manual regions drawn: 1 broad English text region
  - regions resized: 0
  - image/diagram regions: 0
  - text regions: 1
  - ignored regions: 0
- OCR review:
  - regions OCRed: 1
  - variants used: no
  - reviewed_text corrections: accepted OCR output for workflow smoke, not full editorial review
  - quality: partial/noisy
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: text crop included in review bundle
- Operator notes:
  - what worked: basic page-to-export flow
  - what was tedious: manual region creation plus OCR click for each page
  - what was confusing: none beyond OCR quality
  - what was missing: batch OCR over reviewed text regions

### `page_029` / `IMG_3316.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3316.jpg`
- Page type: English text with Japanese/history vocabulary
- Orientation:
  - detected: 270
  - correction: 90 clockwise
  - manual override: no
  - result: upright effective page
- Region review:
  - machine regions kept: 0
  - manual regions drawn: 1 broad English text region
  - regions resized: 0
  - image/diagram regions: 0
  - text regions: 1
  - ignored regions: 0
- OCR review:
  - regions OCRed: 1
  - variants used: no
  - reviewed_text corrections: accepted OCR output for workflow smoke, not full editorial review
  - quality: better than the noisiest pages but still requires review
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: text crop included in review bundle
- Operator notes:
  - what worked: draft text appears in HTML with raw OCR evidence preserved
  - what was tedious: selected OCR repetition
  - what was confusing: none
  - what was missing: macron/romanized Japanese candidate pass is not yet surfaced in the workbench export workflow

### `page_030` / `IMG_3317.jpg`

- Source path: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3317.jpg`
- Page type: awkward/noisy page
- Orientation:
  - detected: 0
  - correction: 0
  - manual override: no
  - result: page served without correction
- Region review:
  - machine regions kept: 0
  - manual regions drawn: 1 broad English text region
  - regions resized: 0
  - image/diagram regions: 0
  - text regions: 1
  - ignored regions: 0
- OCR review:
  - regions OCRed: 1
  - variants used: no
  - reviewed_text corrections: accepted OCR output for workflow smoke, not full editorial review
  - quality: poor; likely needs tighter crops and variants
- Export:
  - review_bundle: pass
  - HTML: pass
  - crop usefulness: text crop included in review bundle
- Operator notes:
  - what worked: pipeline completed
  - what was tedious: poor OCR increases correction burden
  - what was confusing: this page needs closer operator inspection than the scripted pass provided
  - what was missing: fast rerun/variant controls across several text regions

## Cross-Page Findings

- The workbench is now usable enough to process a real batch at research-prototype quality.
- Orientation succeeded on the pages exercised: five pages needed 90-degree correction, one did not.
- Recognition was useful for `IMG_3312` figure detection but did not help much for text-heavy pages.
- Manual region drawing remains essential and works, but repeated text region creation is a workflow cost.
- Selected-region OCR works, but clicking/running OCR one region at a time is the most obvious scaling friction.
- Export v2 is reliable and useful: review bundle and HTML were produced for all six pages.
- HTML is now readable enough to inspect the batch and review warnings.

## Workflow Friction

The main friction is not export anymore. It is getting from reviewed regions to OCR attempts across multiple pages.

Current repeated steps:

```text
select page
draw/select text region
run OCR
accept/edit reviewed_text
repeat
```

This is acceptable for one page, but tedious for 5-10 pages. It will be much worse for a full folder.

## Export Usefulness

- HTML is sufficient as the next readable artifact.
- Review bundle is complete enough for audit/recovery.
- `reviewed_text` appears in the exported text and HTML.
- Raw OCR remains available in JSON/Markdown and collapsible HTML details.
- `source_text_mutated=false` remains visible.
- Image crops are useful for reviewed diagram regions.

DOCX/PDF should still wait until the batch workflow is less click-heavy.

## OCR Quality Notes

- OCR quality is highly crop-dependent.
- The known `IMG_3312` opening line still needed human correction, and reviewed_text handled it correctly.
- Broad text regions on pages `page_026` through `page_030` produced noisy draft OCR.
- Tighter text blocks and OCR variants may improve output, but manual correction remains necessary.

## Region Review Notes

- Figure/image region review is workable when the user manually draws missed regions.
- Recognition helps with some diagram regions but is not a reliable text-block proposer.
- Text-heavy pages currently require manual text region creation.
- The page inventory/export warnings make incomplete review state visible.

## Recommended Next Branch

Choose one primary recommendation:

- [x] OCR all reviewed text regions
- [ ] DOCX export
- [ ] PDF export
- [ ] Region/page navigation improvements
- [ ] Keyboard shortcuts / faster editing
- [ ] Better recognition proposals
- [ ] More batch review before feature work

## Rationale

The workbench is useful enough for real review work, and HTML export is sufficient for the next readable artifact. The biggest blocker is the repeated per-region OCR workflow. Once a reviewer has drawn text regions across several pages, the system should be able to run OCR over all reviewed text-like regions that do not yet have reviewed attempts.

Recommended next implementation:

```text
OCR all reviewed text regions
  - current page and selected/all pages modes
  - only text-like reviewed regions
  - skip ignored/image regions
  - skip regions with accepted/edited reviewed_text unless rerun is explicit
  - store attempts exactly like selected-region OCR
  - preserve raw OCR/reviewed_text separation
```

## Specific Questions

1. Is the workbench useful enough for real review work?

   Yes, for local research-prototype review. It can orient pages, hold reviewed regions, OCR selected regions, preserve corrections, and export useful HTML/audit bundles.

2. Is selected-region OCR too manual for multi-page batches?

   Yes. It is the clearest scaling bottleneck after this 6-page pass.

3. Is HTML export sufficient as the next readable artifact?

   Yes. It is readable enough to defer DOCX/PDF until OCR/page batching improves.

4. Is DOCX now the next useful export format?

   Not yet. DOCX will be more valuable after batch OCR reduces the amount of manual clicking needed to produce reviewed text.

5. Is the biggest blocker OCR, region editing, navigation, or export?

   OCR workflow is the biggest blocker: specifically running OCR one reviewed region at a time. Region editing remains manual, but usable. Export is no longer the blocker.

6. What should be implemented next?

   Implement OCR-all-reviewed-text-regions for current/selected/range/all page scopes, with conservative skip/rerun behavior and no source text mutation.
