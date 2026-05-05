# Workbench DOCX Real Project Review

## Purpose

Validate DOCX export on the same six-page reviewed Donn Draeger batch before adding PDF or further export features.

## Project Reviewed

- Project ID: `ocr_all_batch_20260505`
- Project path: `data/runtime/review_projects/ocr_all_batch_20260505/project_state.json`
- Pages exported: `page_025` through `page_030` (`IMG_3312.jpg` through `IMG_3317.jpg`)
- Export path: `data/runtime/review_projects/ocr_all_batch_20260505/exports/20260505_235545/`
- Formats exported: `review_bundle`, `html`, `docx`
- DOCX path: `data/runtime/review_projects/ocr_all_batch_20260505/exports/20260505_235545/docx/document.docx`

## Environment

- DOCX opened in:
  - [x] Microsoft Word
  - [ ] LibreOffice
  - [ ] Apple Pages
  - [x] Other: macOS `textutil` conversion to text
- Notes:
  - Microsoft Word is installed at `/Applications/Microsoft Word.app`.
  - `open -a "Microsoft Word" .../document.docx` completed without command error.
  - LibreOffice CLI was not installed.
  - Apple Pages is installed, but was not opened in this pass.
  - `textutil -convert txt` successfully converted the DOCX to `/tmp/workbench_docx_real_project_review.txt`.
  - The DOCX ZIP package contained `word/document.xml`, relationships, content types, and four embedded page 025 image crops under `word/media/`.

## Summary Table

| Check | Result | Notes |
|---|---|---|
| DOCX opens | pass | Microsoft Word open command succeeded; `textutil` conversion also succeeded. |
| Page order correct | pass | Text extraction shows Page 1 `IMG_3312.jpg` through Page 6 `IMG_3317.jpg` in order. |
| reviewed_text appears correctly | pass | `[Question.]` and corrected `Aaaa yes. I'm glad you're frank.` appear on Page 1. |
| raw OCR preserved | pass | `Raw / cleaned OCR evidence` appears for text/caption regions. |
| raw OCR not too disruptive | partial | Useful for audit, but verbose in DOCX; likely should become optional/collapsible-equivalent later. |
| image crops embedded | pass | DOCX package includes four page 025 crop images in `word/media/`. |
| image sizing reasonable | pass | Embedded extents are constrained to page width in WordprocessingML. Visual sizing still deserves human review in Word. |
| page headings useful | pass | Page headings include page index and source filename. |
| source/orientation metadata useful | pass | Page source path and orientation summary appear before each page's regions. |
| warnings/notes visible | pass | Warnings and region notes are present in extracted text. |
| DOCX matches HTML/review bundle content | pass | Same pages, reviewed text, raw OCR evidence, crops, and source metadata are present. |

## Page-by-Page Review

### `page_025` / `IMG_3312.jpg`

- Heading/source metadata: present; includes Page 1, filename, page ID, source path, and orientation correction.
- Reviewed text: corrected `[Question.]` and `Aaaa yes. I'm glad you're frank.` appear correctly.
- Raw OCR evidence: present after reviewed text; includes original imperfect `(Question. ]` evidence.
- Image crops: four figure/diagram crops embedded in the DOCX package and referenced in text as crop paths.
- Notes/warnings: `regions_unreviewed`, `reading_order_uncertain`, and region notes are visible.
- Layout/readability: readable research reconstruction; not pixel-perfect.
- Issues: raw OCR section is long and may distract in DOCX.

### `page_026` / `IMG_3313.jpg`

- Heading/source metadata: present.
- Reviewed text: OCR text appears because no reviewed text exists.
- Raw OCR evidence: present.
- Image crops: none expected.
- Notes/warnings: `reading_order_uncertain` visible.
- Layout/readability: readable, but OCR quality is noisy.
- Issues: raw OCR duplicates display text when no reviewed text exists.

### `page_027` / `IMG_3314.jpg`

- Heading/source metadata: present.
- Reviewed text: OCR text appears as display text.
- Raw OCR evidence: present.
- Image crops: none expected.
- Notes/warnings: `reading_order_uncertain` visible.
- Layout/readability: structurally readable, OCR content noisy.
- Issues: same raw-OCR verbosity issue.

### `page_028` / `IMG_3315.jpg`

- Heading/source metadata: present.
- Reviewed text: OCR text appears as display text.
- Raw OCR evidence: present.
- Image crops: none expected.
- Notes/warnings: `reading_order_uncertain` visible.
- Layout/readability: acceptable for review.
- Issues: OCR quality still requires human correction.

### `page_029` / `IMG_3316.jpg`

- Heading/source metadata: present.
- Reviewed text: OCR text appears as display text.
- Raw OCR evidence: present.
- Image crops: none expected.
- Notes/warnings: `reading_order_uncertain` visible.
- Layout/readability: useful draft document.
- Issues: Japanese/history vocabulary still needs reviewer cleanup.

### `page_030` / `IMG_3317.jpg`

- Heading/source metadata: present; Page 6 order confirmed.
- Reviewed text: OCR text appears as display text.
- Raw OCR evidence: present.
- Image crops: none expected.
- Notes/warnings: `reading_order_uncertain` visible.
- Layout/readability: document structure is fine, but OCR quality is poor.
- Issues: this page needs tighter regions, variants, or heavier manual correction.

## Comparison Against HTML

- Same pages: yes, both include `page_025` through `page_030`.
- Same reviewed text: yes, both include corrected `[Question.]`.
- Same crop/image content: yes for page 025; HTML references `assets/page_025_region_*.png`, DOCX embeds corresponding `word/media/page_025_region_*.png`.
- Same warnings/notes: yes, visible in both.
- Differences:
  - HTML raw OCR is collapsible via `<details>`.
  - DOCX raw OCR is inline text and therefore more prominent.
  - HTML is easier to scan because styling and collapsible evidence are richer.

## Comparison Against Review Bundle

- JSON/Markdown/text available: yes for all six pages.
- Raw OCR preserved: yes.
- reviewed_text preferred: yes, `page_025_text.txt`, HTML, and DOCX all use `[Question.]`.
- Crops available: yes in review bundle crops and DOCX embedded media.
- Differences:
  - Review bundle remains the better audit/recovery artifact.
  - DOCX is better for readable sharing/editing, but raw OCR evidence can be verbose.

## Issues Found

Blocking issues: none.

Non-blocking issues:

- Raw OCR evidence is present but too prominent for a clean reading DOCX.
- DOCX headings are functional but plain.
- Image sizing appears constrained in the DOCX XML, but visual sizing should be reviewed in Word/Pages by a human operator.
- DOCX does not yet provide include/exclude options for raw OCR, notes, warnings, or crops.

## Operator Friction

- What felt rough?
  - Raw OCR inline sections make the DOCX longer than a reader-facing document should be.
  - OCR noise is very visible on unreviewed pages.
- What was good enough?
  - DOCX opens/parses, preserves reviewed text, embeds crops, and keeps page order.
  - Metadata is useful for research traceability.
- What would make DOCX more useful?
  - Export options: include raw OCR yes/no, include notes/warnings yes/no.
  - Heading/style polish.
  - Optional clean DOCX mode that omits raw OCR while keeping the review bundle as audit evidence.

## Recommendation

Choose one primary next branch:

- [x] DOCX polish
- [ ] project-wide OCR review queue
- [ ] page/project navigation ergonomics
- [ ] more real batch review
- [ ] PDF export
- [ ] export bug fix

## Rationale

DOCX export is functionally valid and opens/parses, so PDF should still wait. The main issue is not correctness; it is presentation. A small DOCX polish pass should add export options or a cleaner DOCX mode so DOCX can be used as a readable research document while review bundle/Markdown/JSON retain full raw OCR evidence.
