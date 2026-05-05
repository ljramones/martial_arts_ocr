# Workbench HTML Export Polish

## Purpose

Improve the readability of Export v2 `html/document.html` while preserving the research-workbench audit boundary.

The HTML export should be easier to read and navigate, but it must not make OCR/layout output look more certain than it is.

## What Was Polished

- Added a document-level header with project ID, created time, page count, export version, formats, and `source_text_mutated=false`.
- Added a page table of contents with links to page sections.
- Improved page headings with page index, filename/page ID, source path, orientation summary, and region count summary.
- Rendered page warnings as visible badges.
- Rendered regions as bordered blocks with region ID, effective type, bbox, source, status, and review badges.
- Rendered reviewed/display text prominently in text blocks.
- Preserved raw/cleaned OCR in collapsible `<details>` sections.
- Rendered image/diagram/photo crops as `<figure>` blocks with crop asset captions.
- Added simple print-friendly CSS and better spacing/typography.

## How Uncertainty Is Surfaced

The HTML now keeps uncertainty visible through:

- `source_text_mutated=false` badge in the document header.
- Page warning badges such as `reading_order_uncertain`, `regions_unreviewed`, `ocr_unreviewed`, and `no_text_regions`.
- Region badges for block type and review status.
- `needs_review` warning badges when region/attempt metadata indicates review is still needed.
- Collapsible raw OCR evidence so reviewed text can be inspected against OCR output.

## What Remains Deferred

- DOCX export.
- PDF export.
- Translation export.
- Pixel-perfect scanned-page reconstruction.
- Full publication styling.
- Reviewer-controlled reading order.
- Include/exclude toggles for unreviewed machine regions or raw OCR.

## Automated Validation

Updated Export v2 tests cover:

- document-level HTML header;
- `source_text_mutated=false` visibility;
- page table of contents;
- page sections and page labels;
- reviewed text display;
- raw OCR in collapsible details;
- needs-review badge output;
- `reading_order_uncertain` warning output;
- crop image references;
- relative asset paths;
- unchanged review-bundle behavior.

## Notes

This pass changes HTML presentation only. It does not change review-bundle semantics, export manifest semantics, page selection behavior, OCR behavior, detection/orientation behavior, or source text mutation rules.
