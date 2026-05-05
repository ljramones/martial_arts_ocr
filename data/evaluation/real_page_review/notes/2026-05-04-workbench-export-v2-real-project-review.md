# Workbench Export v2 Real Project Review

## Purpose

Validate Export v2 review_bundle and HTML output on a small real reviewed project before adding DOCX/PDF or export polish.

## Project Reviewed

- Project ID: `export_v2_real_review_20260505`
- Project path: `data/runtime/review_projects/export_v2_real_review_20260505/project_state.json`
- Pages reviewed: `page_025`, `page_026`, `page_027`
- Reviewed text regions: 1 reviewed English text block on `page_025`
- Edited reviewed_text present: yes
- Image/diagram regions: 4 diagram/image crops on `page_025`
- Ignored regions: 1 runtime-only ignored region added to this copied validation project
- Notes: This was an ignored runtime-only copy derived from the real `img3312_e2e_20260504` project so the export review could cover the ignored-region case without modifying committed data or private image payloads.

## Export Scenarios

| Scenario | Page Selection | Formats | Result | Notes |
|---|---|---|---|---|
| Current page | `current`, `page_025` | `review_bundle`, `html` | pass | Manifest recorded `page_025`; HTML and review bundle were written. |
| Selected pages | `selected`, `page_025`, `page_026` | `review_bundle`, `html` | pass | Manifest recorded the two selected pages in project order. |
| Page range | `range`, `page_025` to `page_027` | `review_bundle`, `html` | pass | Manifest recorded `page_025`, `page_026`, `page_027`. |
| All pages | `all` | `review_bundle`, `html` | pass | Exported all 3 pages in the runtime validation project. |

## Export Folder Structure

- Export path: `data/runtime/review_projects/export_v2_real_review_20260505/exports/20260505_192822_3/`
- Manifest: `export_manifest.json`
- Document export model: `document_export_model.json`
- Review bundle: `review_bundle/pages/` and `review_bundle/crops/`
- HTML: `html/document.html`
- Assets/crops: `html/assets/`

The all-pages export contained:

```text
export_manifest.json
project_state_snapshot.json
document_export_model.json
review_bundle/
  pages/page_025_review.json
  pages/page_025_review.md
  pages/page_025_text.txt
  pages/page_026_review.json
  pages/page_026_review.md
  pages/page_026_text.txt
  pages/page_027_review.json
  pages/page_027_review.md
  pages/page_027_text.txt
  crops/page_025_region_*.png
html/
  document.html
  assets/page_025_region_*.png
```

## HTML Review

- Readability: usable as a clean research reconstruction.
- Page sections: present for all exported pages.
- Text blocks: reviewed text appeared before image blocks for `page_025`.
- Image/crop blocks: diagram crops appeared with region IDs, bbox metadata, and relative asset paths.
- Captions/labels: caption region appeared as text/notes when no OCR attempt existed.
- Warnings/metadata: warnings were visible, including `regions_unreviewed`, `ocr_unreviewed`, `no_text_regions`, and `reading_order_uncertain`.
- Main issues: HTML faithfully included both the unreviewed machine-detected central diagram and a manually reviewed duplicate of the same central diagram. This is not an export correctness bug, but it shows the HTML needs later presentation/filtering options or the reviewed project should clean up unreviewed duplicates before final export.

## Review Bundle Review

- Page JSON present: yes, for all three pages.
- Page Markdown present: yes, for all three pages.
- Page text present: yes, for all three pages.
- Crops present: yes, for non-ignored `page_025` regions.
- Raw OCR preserved: yes. `page_025_review.json` preserved the raw OCR for `r_001`.
- reviewed_text preferred: yes. `page_025_text.txt` and HTML start with `[Question.]`.
- ignored regions handled: yes. `r_ignored_export_check` appeared in JSON with `effective_type=ignore`, `ignored=true`, and `crop_path=null`; it did not appear in text/crops.
- Main issues: review bundle is complete, but the current bundle crops all non-ignored regions, including text and caption regions. That is acceptable for audit/recovery, but future UI wording should make clear that review-bundle crops are broader than HTML image assets.

## Manifest / Model Review

- Page selection correct: yes for current, selected, range, and all scenarios.
- Formats correct: yes, `["review_bundle", "html"]`.
- source_text_mutated=false: yes in manifest, document model, page review JSON, and OCR attempt export.
- Relative asset paths: yes for HTML assets and review-bundle crop paths inside page JSON/Markdown.
- Main issues: manifest artifact paths are absolute local paths. That is acceptable for local runtime artifacts, while HTML asset paths and page review crop paths remain relative/useful.

## Selection Mode Review

### Current Page

Passed. Exported only `page_025`; manifest page IDs were `["page_025"]`.

### Selected Pages

Passed. Exported `page_025` and `page_026`; manifest page IDs were `["page_025", "page_026"]`.

### Range

Passed. Exported `page_025`, `page_026`, and `page_027`; manifest page IDs followed project order.

### All Pages

Passed. Exported all three pages in the runtime validation project.

## Bugs Found

No blocking Export v2 correctness bugs found.

Observed non-blocking issues:

- HTML surfaces duplicate/unreviewed regions exactly as stored in project state.
- Empty/lightly reviewed pages produce page sections with warnings but little content.
- Review-bundle crop output includes non-image text/caption crops by design.

## Friction / Usability Notes

- HTML is readable enough to inspect reviewed text and diagram crops.
- The warnings are useful, but they are visually plain and could be grouped better.
- Duplicate/unreviewed machine regions are easy to notice in HTML, which is useful, but later export options may need `include_unreviewed_machine_regions` or similar.
- The review bundle is more complete than the HTML, which matches the intended audit/recovery split.

## Recommendation

Choose one primary next branch:

- [x] HTML export polish
- [ ] OCR all reviewed text regions
- [ ] DOCX export
- [ ] PDF export
- [ ] Export bug fix
- [ ] More real export testing

## Rationale

Export v2 is functionally correct on a real reviewed project. The next bottleneck is not data preservation; it is readability and export presentation. Before adding DOCX/PDF, improve HTML enough to establish the document structure that those later exporters should follow.

Recommended HTML polish scope:

- clearer page headings using page filename/source metadata;
- better warning grouping;
- optional hiding or de-emphasizing unreviewed machine regions;
- clearer caption/label display;
- modest styling for text blocks and image blocks;
- keep the audit/review bundle unchanged.
