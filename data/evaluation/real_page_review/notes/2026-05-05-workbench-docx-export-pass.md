# Workbench DOCX Export Pass

## Purpose

Add DOCX as the next readable Export v2 format after review bundle and polished HTML proved useful. The goal is a practical research document, not a pixel-perfect recreation of scanned pages.

## Added Behavior

- Export v2 now accepts `docx` in the format list.
- The workbench export controls include a DOCX checkbox.
- The project-level export endpoint writes:

```text
docx/
  document.docx
  assets/
    page_<page_id>_region_<region_id>.png
```

- The export manifest and API response include the DOCX artifact path.

## DOCX Content

The DOCX is generated from the shared Export v2 document model and includes:

- document heading and export metadata;
- page headings and source/orientation metadata;
- region headings and bbox/source/status metadata;
- reviewed/display text;
- raw/cleaned OCR evidence;
- image/diagram/photo crops where available;
- notes and warnings.

## Non-Destructive Rules

DOCX export does not alter workbench state, OCR attempts, reviewed text, raw OCR text, source images, detector output, or export bundle semantics. `source_text_mutated=false` remains explicit.

## Implementation Note

No new dependency was added. The DOCX writer uses the standard library to write a minimal WordprocessingML package with embedded image crops.

## Deferred

- PDF export;
- pixel-perfect scan reproduction;
- advanced DOCX styling;
- table of contents fields;
- page/project-level packaging.

## Verification

Automated tests cover DOCX generation, manifest/response paths, reviewed text, raw OCR evidence, and embedded image crops. Full project tests, Python compile checks, utility compile checks, JavaScript syntax check, and diff whitespace checks were run.
