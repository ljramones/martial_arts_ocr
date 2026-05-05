# Workbench DOCX Polish Pass

## Purpose

Make DOCX export more readable by default after real-project validation showed that inline raw OCR evidence was useful for audit but visually disruptive in the main document flow.

## Problem From Validation

The DOCX real-project review confirmed that DOCX opened structurally, included all pages in order, preserved reviewed text, preserved raw OCR evidence, embedded crops, and matched the HTML/review bundle content.

The main operator friction was that raw OCR appeared inline after text regions, which made the DOCX harder to read as a research document.

## Default DOCX Behavior

Default DOCX export now uses:

```text
include_raw_ocr=false
```

The DOCX main body includes:

- page headings and source/orientation metadata;
- region metadata;
- reviewed/display text;
- image/diagram crops;
- warnings and notes;
- a concise note that raw OCR is preserved in the review bundle.

The DOCX main body no longer includes full raw OCR evidence inline by default.

## Raw OCR Option

Export v2 accepts:

```json
{
  "options": {
    "docx": {
      "include_raw_ocr": true
    }
  }
}
```

When enabled, DOCX includes:

```text
Appendix: Raw OCR Evidence
```

Raw OCR appears in the appendix by page and region rather than inline after each region.

## Audit Preservation

The DOCX option affects only DOCX presentation.

The review bundle continues to preserve full raw OCR evidence in JSON and Markdown artifacts. HTML keeps its existing raw OCR details behavior. Source OCR text remains non-mutated and `source_text_mutated=false` remains explicit.

## UI Behavior

The Export v2 UI adds:

```text
Include raw OCR appendix in DOCX
```

It is unchecked by default.

## Deferred

- PDF export remains deferred.
- DOCX publication-style layout remains deferred.
- Translation export remains deferred.
- No OCR, detection, orientation, or review state semantics changed.

## Validation

Automated tests cover:

- default DOCX omits full raw OCR from the document body;
- default DOCX includes reviewed/display text;
- default DOCX notes that raw OCR is preserved in the review bundle;
- `include_raw_ocr=true` creates a raw OCR appendix;
- review bundle raw OCR preservation is unchanged.
