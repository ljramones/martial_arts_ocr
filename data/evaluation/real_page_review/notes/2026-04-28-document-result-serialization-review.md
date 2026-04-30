# DocumentResult Serialization Review

## Purpose

Check whether the serialized `DocumentResult` output is easier to inspect after adding page/document text summaries and explicit line/word region aliases.

## Scope

Reviewed generated artifacts from three representative real OCR runs under ignored output:

```text
data/notebook_outputs/document_result_serialization_review/
```

Generated JSON, HTML, and text artifacts were used for inspection only and should not be committed.

## Pages Reviewed

| Sample ID | Output Doc | Page Type | Notes |
|---|---|---|---|
| original_img_3288 | doc_920001 | simple/list-like text | Good sanity check for compact summary and text.txt readability. |
| original_img_3337 | doc_920002 | dense text | Checks larger canonical word/line hierarchy. |
| corpus2_new_doc_2026_04_28_16_55_48 | doc_920003 | mixed/noisy layout | Checks uncertain reading-order metadata and compact structure on harder output. |

## Summary

| Output Doc | Word Count | Line Count | Reading Order Uncertain | Line Regions | Word Regions | Text Regions | Result |
|---|---:|---:|---|---:|---:|---:|---|
| doc_920001 | 56 | 10 | false | 10 | 56 | 66 | Easier to inspect. |
| doc_920002 | 598 | 59 | false | 59 | 598 | 657 | Large output remains detailed, but summary and aliases make it navigable. |
| doc_920003 | 110 | 22 | true | 22 | 110 | 132 | Mixed layout remains imperfect, but uncertainty is visible. |

## Findings

- `data.json` now exposes document-level `text_summary` with page count, word count, line count, readable text, and reading-order uncertainty.
- Each page now exposes `text_summary`, `readable_text`, `line_regions`, and `word_regions` without removing the legacy `text_regions` list.
- `text.txt` uses compact readable text, so it is no longer dominated by low-level OCR region serialization.
- Alternate OCR candidates remain metadata diagnostics rather than canonical word/line content.
- Mixed-layout OCR is still limited by reading order and OCR quality; serialization now surfaces that limitation more clearly instead of hiding it in a large flat region list.

## Decision

The serialization polish improves artifact inspection without changing OCR or extraction behavior. The next useful pass should focus on page reconstruction or review artifact usability, not another data model expansion.
