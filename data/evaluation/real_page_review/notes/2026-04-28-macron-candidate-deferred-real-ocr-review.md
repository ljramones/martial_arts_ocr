# Macron Candidate Deferred Real OCR Review

## Purpose

Review the deferred real OCR macron candidates from the first review batch using source image context.

Focus:

```text
BUDO -> budō
```

These candidates were deferred because they appeared in a title/publication-style context and needed source image review before accepting any normalization.

## Source Reviewed

All 8 deferred real OCR candidates resolve to the same source sample:

```text
corpus: dfd
sample_id: original_img_3288
path: data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg
page_type: mixed English/Japanese / manual selection
```

Source image inspected:

```text
yes
```

The source image visibly contains:

```text
1. BUJUTSU AND BUDO.
```

This is an all-caps typewritten title/list item.

## Candidate Queue Reviewed

Input queue:

```text
data/notebook_outputs/macron_candidate_review/review_queue_deferred_real_ocr.md
data/notebook_outputs/macron_candidate_review/review_queue_deferred_real_ocr.csv
```

The 8 queue rows are duplicate appearances of the same source occurrence across related OCR review summaries:

```text
ocr_text_quality_review
ocr_text_readability_sampling/eng_auto
ocr_text_reading_order_after_line_grouping/eng_auto
document_result_serialization_review
```

## Decisions

Decision after image review:

```text
edit
```

Reviewed value:

```text
BUDŌ
```

Rationale:

- The source image is an all-caps title/list item.
- The semantic term is `budō`.
- The generated candidate `budō` is directionally correct but does not preserve the title capitalization.
- `BUDŌ` is a better reviewed value for a normalized title export.
- Original OCR/source text remains unchanged.

Updated review export counts:

```text
accept: 32
reject: 0
defer: 0
edit: 8
pending: 372
stale: 0
reviewed: 40
source_text_mutated: false
```

## Examples

Before review:

```json
{
  "observed": "BUDO",
  "candidate": "budō",
  "decision": "defer",
  "context": "1. BUJUTSU AND BUDO."
}
```

After review:

```json
{
  "observed": "BUDO",
  "candidate": "budō",
  "decision": "edit",
  "reviewed_value": "BUDŌ",
  "notes": [
    "Source image inspected: term appears in all-caps typewritten title line; reviewed value preserves title capitalization while adding macron."
  ]
}
```

## Source Image Finding

The source image confirms the OCR text is faithful to the source typography:

```text
BUJUTSU AND BUDO
```

The review decision is therefore not an OCR correction in the narrow transcription sense. It is a normalized review/export value for a domain term while preserving source capitalization.

This distinction matters:

```text
source OCR text: BUDO
candidate suggestion: budō
reviewed export value: BUDŌ
source_text_mutated: false
```

## Candidate Context Assessment

The queue context was enough to identify the term and title-like phrase:

```text
1. BUJUTSU AND BUDO.
```

But the source image was needed to confirm:

- all-caps typewriter style,
- list/title context,
- no surrounding prose that would justify lower-case replacement.

## Glossary / Rule Findings

No glossary rule change is needed.

The candidate generator correctly suggested:

```text
BUDO -> budō
```

The review workflow correctly allowed the reviewer to choose:

```text
edit -> BUDŌ
```

Future improvement should not be automatic capitalization rewriting. It should be review UI/export support that makes capitalization-preserving edits easy.

## Workflow Findings

What worked:

- deferred candidates were easy to isolate with `--filter deferred --source-filter real_ocr`;
- source summaries contained enough metadata to trace the image path;
- `edit` decisions handled capitalization-preserving reviewed values;
- rerunning the helper preserved the local decision file;
- stale count remained zero;
- source text remained unmutated.

Pain points:

- the review queue itself did not include `sample_id` or original image path directly;
- duplicate source occurrences across review summaries required treating 8 rows as one source decision;
- source-image review required manually tracing from queue row to summary JSON to image path.

## Decision

- [x] Source image inspected.
- [x] `BUDO -> budō` is semantically appropriate but exact candidate casing is not.
- [x] All 8 duplicate real OCR rows changed from `defer` to `edit`.
- [x] Reviewed value is `BUDŌ`.
- [x] No source OCR text was changed.
- [x] No glossary rule change is needed.

## Recommended Next Pass

Improve queue source references and duplicate grouping:

```text
- include sample_id when available
- include original image/source path when available
- group duplicate observed/candidate/context rows
- show occurrence count and source summary list
```

This should happen before reviewing many more real OCR candidates.

Do not add automatic normalization.
