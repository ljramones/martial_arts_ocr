# Macron Candidate Review Workflow Design

## Status

This is a design-only pass.

The project already has:

- glossary-backed macron candidate generation,
- review artifact scanning over OCR outputs,
- a hard rule that source OCR text is not mutated,
- no runtime integration,
- no canonical Japanese or romanization fields promoted.

This document defines how a human review workflow should accept, reject, or defer macron candidates without violating those constraints.

## Problem Statement

The candidate layer can identify likely missing-macron martial arts terms, such as:

```text
Daito-ryu -> Daitō-ryū
BUDO -> budō
koryG -> koryū
```

But candidates are not corrections. Publication style, historical spelling, headings, proper names, and OCR uncertainty can all make a candidate wrong or ambiguous.

The system needs a review/export workflow that:

- preserves original OCR text,
- stores reviewer decisions separately,
- records enough context for audit,
- can export reviewed corrections,
- can feed future glossary/fixture work,
- does not silently normalize text.

## Non-Goals

- No automatic candidate acceptance.
- No silent mutation of `raw_text`, `readable_text`, `text.txt`, or `data.json["text"]`.
- No runtime default integration.
- No schema/database migration in the first pass.
- No canonical Japanese or romanization field promotion yet.
- No broad fuzzy matching expansion.
- No correction UI implementation in this design pass.

## Review Workflow Principles

- Original OCR output is immutable.
- Candidate decisions are separate review records.
- Every accepted correction remains traceable to the observed OCR text and source span.
- Rejected candidates are valuable and should be preserved for glossary tuning.
- Deferred candidates should remain visible in future review passes.
- Exported corrected text must be explicitly labeled as reviewed or derived.
- Review data should be local/private by default until provenance and privacy are clear.

## Reviewer Actions

Each candidate can receive one decision:

| Decision | Meaning |
|---|---|
| `accept` | Candidate is correct for this occurrence. |
| `reject` | Candidate is not correct for this occurrence. |
| `defer` | Candidate needs more context or a domain reviewer. |
| `edit` | Candidate is close, but reviewer supplies a different correction. |

Reviewer decisions should also allow optional notes:

```text
publication style intentionally uses ASCII
heading preserves source capitalization
proper name; leave unchanged
accepted as normalized research spelling
needs source image review
```

## Decision Record Shape

Suggested JSON shape:

```json
{
  "candidate_id": "sha256:...",
  "document_id": "doc_123",
  "page_number": 1,
  "source_path": "data/runtime/processed/doc_123/data.json",
  "field_path": "$.pages[0].metadata.readable_text",
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [124, 133],
  "context": "the Daito-ryu tradition...",
  "match_type": "variant_exact",
  "source": "martial_arts_macron_glossary",
  "decision": "accept",
  "reviewed_value": "Daitō-ryū",
  "reviewer": "local",
  "reviewed_at": "2026-04-28T00:00:00Z",
  "notes": ["accepted as normalized research spelling"]
}
```

For `reject`, `reviewed_value` should usually be `null`:

```json
{
  "candidate_id": "sha256:...",
  "decision": "reject",
  "reviewed_value": null,
  "notes": ["source heading intentionally uses ASCII"]
}
```

For `edit`, `reviewed_value` records the reviewer-supplied correction:

```json
{
  "candidate_id": "sha256:...",
  "decision": "edit",
  "reviewed_value": "Daitō ryū",
  "notes": ["publication omits hyphen in this title"]
}
```

## Candidate Identity

Candidate IDs should be deterministic and stable across reruns when source text does not change.

Recommended identity inputs:

```text
source_path
document_id
page_number
field_path
span
observed
candidate
match_type
```

Recommended format:

```text
sha256:<hex>
```

This avoids relying on list position in a generated report.

## Storage Locations

Initial implementation should be local and review-artifact oriented.

Recommended ignored local location:

```text
data/notebook_outputs/macron_candidate_review/decisions.local.json
```

Recommended export location for a reviewed correction set:

```text
data/evaluation/real_page_review/notes/<date>-macron-candidate-review-export.json
```

Commit policy:

- `decisions.local.json` should stay ignored by default.
- Small anonymized/exported review examples may be committed only when source paths and text are safe.
- Private corpus paths, large artifacts, generated OCR output, and unreviewed bulk decisions should not be committed.

## Review Artifact Shape

A review artifact should combine:

```text
source identity
readable text preview
candidate list
candidate context
decision placeholder
reviewer notes placeholder
```

Suggested shape:

```json
{
  "source_id": "doc_123:page_1:readable_text",
  "source_path": "data/runtime/processed/doc_123/data.json",
  "document_id": "doc_123",
  "page_number": 1,
  "field_path": "$.pages[0].metadata.readable_text",
  "text_preview": "The Daito-ryu tradition...",
  "candidates": [
    {
      "candidate_id": "sha256:...",
      "observed": "Daito-ryu",
      "candidate": "Daitō-ryū",
      "span": [4, 13],
      "context": "The Daito-ryu tradition...",
      "match_type": "variant_exact",
      "requires_review": true,
      "decision": null,
      "reviewed_value": null,
      "reviewer_notes": []
    }
  ]
}
```

The artifact may also include aggregate counts:

```text
pending
accepted
rejected
deferred
edited
```

## Applying Reviewed Decisions

Applying decisions should produce derived review output, not mutate original OCR artifacts.

Possible derived outputs:

```text
reviewed_text
reviewed_text_diff
accepted_normalizations
rejected_candidates
```

Example:

```json
{
  "original_text": "The Daito-ryu tradition is discussed.",
  "reviewed_text": "The Daitō-ryū tradition is discussed.",
  "applied_decisions": ["sha256:..."],
  "source_text_mutated": false
}
```

Rules:

- Apply only accepted or edited decisions.
- Never overwrite `raw_text` or `readable_text`.
- Preserve a diff or list of applied spans.
- If source text changed after review, do not apply stale decisions automatically.

## Export Formats

Recommended first export formats:

1. Candidate review JSON:

```text
macron_candidate_review.json
```

Contains pending candidates and decision placeholders.

2. Decision JSON:

```text
macron_candidate_decisions.local.json
```

Contains reviewer decisions.

3. Reviewed correction export:

```text
macron_candidate_review_export.json
```

Contains accepted/rejected/deferred counts and optionally derived reviewed text.

CSV can be added later for spreadsheet review:

```text
candidate_id,source_id,observed,candidate,context,decision,reviewed_value,notes
```

JSON should be the canonical format because spans, context, and metadata are structured.

## Glossary Feedback Loop

Accepted and rejected decisions can improve future data, but only through explicit review.

Accepted decisions can suggest:

- new glossary variants,
- observed OCR-confusion forms,
- better term categories,
- fixture strings for tests.

Rejected decisions can suggest:

- terms that are too ambiguous,
- variants that should require stronger context,
- publication-style exceptions,
- stop-list rules.

No glossary update should happen automatically from a single accepted decision.

## Privacy And Commit Policy

Review artifacts may contain OCR text from private or copyrighted sources.

Default policy:

- local decision files stay ignored,
- generated review artifacts stay ignored,
- only design docs, helper code, tests, and safe summary notes are committed,
- exported reviewed examples require explicit privacy/provenance review before commit.

Do not commit:

```text
data/notebook_outputs/
data/runtime/
private images
bulk OCR outputs
local reviewer decision files
```

## Integration Points

Initial integration should stay experiment-only:

```text
experiments/review_macron_candidates.py
  -> candidate report
  -> decision template
  -> optional reviewed export
```

Later review-mode integration may add:

```text
PageResult.metadata["normalization_candidates"]
DocumentResult.metadata["normalization_candidates"]
```

That should wait until:

- the decision file format is stable,
- reviewer workflow exists,
- privacy and commit policy are clear,
- consumers understand candidates are not corrections.

## Test Strategy

Future implementation tests should cover:

- deterministic candidate ID generation,
- candidate review artifact includes decision placeholders,
- source text remains unchanged,
- accept decision produces derived reviewed text only,
- reject decision does not alter derived text,
- edit decision uses reviewer-supplied value,
- stale decisions are detected when observed text/span no longer match,
- local decision files are not required for candidate generation,
- JSON export is stable and deterministic.

## Future Backlog

- Build experiment-only decision-template export.
- Add deterministic candidate IDs.
- Add local decision merge support.
- Add reviewed-text derived export.
- Add CSV export for spreadsheet review.
- Add reviewer notes and decision status summaries.
- Add glossary update proposal generation from accepted/rejected decisions.
- Add UI or notebook review flow.
- Consider review-mode metadata integration only after the workflow stabilizes.

## Recommended Next Implementation Pass

Add experiment-only review/export support to the macron candidate helper.

Suggested target:

```text
experiments/review_macron_candidates.py
tests/test_macron_candidate_review_export.py
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-export-design-pass.md
```

First implementation scope:

- deterministic `candidate_id`,
- decision placeholders in the generated candidate report,
- optional empty decision-template export,
- no application of decisions yet,
- no runtime integration,
- no mutation of OCR text.
