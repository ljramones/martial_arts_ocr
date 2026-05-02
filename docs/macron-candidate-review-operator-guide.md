# Macron Candidate Review Operator Guide

## Purpose

Use this guide to run the macron candidate review helper and inspect candidate decisions without changing OCR text.

The workflow is review-only:

```text
OCR/review text sources
  -> glossary-backed candidates
  -> local decisions template
  -> reviewed decisions export
```

The helper never mutates source OCR text.

## Prerequisites

Run from the repository root:

```bash
cd /path/to/martial_arts_ocr
```

Use the normal project virtual environment:

```bash
.venv/bin/python
```

The helper scans existing OCR review summaries when available. If a summary is missing, it records the missing path and continues.

## Run The Candidate Review Helper

Default run:

```bash
.venv/bin/python experiments/review_macron_candidates.py
```

Default output directory:

```text
data/notebook_outputs/macron_candidate_review/
```

Default output files:

```text
summary.json
decisions.local.json
reviewed_decisions.json
```

To regenerate the local decision template from the current candidate set:

```bash
.venv/bin/python experiments/review_macron_candidates.py --overwrite-decisions-template
```

Use that overwrite flag carefully. It replaces any local pending review edits in `decisions.local.json`.

## Custom Inputs

Scan one or more specific summary JSON files:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --summary-json data/notebook_outputs/macron_ocr_eval/summary.json \
  --summary-json data/notebook_outputs/ocr_text_quality_review/summary.json
```

Skip built-in fixture strings:

```bash
.venv/bin/python experiments/review_macron_candidates.py --no-fixtures
```

Write to a different ignored review directory:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --output-dir data/notebook_outputs/macron_candidate_review_custom/
```

## Output Files

### summary.json

Contains the candidate report:

```text
input summary paths
missing summary paths
source counts
candidate counts
candidate summaries
candidate_sources[]
```

Each candidate includes:

```text
candidate_id
observed
candidate
span
context
match_type
requires_review
decision
reviewed_value
reviewer_notes
```

The `decision`, `reviewed_value`, and `reviewer_notes` fields are placeholders in this report.

### decisions.local.json

This is the file to edit manually.

It contains one decision entry per candidate and is local/private by default.

Do not commit it.

### reviewed_decisions.json

This is a derived export produced by the helper.

It separates:

```text
accepted
rejected
deferred
edited
pending
stale
```

It also records:

```json
"source_text_mutated": false
```

The current implementation does not apply decisions to produce corrected text. It only exports reviewed decision records.

## How To Edit decisions.local.json

Open:

```text
data/notebook_outputs/macron_candidate_review/decisions.local.json
```

For each reviewed item, set:

```json
"decision": "accept"
```

or:

```json
"decision": "reject"
```

or:

```json
"decision": "defer"
```

or:

```json
"decision": "edit"
```

For `accept`, set `reviewed_value` to the candidate value:

```json
{
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "decision": "accept",
  "reviewed_value": "Daitō-ryū",
  "reviewer": "local",
  "reviewed_at": "2026-04-28T00:00:00Z",
  "notes": ["accepted as normalized research spelling"]
}
```

For `reject`, leave `reviewed_value` as `null`:

```json
{
  "observed": "BUDO",
  "candidate": "budō",
  "decision": "reject",
  "reviewed_value": null,
  "notes": ["source heading intentionally uses ASCII"]
}
```

For `edit`, set `reviewed_value` to the reviewer-supplied value:

```json
{
  "observed": "Daito ryu",
  "candidate": "Daitō-ryū",
  "decision": "edit",
  "reviewed_value": "Daitō ryū",
  "notes": ["publication omits hyphen in this title"]
}
```

After editing, rerun:

```bash
.venv/bin/python experiments/review_macron_candidates.py
```

The helper preserves the existing decision file by default and updates `reviewed_decisions.json`.

## Decision Values

Allowed values:

| Decision | Meaning |
|---|---|
| `accept` | Candidate is correct for this occurrence. |
| `reject` | Candidate is not correct for this occurrence. |
| `defer` | Candidate needs more context or review. |
| `edit` | Candidate is close, but the reviewer supplies a different value. |

Blank or `null` means pending.

## Stale Decisions

A stale decision is a decision whose `candidate_id` no longer appears in the current candidate report.

This can happen when:

- input OCR summaries changed,
- candidate spans changed,
- glossary variants changed,
- the source field changed,
- a decision file from an older run is reused.

Stale decisions are preserved in `reviewed_decisions.json` under `stale_decisions`.

Do not apply stale decisions without checking the source text again.

## What Not To Commit

Do not commit:

```text
data/notebook_outputs/
data/runtime/
data/notebook_outputs/macron_candidate_review/summary.json
data/notebook_outputs/macron_candidate_review/decisions.local.json
data/notebook_outputs/macron_candidate_review/reviewed_decisions.json
private corpus images
bulk OCR outputs
```

Safe files to commit are usually:

```text
docs/
experiments/
tests/
data/evaluation/real_page_review/notes/*.md
```

Only commit exported review examples when source text and provenance are safe to share.

## Troubleshooting

### The helper reports missing summary paths

This is expected if earlier OCR review experiments have not been run locally.

Either run the relevant OCR review experiment first, or pass specific summaries with:

```bash
--summary-json path/to/summary.json
```

### My decisions disappeared

Check whether you used:

```bash
--overwrite-decisions-template
```

That flag regenerates `decisions.local.json`.

By default, the helper preserves an existing decision file.

### reviewed_decisions.json shows everything pending

The decision file probably still has `decision: null` for every candidate.

Edit `decisions.local.json`, then rerun the helper without overwrite.

### reviewed_decisions.json shows stale decisions

The decision file contains IDs not found in the current candidate report.

Review the `stale_decisions` section and decide whether to discard or manually reconcile those decisions.

### I need corrected text

The current helper does not apply decisions to produce corrected text.

That should be a separate derived review-artifact pass. It must still preserve:

```json
"source_text_mutated": false
```

## Current Guardrail

Do not silently normalize OCR text.

The candidate/review workflow exists to support human review and future export, not automatic replacement.
