# Macron Candidate Review Export Pass

## Purpose

Add experiment-only review/export support for macron normalization candidates while keeping OCR text immutable and runtime behavior unchanged.

This pass implements the first workflow step described in:

```text
docs/macron-candidate-review-workflow-design.md
```

## Scope

This is still review-artifact only.

It does not:

- apply accepted decisions to OCR text,
- mutate `raw_text`, `readable_text`, `text.txt`, or `data.json["text"]`,
- add runtime integration,
- add canonical Japanese or romanization fields,
- change OCR defaults,
- change cleanup or serialization behavior.

## Implementation

Updated:

```text
experiments/review_macron_candidates.py
tests/test_macron_candidate_review_experiment.py
```

The helper now emits:

```text
data/notebook_outputs/macron_candidate_review/summary.json
data/notebook_outputs/macron_candidate_review/decisions.local.json
data/notebook_outputs/macron_candidate_review/reviewed_decisions.json
```

All generated files remain under ignored `data/notebook_outputs/`.

## Candidate IDs

Each candidate now gets a deterministic ID:

```text
sha256:<hex>
```

Identity inputs:

```text
source_id
source_path
field_path
span
observed
candidate
match_type
```

Example:

```json
{
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [0, 9],
  "candidate_id": "sha256:eb3dbe6271032d87adcdcd3638b91532c89f789b7bd8cfa7d27a0755ed5088fb",
  "decision": null,
  "reviewed_value": null,
  "reviewer_notes": []
}
```

## Decisions Template

The helper creates a local decisions template:

```text
data/notebook_outputs/macron_candidate_review/decisions.local.json
```

The template contains one entry per candidate:

```json
{
  "candidate_id": "sha256:...",
  "source_id": "fixture_ascii_variants",
  "source_path": null,
  "field_path": null,
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [0, 9],
  "context": "Daito-ryu aikijujutsu appears beside kory",
  "match_type": "variant_exact",
  "decision": null,
  "reviewed_value": null,
  "reviewer": null,
  "reviewed_at": null,
  "notes": []
}
```

Supported future decision values:

```text
accept
reject
defer
edit
```

Existing local decision files are preserved by default. The helper only overwrites them when `--overwrite-decisions-template` is passed.

## Reviewed Export

The helper also writes:

```text
data/notebook_outputs/macron_candidate_review/reviewed_decisions.json
```

This export separates reviewed, pending, and stale decisions.

Initial run:

```json
{
  "schema_version": "macron_candidate_review_export.v1",
  "source_text_mutated": false,
  "counts": {
    "accept": 0,
    "reject": 0,
    "defer": 0,
    "edit": 0,
    "pending": 412,
    "stale": 0
  },
  "pending_decision_count": 412
}
```

Because no reviewer decisions were filled in yet, all candidates are pending.

## Run Result

Command:

```bash
.venv/bin/python experiments/review_macron_candidates.py --overwrite-decisions-template
```

Result:

```text
text sources scanned: 574
sources with candidates: 68
candidate count: 412
decisions template entries: 412
reviewed decisions: 0
pending decisions: 412
stale decisions: 0
```

## Tests Added / Updated

Updated:

```text
tests/test_macron_candidate_review_experiment.py
```

Coverage now includes:

- CLI review/export options,
- deterministic candidate IDs,
- source-sensitive candidate IDs,
- decision placeholders in candidate reports,
- local decisions template shape,
- existing local decision files preserved by default,
- reviewed export separation of accepted, rejected, pending, and stale decisions,
- source text unchanged.

## Decision

- [x] Review/export support is useful enough for the next workflow step.
- [x] Candidate IDs are deterministic.
- [x] Local decision templates are generated under ignored output paths.
- [x] Existing local decisions are not overwritten by default.
- [x] Reviewed decisions are exported separately.
- [x] Source text is not mutated.
- [x] Runtime/OCR/extraction/serialization behavior remains unchanged.

## Remaining Work

Next implementation should add decision application as a derived review artifact only:

```text
original readable_text
accepted/edit decisions
reviewed_text
applied decision IDs
source_text_mutated=false
```

Do not overwrite canonical OCR artifacts.

## Verification

Focused test:

```bash
.venv/bin/python -m pytest -q tests/test_macron_candidate_review_experiment.py
```

Result:

```text
9 passed
```
