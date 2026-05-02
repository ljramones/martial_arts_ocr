# Macron Candidate Review Ergonomics Pass

## Purpose

Improve the experiment-only macron candidate review workflow after the first 40-candidate review batch showed that raw JSON editing works but needs better filtering, sorting, and context display.

This pass does not change candidate generation rules, OCR behavior, runtime behavior, or source text.

## Implementation

Updated:

```text
experiments/review_macron_candidates.py
tests/test_macron_candidate_review_experiment.py
```

Added review queue options:

```text
--filter
--source-filter
--sort
--limit
```

Supported decision filters:

```text
all
pending
accepted
rejected
deferred
edited
reviewed
stale
```

Supported source filters:

```text
all
fixture
summary_json
real_ocr
synthetic
macron_eval
```

Supported sort keys:

```text
source
candidate
observed
decision
match_type
```

## Review Queue Outputs

The helper now writes reviewer-friendly Markdown and CSV queues under the ignored review directory:

```text
data/notebook_outputs/macron_candidate_review/review_queue_<filter>_<source-filter>.md
data/notebook_outputs/macron_candidate_review/review_queue_<filter>_<source-filter>.csv
```

Each row includes:

```text
candidate_id
decision
source_kind
source_id
source_path
field_path
observed
candidate
match_type
context
reviewed_value
notes
```

The generated outputs remain ignored.

## Commands Run

Pending real OCR queue:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --filter pending \
  --source-filter real_ocr \
  --sort candidate \
  --limit 50
```

Deferred real OCR queue:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --filter deferred \
  --source-filter real_ocr \
  --sort source
```

Pending synthetic queue:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --filter pending \
  --source-filter synthetic \
  --sort match_type \
  --limit 25
```

## Results

Current decision state after first review batch:

```text
accept: 32
reject: 0
defer: 8
edit: 0
pending: 372
stale: 0
source_text_mutated: false
```

Queue checks:

```text
pending real OCR queue: 0 candidates
deferred real OCR queue: 8 candidates
pending synthetic queue: 25 candidates in limited Markdown/CSV views
```

CSV row counts were verified with Python's `csv.DictReader` because quoted context fields can contain embedded newlines and make `wc -l` misleading.

Repeated runs overwrite same-named queue files for the same filter/source pair, so operators should treat queue files as generated snapshots.

## Example Deferred Queue

The deferred real OCR queue isolates the repeated real `BUDO -> budō` cases:

```text
observed: BUDO
candidate: budō
source_kind: real_ocr
context: BUJUTSU AND BUDO
decision: defer
notes: Real OCR title/context needs source image or publication-style review before accepting macron normalization.
```

This is more usable than scanning `decisions.local.json` directly.

## Ergonomics Findings

Improved:

- source type is explicit as `fixture`, `synthetic`, or `real_ocr`;
- pending/deferred/reviewed queues can be isolated;
- sorting by candidate/observed/match type makes repeated patterns visible;
- Markdown is easy to skim;
- CSV is better for spreadsheet review;
- context and notes are visible without opening raw JSON.

Remaining pain points:

- repeated candidate instances across related summaries still appear separately;
- queue files are regenerated snapshots and can overwrite earlier queues with the same name;
- source IDs are useful but not human-friendly sample/page labels;
- source image or page artifact links are not included yet;
- stale decision queue is not populated from stale-only records yet.

## Decision

- [x] Review ergonomics improved enough for another small batch.
- [x] No candidate generation rules changed.
- [x] No source text mutation was introduced.
- [x] Generated queue artifacts remain ignored.
- [x] Next work should focus on duplicate grouping and better source/page labels before a UI.

## Recommended Next Pass

Add duplicate grouping and better source labels:

```text
- group identical observed/candidate/context patterns
- include occurrence count
- include source paths / field paths compactly
- optionally emit group-level review queues
```

Do not build a full UI yet.

Do not add automatic normalization.
