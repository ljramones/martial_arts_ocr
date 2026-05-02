# Macron Candidate Case Suggestion Review

## Purpose

Review a small pending synthetic batch using the new `case_pattern` and `reviewed_value_suggestion` metadata.

Questions:

```text
- do reviewed_value_suggestion values reduce manual editing?
- are casing suggestions correct?
- are any suggestions misleading?
- do we need title-case / mixed-case handling changes?
```

## Setup

Before the review, the pending synthetic queue did not expose the new suggestion fields directly.

Small ergonomics fix made in this pass:

```text
experiments/review_macron_candidates.py
tests/test_macron_candidate_review_experiment.py
```

The Markdown/CSV review queues now include:

```text
reviewed_value_suggestion
case_pattern
```

This keeps the suggestion visible where review happens.

## Queue Reviewed

Command:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --filter pending \
  --source-filter synthetic \
  --sort candidate \
  --limit 50
```

Queue:

```text
data/notebook_outputs/macron_candidate_review/review_queue_pending_synthetic.csv
```

Reviewed the first 25 pending synthetic candidates from the queue.

## Decisions

All 25 reviewed candidates were accepted using `reviewed_value_suggestion`.

Updated export counts:

```text
accept: 57
reject: 0
defer: 0
edit: 8
pending: 347
stale: 0
reviewed: 65
source_text_mutated: false
```

The pending count dropped from 372 to 347.

## Candidate Patterns Reviewed

The batch focused mostly on `aikijūjutsu` variants from synthetic OCR outputs:

| Observed | Candidate | case_pattern | reviewed_value_suggestion | Decision |
|---|---|---|---|---|
| `aikijdjutsu` | `aikijūjutsu` | `lowercase` | `aikijūjutsu` | accept |
| `aikijGjutsu` | `aikijūjutsu` | `mixed` | `aikijūjutsu` | accept |
| `aikijujutsu` | `aikijūjutsu` | `lowercase` | `aikijūjutsu` | accept |
| `aikiytjutsu` | `aikijūjutsu` | `lowercase` | `aikijūjutsu` | accept |

The contexts were synthetic sentence or term-list outputs, for example:

```text
and Daité-rya. The d6j6 taught aikijdjutsu and iaidd.
```

and:

```text
dojo ryaha soke iaid6 kenjutsu aikijGjutsu
```

## Findings

The suggestions reduced manual editing for this batch:

- lowercase observed variants suggested lowercase macronized values,
- mixed OCR-confusion forms such as `aikijGjutsu` suggested the canonical lowercase value,
- reviewed values could be copied directly from `reviewed_value_suggestion`,
- no source text was changed.

No misleading suggestions were found in the reviewed synthetic batch.

The `mixed` case pattern behaved acceptably here because the mixed casing came from OCR corruption, not meaningful title casing.

## Title / Mixed Case Handling

Current behavior remains appropriate:

```text
uppercase observed text -> uppercase suggestion
titlecase observed text -> titlecase suggestion
lowercase observed text -> canonical/lowercase suggestion
mixed observed text -> canonical glossary suggestion
```

No change is needed for this batch.

The previous real OCR review remains the important uppercase case:

```text
BUDO -> reviewed_value_suggestion: BUDŌ
```

## Ergonomics Finding

Review queues must expose suggestion fields. Without them, reviewers have to cross-reference `summary.json` or infer casing manually.

The queue export now includes:

```text
reviewed_value_suggestion
case_pattern
```

This should remain part of the operator workflow.

## Decision

- [x] Case suggestions were useful.
- [x] No misleading suggestions found in this batch.
- [x] No title-case or mixed-case rule change needed now.
- [x] Queue output needed and received suggestion columns.
- [x] Source text remained unmutated.

## Recommended Next Pass

Implement duplicate grouping and source labels before reviewing many more candidates:

```text
- group repeated observed/candidate/context patterns
- show occurrence count
- include sample_id and source image path when available
- preserve all candidate IDs for decision export
```

Do not add automatic normalization.
