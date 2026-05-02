# Macron Candidate First Review Batch

## Purpose

Exercise the macron candidate review workflow on a small batch before building more review tooling.

Goal:

```text
candidate generator
  -> decisions.local.json
  -> reviewer decisions
  -> rerun helper
  -> reviewed_decisions.json
  -> pending count drops
  -> stale count remains sane
  -> source_text_mutated=false
```

## Inputs

Candidate report:

```text
data/notebook_outputs/macron_candidate_review/summary.json
```

Local decision file:

```text
data/notebook_outputs/macron_candidate_review/decisions.local.json
```

Reviewed export:

```text
data/notebook_outputs/macron_candidate_review/reviewed_decisions.json
```

All generated files are ignored and were not staged.

## Batch Selection

Reviewed 40 candidates total:

- 10 controlled fixture candidates,
- 22 synthetic macron OCR fixture candidates,
- 8 repeated real OCR `BUDO -> budō` candidates from review summaries.

This was intentionally a workflow validation batch, not a full correction pass.

## Review Decisions

After editing `decisions.local.json` and rerunning:

```bash
.venv/bin/python experiments/review_macron_candidates.py
```

Export counts:

```text
accept: 32
reject: 0
defer: 8
edit: 0
pending: 372
stale: 0
reviewed: 40
source_text_mutated: false
```

The pending count dropped from 412 to 372.

No stale decisions appeared.

## Accepted Examples

Controlled fixture examples:

| Observed | Candidate | Decision | Reason |
|---|---|---|---|
| `Daito-ryu` | `Daitō-ryū` | accept | Controlled fixture; expected candidate. |
| `aikijujutsu` | `aikijūjutsu` | accept | Controlled fixture; expected candidate. |
| `koryu` | `koryū` | accept | Controlled fixture; expected candidate. |
| `budo` | `budō` | accept | Controlled fixture; expected candidate. |
| `jujutsu` | `jūjutsu` | accept | Controlled fixture; expected candidate. |

OCR-confusion fixture examples:

| Observed | Candidate | Decision | Reason |
|---|---|---|---|
| `koryG` | `koryū` | accept | Listed OCR-confusion fixture. |
| `bud6` | `budō` | accept | Listed OCR-confusion fixture. |
| `Dait6-rya` | `Daitō-ryū` | accept | Listed OCR-confusion fixture. |
| `d6j6` | `dōjō` | accept | Listed OCR-confusion fixture. |
| `aikijGjutsu` | `aikijūjutsu` | accept | Listed OCR-confusion fixture. |

Synthetic OCR examples:

```text
koryG -> koryū
bud6 -> budō
Dait6-rya -> Daitō-ryū
jGjutsu -> jūjutsu
dojo -> dōjō
ryaha -> ryūha
soke -> sōke
iaid6 -> iaidō
aikijGjutsu -> aikijūjutsu
```

These were accepted only as synthetic review-loop validation examples.

## Deferred Examples

The real OCR candidates were all deferred:

```text
BUDO -> budō
context: BUJUTSU AND BUDO
```

Reason:

```text
Real OCR title/context needs source image or publication-style review before accepting macron normalization.
```

This is the right outcome for the first batch. `BUDO` may be an all-caps title or an intentional ASCII publication style. The candidate is useful, but it should not be accepted without image/source review.

## Bad / Ambiguous Candidate Patterns

No clear false-positive glossary bug was found in this small batch.

The main ambiguity is publication style:

- uppercase headings may intentionally omit macrons,
- source typography may intentionally use ASCII,
- title capitalization can make direct candidate acceptance unsafe.

The `BUDO -> budō` candidates demonstrate that candidate context is enough to flag the issue, but not enough to accept the correction confidently.

## Glossary Gaps Found

No new glossary terms were added from this batch.

The batch did show that known OCR-confusion forms are useful for synthetic OCR output:

```text
koryG
bud6
Dait6-rya
jGjutsu
iaid6
aikijGjutsu
```

These should remain explicitly listed, not generalized into broad fuzzy matching.

## Workflow Usability

The JSON review loop is usable for a small batch:

- candidate IDs are stable enough to edit decisions,
- `decisions.local.json` is straightforward to update,
- rerunning the helper preserves existing decisions by default,
- `reviewed_decisions.json` cleanly separates reviewed, pending, and stale decisions,
- `source_text_mutated=false` remains explicit.

Pain points:

- repeated candidate instances appear across related summaries,
- there is no sorting/filtering mode for "real OCR only" or "first N candidates",
- context is useful but sometimes not enough without page/image reference,
- editing raw JSON works for 25-50 candidates but will not scale comfortably.

## Candidate Context Assessment

Context was enough to:

- accept controlled/synthetic examples,
- defer real `BUDO -> budō` title candidates,
- verify that no source text was being changed.

Context was not enough to:

- decide whether all-caps source headings should be normalized,
- distinguish publication convention from OCR loss,
- confidently accept real OCR candidates without source image review.

## Decision

- [x] First batch review workflow is usable.
- [x] Pending count dropped.
- [x] Stale count stayed at zero.
- [x] Local decisions remained ignored.
- [x] Source text was not mutated.
- [x] More review tooling should focus on filtering/sorting and context, not automatic normalization.

## Recommended Next Pass

Improve review ergonomics before reviewing many more candidates:

```text
- add filtering/sorting for candidate reports
- support --limit / --decision-filter / --source-filter
- optionally group duplicate observed/candidate/context patterns
- add source/page reference columns where available
```

Do not build a full UI yet.

Do not add automatic normalization.
