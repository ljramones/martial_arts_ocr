# Macron Candidate Workflow State — 2026-04-28

## Status

The macronized romanization workflow is now a usable review-only loop.

Current flow:

```text
OCR/review text sources
  -> glossary-backed candidates
  -> case-aware reviewed value suggestions
  -> deterministic candidate IDs
  -> local decisions template
  -> filtered/sorted Markdown and CSV queues
  -> reviewed decision export
```

The invariant remains:

```text
source_text_mutated=false
```

No automatic normalization is performed.

## Stable / Accepted

- OCR engines tested so far do not reliably preserve macrons.
- Cleanup and serialization preserve macrons when OCR emits them.
- Candidate generation is glossary-backed and conservative.
- Every candidate remains review-required.
- Candidate IDs are deterministic.
- Local decisions are stored outside committed artifacts by default.
- Reviewed exports separate accepted, rejected, deferred, edited, pending, and stale decisions.
- Review queues can be filtered and sorted for human review.
- Case-aware suggestions improve review speed but do not replace reviewer decisions.

## Candidate Generation Behavior

Candidate generation lives in:

```text
utils/text/macron_candidates.py
```

It supports:

```text
variant_exact
variant_hyphen_space
observed_ocr_confusion
```

It is intentionally conservative:

- no broad fuzzy matching,
- no automatic source text mutation,
- no replacement candidate when text already contains the canonical macronized form,
- word-boundary aware matching,
- OCR-confusion candidates only when listed in the glossary.

## Review / Export Behavior

Review/export support lives in:

```text
experiments/review_macron_candidates.py
```

Generated local outputs:

```text
data/notebook_outputs/macron_candidate_review/summary.json
data/notebook_outputs/macron_candidate_review/decisions.local.json
data/notebook_outputs/macron_candidate_review/reviewed_decisions.json
data/notebook_outputs/macron_candidate_review/review_queue_*.md
data/notebook_outputs/macron_candidate_review/review_queue_*.csv
```

These files are ignored and should not be committed.

Supported queue options:

```text
--filter pending|accepted|rejected|deferred|edited|reviewed|stale|all
--source-filter fixture|summary_json|real_ocr|synthetic|macron_eval|all
--sort source|candidate|observed|decision|match_type
--limit N
```

## Decision Values

| Decision | Meaning |
|---|---|
| `accept` | Candidate is correct for this occurrence. |
| `reject` | Candidate is not correct for this occurrence. |
| `defer` | Candidate needs more context or source review. |
| `edit` | Candidate is semantically useful, but reviewer supplies a different value. |

Blank or `null` means pending.

## Case-Aware Suggestions

Candidates now include:

```text
case_pattern
reviewed_value_suggestion
```

Example:

```json
{
  "observed": "BUDO",
  "candidate": "budō",
  "case_pattern": "uppercase",
  "reviewed_value_suggestion": "BUDŌ"
}
```

The canonical `candidate` remains the glossary form. The reviewed value suggestion is advisory only.

Observed behavior:

- uppercase source terms suggest uppercase macronized review values,
- lowercase source terms suggest canonical lowercase values,
- mixed OCR-corruption casing falls back to canonical glossary form,
- no misleading suggestions were found in the reviewed synthetic batch.

## Current Review Totals

Current reviewed export state:

```text
accepted: 57
rejected: 0
deferred: 0
edited: 8
pending: 347
stale: 0
reviewed total: 65
source_text_mutated: false
```

Reviewed batches so far:

- first 40-candidate batch,
- deferred real OCR `BUDO -> budō` review,
- 25-candidate synthetic batch using case-aware suggestions.

## Important Evidence

The real OCR review of `original_img_3288` showed why review is required:

```text
source OCR: BUDO
candidate: budō
reviewed value: BUDŌ
decision: edit
```

The source image showed an all-caps title/list item:

```text
1. BUJUTSU AND BUDO.
```

This proves the system must not blindly replace ASCII forms with lowercase canonical forms.

## Known Limitations

- Most remaining pending items are synthetic or fixture-derived.
- Real OCR candidate evidence remains sparse.
- Review queues still repeat duplicate candidate occurrences across related summaries.
- Queue rows do not yet group duplicates.
- Queue rows do not always show human-friendly sample IDs or source image paths.
- There is no applied reviewed-text artifact yet.
- There is no UI.
- There is no glossary feedback export yet.

## Do Not Change Right Now

- Do not add automatic normalization.
- Do not mutate OCR source text.
- Do not promote macron candidates into canonical Japanese/romanization fields.
- Do not add broad fuzzy matching.
- Do not build a UI before grouping/source-label ergonomics are clearer.
- Do not spend much more effort reviewing synthetic-only candidates unless testing workflow behavior.

## Recommended Next Branch Options

1. Continue manual review only when real OCR candidates appear.

2. Improve review queue grouping:

```text
group repeated observed/candidate/context patterns
show occurrence count
preserve all candidate IDs
include source path and sample_id when available
```

3. Build a small review/export convenience tool only after grouping is in place.

4. Return to broader Japanese OCR region evaluation or corpus sampling.

## Recommended Immediate Next Step

Pause additional macron workflow implementation.

The workflow has enough capability for current evidence. The highest-value next work is not more macron logic; it is either:

- gather more real OCR candidates from broader corpus sampling,
- improve duplicate grouping/source labels if reviewing many more candidates,
- return to Japanese OCR region/corpus evaluation.

## Relevant Files

```text
utils/text/macron_candidates.py
experiments/review_macron_candidates.py
tests/test_macron_normalization_candidates.py
tests/test_macron_candidate_review_experiment.py
docs/macron-normalization-candidate-design.md
docs/macron-candidate-review-workflow-design.md
docs/macron-candidate-review-operator-guide.md
```

## Relevant Review Notes

```text
data/evaluation/real_page_review/notes/2026-04-28-macron-ocr-sample-curation.md
data/evaluation/real_page_review/notes/2026-04-28-macron-ocr-engine-comparison.md
data/evaluation/real_page_review/notes/2026-04-28-macron-normalization-candidate-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-artifact-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-export-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-first-review-batch.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-ergonomics-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-deferred-real-ocr-review.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-case-suggestion-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-case-suggestion-review.md
```
