# Macron Candidate Review Artifact Pass

## Purpose

Verify that glossary-backed macron candidate detection can run over OCR review artifacts and produce useful review summaries without mutating source text, changing `DocumentResult`, or changing runtime OCR behavior.

## Scope

This pass is review-artifact only.

It scans:

- existing ignored OCR review summaries under `data/notebook_outputs/`
- prior synthetic macron OCR experiment summaries
- controlled fixture strings

It does not:

- replace OCR text
- promote canonical Japanese or romanization fields
- change OCR defaults
- change cleanup or serialization behavior
- write generated outputs outside ignored review-output paths

## Helper

Added:

```text
experiments/review_macron_candidates.py
```

Default ignored output:

```text
data/notebook_outputs/macron_candidate_review/summary.json
```

The helper:

- extracts OCR text-like fields such as `readable_text`, `raw_output`, `cleaned_output`, `serialized_text`, and text previews from summary JSON files
- adds controlled fixture strings for ASCII variants, listed OCR-confusion variants, and canonical macron controls
- calls `utils.text.macron_candidates.find_macron_normalization_candidates`
- writes compact candidate summaries with source IDs, contexts, spans, match types, and `requires_review=true`

## Inputs Scanned

Existing summaries found:

```text
data/notebook_outputs/macron_ocr_eval/summary.json
data/notebook_outputs/macron_ocr_engine_comparison/summary.json
data/notebook_outputs/ocr_text_quality_review/summary.json
data/notebook_outputs/ocr_text_readability_sampling/eng_auto/summary.json
data/notebook_outputs/ocr_text_reading_order_after_line_grouping/eng_auto/summary.json
data/notebook_outputs/document_result_serialization_review/summary.json
```

Controlled fixtures:

```text
fixture_ascii_variants
fixture_ocr_confusions
fixture_canonical_control
```

## Summary Result

```text
text sources scanned: 574
sources with candidates: 68
candidate count: 412
```

Candidate counts by canonical term:

| Candidate | Count |
|---|---:|
| Daitō-ryū | 56 |
| aikijūjutsu | 35 |
| budō | 68 |
| dōjō | 43 |
| iaidō | 48 |
| jūjutsu | 28 |
| koryū | 51 |
| ryūha | 37 |
| sōke | 46 |

Candidate counts by match type:

| Match Type | Count |
|---|---:|
| variant_exact | 308 |
| observed_ocr_confusion | 104 |

Candidate counts by source type:

| Source Type | Count |
|---|---:|
| fixture | 10 |
| summary_json | 402 |

## Real OCR Artifact Findings

The real OCR review summaries produced a small number of candidate hits, mostly repeated across related summaries:

```text
observed: BUDO
candidate: budō
match_type: variant_exact
context: BUJUTSU AND BUDO
```

This is useful as a review artifact because it identifies a domain term that may need a macron. It also demonstrates why the layer must remain candidate-only: the source title may intentionally use ASCII `BUDO`, and a human reviewer should decide whether to accept `budō`.

No source text was changed.

## Synthetic OCR Artifact Findings

The prior synthetic macron OCR experiments produced many useful candidate hits from OCR-stripped and OCR-corrupted terms, for example:

```text
koryu -> koryū
budo -> budō
Daito-ryu -> Daitō-ryū
jujutsu -> jūjutsu
dojo -> dōjō
ryuha -> ryūha
soke -> sōke
iaid6 -> iaidō
aikijGjutsu -> aikijūjutsu
```

These outputs confirm that the candidate generator can recover useful review suggestions from both ASCII-stripped OCR and explicitly listed OCR-confusion forms.

## Fixture Findings

The fixture scan behaved as intended:

- ASCII variants emitted review-required candidates.
- Known OCR-confusion variants emitted review-required candidates only because they are listed in the glossary.
- Already-canonical macron text emitted no replacement candidates.

## Artifact Shape

Each candidate includes:

```json
{
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [0, 9],
  "context": "Daito-ryu aikijujutsu appears beside kory",
  "source": "martial_arts_macron_glossary",
  "match_type": "variant_exact",
  "requires_review": true,
  "confidence": "candidate",
  "term_category": "style",
  "ambiguous": false,
  "notes": []
}
```

The review summary also records:

```text
source_id
source_type
source_path
field_path
text_preview
candidate_count
candidate_summary
```

## Decision

- [x] Candidate review artifacts are useful.
- [x] Source OCR text remains unchanged.
- [x] Candidates remain `requires_review=true`.
- [x] Synthetic OCR outputs produce useful candidate reports.
- [x] Real OCR outputs can produce candidate reports, but current real evidence remains sparse.
- [x] Do not integrate into runtime yet.
- [x] Do not add automatic normalization.

## Recommended Next Implementation Pass

Add an opt-in review/export report that can combine:

```text
readable_text
normalization candidates
source path / page ID
candidate context
accept/reject placeholders
```

This should remain outside runtime defaults until a reviewer workflow exists.

## Verification

The helper was run successfully:

```bash
.venv/bin/python experiments/review_macron_candidates.py
```

Generated output remained under ignored:

```text
data/notebook_outputs/macron_candidate_review/
```
