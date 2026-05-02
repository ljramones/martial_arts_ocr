# Macron Normalization Candidate Design

## Status

OCR engines tested so far did not preserve macrons from controlled synthetic fixtures.

Current evidence:

- Tesseract `eng`, `eng+jpn`, and `jpn` stripped or misread macron-bearing romanized Japanese terms.
- EasyOCR ran with local Japanese models but did not recover the Latin romanized terms.
- PaddleOCR was not evaluated as OCR because `.venv-eval` import failed on local PaddleX cache permissions.
- Cleanup and serialization preserve macrons when they are already present in text.
- Confirmed real scanned-page samples with visible macronized romanization are still missing.

Therefore this design is review-only and candidate-based. It must not silently rewrite OCR text.

## Problem Statement

Martial arts research material often uses romanized Japanese terms with macrons.

OCR may emit ASCII approximations or corrupted terms:

```text
koryū -> koryu / koryG
budō -> budo / bud6
Daitō-ryū -> Daito-ryu / Dait6-rya
jūjutsu -> jujutsu / jGjutsu
dōjō -> dojo / d6j6
```

The system should suggest likely corrections for review, while preserving the source OCR text exactly as observed.

## Evidence Summary

The macron OCR sample curation pass showed:

- no confirmed real image samples with visible macrons,
- synthetic fixture OCR did not preserve macrons,
- direct cleanup/serialization controls preserved all expected macron terms.

The macron OCR engine comparison showed:

- no tested engine preserved macron-bearing terms,
- Tesseract produced the most useful Latin text but stripped/misread macrons,
- EasyOCR did not recover the Latin terms,
- PaddleOCR remains unevaluated for this question.

The evidence points to a candidate generation layer, not automatic normalization.

## Non-Goals

- No automatic normalization.
- No silent OCR text mutation.
- No canonical Japanese or romanization field promotion yet.
- No broad romanization engine.
- No machine translation.
- No model training or fine-tuning.
- No runtime behavior changes in the initial pass.
- No global `koryu -> koryū` rewrite without context and review.

## Candidate Layer Principles

- Preserve original OCR text.
- Generate auditable candidates.
- Require human review.
- Prefer high-precision glossary matches over broad fuzzy matching.
- Keep spans and surrounding context.
- Mark ambiguity explicitly.
- Avoid changing `readable_text`, `raw_text`, or serialized source text.
- Support a future accept/reject workflow.
- Make candidate generation deterministic and testable.

## Glossary Format

Recommended tracked example location:

```text
data/glossaries/martial_arts_terms/macron_terms.example.json
```

Local reviewed glossaries can live beside it as ignored files later, for example:

```text
data/glossaries/martial_arts_terms/macron_terms.local.json
```

Recommended JSON shape:

```json
{
  "terms": [
    {
      "canonical": "Daitō-ryū",
      "variants": ["Daito-ryu", "Daito ryu", "Daitoryu", "Daitō ryu"],
      "ocr_confusions": ["Dait6-rya", "Daito-ryt"],
      "category": "style",
      "language": "ja-Latn",
      "requires_review": true,
      "notes": "Candidate only; requires review"
    }
  ]
}
```

Glossary policy:

- Small reviewed example glossaries can be tracked.
- Large or copyrighted terminology lists should stay local/private until provenance is clear.
- Every canonical form and variant should be human-reviewed before use.
- OCR confusion forms should be added only when observed in project outputs or controlled experiments.

## Initial Glossary Seed

Initial seed terms for a reviewed example glossary:

| Canonical | ASCII Variants | Notes |
|---|---|---|
| `koryū` | `koryu` | Classical school/tradition term. |
| `budō` | `budo` | Martial way; common term. |
| `Daitō-ryū` | `Daito-ryu`, `Daito ryu`, `Daitoryu` | Style/school name; preserve hyphen/macrons only as candidate. |
| `jūjutsu` | `jujutsu`, `ju-jutsu` | Art term; spelling variants may be historically meaningful. |
| `dōjō` | `dojo` | Training hall. |
| `ryūha` | `ryuha` | School/lineage term. |
| `sōke` | `soke` | Title/head-family term. |
| `iaidō` | `iaido` | Art term. |
| `aikijūjutsu` | `aikijujutsu`, `aiki-jujutsu` | Art term; hyphen variants need review. |
| `kenjutsu` | `kenjutsu` | Generally does not require a macron. Do not invent one. |

Caution:

- The glossary must be reviewed.
- Do not add macrons to terms that do not take them.
- Some historical spellings or publication conventions may intentionally omit macrons.

## Candidate Generation Rules

Use conservative rules:

1. Exact ASCII variant match.
2. Case-insensitive match while preserving the observed form.
3. Hyphen/space tolerant match only for known glossary variants.
4. Optional light OCR-confusion handling only for known observed failures, such as `koryG -> koryū` or `Dait6-rya -> Daitō-ryū`.
5. No broad edit-distance fuzzy matching by default.
6. Word-boundary aware matching to avoid substrings inside unrelated words.
7. Do not emit replacement candidates when the observed text already contains the canonical macronized form.
8. Do not mutate the input string.

Examples:

```text
Observed: Daito-ryu
Candidate: Daitō-ryū
Match type: variant_exact

Observed: Daito ryu
Candidate: Daitō-ryū
Match type: variant_hyphen_space

Observed: Dait6-rya
Candidate: Daitō-ryū
Match type: observed_ocr_confusion
```

## Ambiguity and Review Requirements

Every candidate should initially include:

```text
requires_review: true
confidence: candidate
```

Handle ambiguity explicitly:

- If one observed string maps to multiple canonical terms, emit multiple candidates with `ambiguous: true`.
- If the term appears inside a proper name or title, preserve context and require review.
- If OCR confidence is low, include confidence metadata but do not suppress the candidate automatically.
- If surrounding context is unclear, keep the candidate but mark context uncertainty.
- If the term already contains a macron, do not emit a replacement candidate; optionally record that it is already canonical.

Potential ambiguity examples:

```text
ryu:
  could be a stripped form of ryū,
  but can also be part of a name or non-Japanese text.

dojo:
  usually dōjō in romanized Japanese,
  but may appear intentionally without macrons in publication style.
```

## Metadata Shape

Candidate metadata should be compact and auditable:

```json
{
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [124, 132],
  "context": "the Daito-ryu tradition...",
  "source": "martial_arts_macron_glossary",
  "match_type": "variant_exact",
  "requires_review": true,
  "confidence": "candidate",
  "term_category": "style",
  "ambiguous": false,
  "notes": []
}
```

Possible future locations:

```text
PageResult.metadata["normalization_candidates"]
DocumentResult.metadata["normalization_candidates"]
review artifact only
```

Do not implement model/schema changes in the design pass. The first implementation should keep candidates in experiment/review output.

## Integration Points

Future integration should be phased:

1. Experiment helper for OCR text review.
2. Review-mode report that lists candidates without changing text.
3. Optional `DocumentResult` metadata in review mode only.
4. Review/export workflow for accept/reject decisions.
5. Canonical Japanese/romanization fields only after reviewed corrections and real OCR evidence exist.

Initial integration target:

```text
experiment-only utility
  -> input text or summary JSON
  -> normalization_candidates[]
  -> review note/report
```

## Test Strategy

Tests for the next implementation pass:

- Exact variant creates a candidate.
- Canonical text with macron does not create a replacement candidate.
- Word-boundary matching avoids false positives inside unrelated words.
- Hyphen/space variants match only when present in glossary variants.
- Known OCR confusion forms create review-required candidates.
- Ambiguous candidate lists remain review-required.
- Context span is preserved.
- Source text is not mutated.
- Japanese/macron cleanup tests remain unchanged.

Suggested test examples:

```text
"Daito-ryu Aiki-jujutsu" -> candidates for Daitō-ryū and jūjutsu/aikijūjutsu as configured
"koryū budō" -> no replacement candidates
"embudoed" -> no budō candidate
"Dait6-rya" -> candidate only if listed as observed OCR confusion
```

## Future Backlog

- Curated real macron-bearing source pages.
- Reviewer accept/reject workflow.
- Glossary expansion.
- Term categories:
  - art
  - rank
  - school
  - place
  - person/title
- Optional confidence scoring.
- OCR confusion lists derived from project outputs.
- Integration with `JapaneseProcessor` later.
- Export reviewed normalization decisions as training/review data.

## Recommended Next Implementation Pass

Implement an experiment-only macron candidate generator utility and tests.

Suggested files:

```text
utils/text/macron_candidates.py
tests/test_macron_normalization_candidates.py
data/evaluation/real_page_review/notes/2026-04-28-macron-normalization-candidate-pass.md
```

The initial implementation should:

- include a tiny reviewed in-code or fixture glossary,
- produce `normalization_candidates`,
- preserve source text,
- require review for every candidate,
- avoid runtime integration.
