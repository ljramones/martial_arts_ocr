# Macron Normalization Candidate Pass

## Purpose

Implement a small review-only utility that proposes glossary-backed macron normalization candidates without mutating OCR text.

This pass follows:

```text
docs/macron-normalization-candidate-design.md
```

## Scope

Changed:

```text
utils/text/macron_candidates.py
tests/test_macron_normalization_candidates.py
```

Not changed:

```text
runtime OCR behavior
OCR defaults
cleanup behavior
serialization behavior
canonical Japanese/romanization fields
database schema
extraction behavior
```

## Implementation Summary

The utility exposes:

```python
find_macron_normalization_candidates(text)
```

It returns `MacronCandidate` objects with JSON-friendly `to_dict()` output.

Candidate output includes:

```text
observed
candidate
span
context
source
match_type
requires_review
confidence
term_category
ambiguous
notes
```

All candidates are review-required:

```text
requires_review: true
confidence: candidate
```

The source OCR text is never modified.

## Glossary Seed

The initial in-code reviewed seed includes:

```text
koryū
budō
Daitō-ryū
jūjutsu
dōjō
ryūha
sōke
iaidō
aikijūjutsu
```

It also intentionally treats:

```text
kenjutsu
```

as a non-macron term. The utility does not invent a macron for it.

## Matching Rules Implemented

Implemented:

```text
exact known variant match
case-insensitive match preserving observed text
hyphen/space tolerant known variants
word-boundary aware matching
known OCR-confusion variants only when listed in the glossary
ambiguity marking for multiple candidates on the same observed span
```

Not implemented:

```text
broad edit-distance fuzzy matching
automatic text replacement
runtime integration
canonical field promotion
```

## Examples

Input:

```text
The Daito-ryu tradition is discussed.
```

Candidate:

```json
{
  "observed": "Daito-ryu",
  "candidate": "Daitō-ryū",
  "span": [4, 13],
  "context": "The Daito-ryu tradition is discussed.",
  "source": "martial_arts_macron_glossary",
  "match_type": "variant_exact",
  "requires_review": true,
  "confidence": "candidate",
  "term_category": "style",
  "ambiguous": false,
  "notes": []
}
```

Input:

```text
OCR saw koryG, bud6, and Dait6-rya.
```

Candidates:

```text
koryG -> koryū
bud6 -> budō
Dait6-rya -> Daitō-ryū
```

Each uses:

```text
match_type: observed_ocr_confusion
requires_review: true
```

## Tests Added

Added:

```text
tests/test_macron_normalization_candidates.py
```

Coverage:

- exact variant creates candidate
- JSON-friendly candidate shape
- case-insensitive observed-form preservation
- canonical macron text does not create replacement candidate
- word-boundary matching avoids false positives
- hyphen/space variants are matched only when listed
- known OCR-confusion forms create review-required candidates
- ambiguous candidates are marked, not suppressed
- context and span are preserved
- source text is not mutated

Focused result:

```text
10 passed
```

## Verification

Full verification is run separately before commit.

## Recommendation

Keep this utility review-only for now.

Next useful branch:

```text
experiment/report integration
  -> feed OCR text or review summaries into macron candidate generation
  -> emit normalization_candidates[] in review artifacts
  -> do not mutate readable_text/raw_text
```

Do not add automatic normalization until reviewed accept/reject data exists.
