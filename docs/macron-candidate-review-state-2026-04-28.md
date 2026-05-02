# Macron Candidate Review State — 2026-04-28

## Status

Macronized romanization handling is now a review-only candidate layer.

Current behavior:

- OCR text is preserved exactly as emitted.
- Cleanup and serialization preserve macrons when OCR emits them.
- A glossary-backed candidate generator can suggest likely macronized martial arts terms.
- A review artifact helper can scan OCR review outputs and summarize candidates.
- Every candidate requires human review.
- No automatic replacement is performed.
- No runtime, schema, OCR default, extraction, or serialization behavior has changed.

## Why This Exists

Martial arts research material often contains romanized Japanese terms with macrons, such as:

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

The OCR engine comparison showed that current OCR paths do not reliably emit these macrons. Tesseract often strips or corrupts them, EasyOCR did not recover the Latin romanized terms in the controlled test, and PaddleOCR was not evaluated due local eval-environment setup constraints.

The text pipeline itself is not the blocker: when macrons are present in text, cleanup and serialization preserve them. The appropriate bridge is therefore a review-required candidate layer, not blind normalization.

## Evidence Summary

Macron sample curation found:

- no confirmed real scanned-page samples with visible macronized romanization,
- controlled synthetic fixture images were needed,
- Tesseract `eng`, `eng+jpn`, and `jpn` did not preserve macrons reliably,
- cleanup and serialization controls preserved all macron terms when the text already contained them.

Macron OCR engine comparison found:

- no tested engine preserved macron-bearing terms well enough for direct trust,
- Tesseract produced useful Latin text but stripped or misread macrons,
- EasyOCR did not recover the target Latin romanized terms,
- PaddleOCR remained unevaluated for macron OCR output.

The candidate generator pass added a high-precision glossary-backed utility.

The review artifact pass showed that the utility can scan existing OCR review outputs and generate useful candidate summaries without mutating source text.

## Current Components

Candidate generator:

```text
utils/text/macron_candidates.py
```

Review artifact helper:

```text
experiments/review_macron_candidates.py
```

Tests:

```text
tests/test_macron_normalization_candidates.py
tests/test_macron_candidate_review_experiment.py
```

Design:

```text
docs/macron-normalization-candidate-design.md
```

Latest review note:

```text
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-artifact-pass.md
```

## Candidate Behavior

Input:

```text
OCR text
```

Output:

```text
normalization candidate suggestions
```

The source text is not changed.

Example candidate:

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

Supported match types:

```text
variant_exact
variant_hyphen_space
observed_ocr_confusion
```

Candidate generation is intentionally conservative:

- exact glossary variants are allowed,
- case-insensitive matches preserve the observed source form,
- hyphen/space tolerance applies only to listed terms,
- OCR-confusion matches apply only when explicitly listed,
- broad fuzzy matching is not enabled,
- word-boundary matching avoids substring replacement,
- canonical macronized text does not produce replacement candidates.

## Review Artifact Behavior

The review helper scans:

- existing ignored OCR review summary JSON files,
- prior synthetic macron OCR experiment summaries,
- controlled fixture strings.

Default ignored output:

```text
data/notebook_outputs/macron_candidate_review/summary.json
```

Latest run:

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

Real OCR review artifacts produced a small but useful candidate signal, including:

```text
BUDO -> budō
```

That example is useful precisely because it still requires review. The page may intentionally use ASCII publication style, so the system should not rewrite it automatically.

## Stable / Accepted

- OCR engines currently do not reliably preserve macrons.
- Cleanup and serialization preserve macrons if present.
- Candidate generation is glossary-backed and review-required.
- Candidate reports preserve spans and context.
- Source OCR text is never mutated.
- Generated review reports stay under ignored output paths.
- Runtime defaults remain unchanged.
- Canonical Japanese/romanization fields remain deferred.

## Known Limitations

- Confirmed real macron-bearing scanned-page samples are still missing.
- Current glossary seed is intentionally small.
- Candidate confidence is categorical, not statistical.
- Real OCR candidate evidence is sparse.
- The helper reports candidates but does not support reviewer decisions yet.
- No accept/reject workflow exists.
- No reviewed correction export format exists.
- Publication style may intentionally omit macrons, so candidates cannot be treated as corrections automatically.

## Do Not Change Right Now

- Do not add blind automatic normalization.
- Do not mutate `raw_text`, `readable_text`, `text.txt`, or `data.json["text"]`.
- Do not promote canonical Japanese or romanization fields from candidate output alone.
- Do not add broad fuzzy matching.
- Do not treat ASCII forms as incorrect without human review.
- Do not integrate the candidate layer into runtime defaults.

## Future Backlog

- Curate real macron-bearing source pages.
- Expand the martial arts glossary with reviewed provenance.
- Add reviewer accept/reject workflow.
- Add correction export format.
- Add context scoring for terms, titles, headings, and names.
- Add ambiguity handling UI.
- Add optional review-mode `DocumentResult` metadata only after review workflow design.
- Revisit alternate OCR engines if real macron samples become available.

## Recommended Next Branch

Design a review/export workflow for normalization candidates.

The next branch should define:

```text
readable_text
normalization_candidates
source path / page ID
candidate context
accept/reject placeholders
reviewer notes
export format
```

This should remain opt-in and review-only. It should not perform automatic text replacement or change runtime defaults.

## Relevant Files

```text
utils/text/macron_candidates.py
experiments/review_macron_candidates.py
tests/test_macron_normalization_candidates.py
tests/test_macron_candidate_review_experiment.py
docs/macron-normalization-candidate-design.md
```

## Relevant Review Notes

```text
data/evaluation/real_page_review/notes/2026-04-28-macron-ocr-sample-curation.md
data/evaluation/real_page_review/notes/2026-04-28-macron-ocr-engine-comparison.md
data/evaluation/real_page_review/notes/2026-04-28-macron-normalization-candidate-pass.md
data/evaluation/real_page_review/notes/2026-04-28-macron-candidate-review-artifact-pass.md
```
