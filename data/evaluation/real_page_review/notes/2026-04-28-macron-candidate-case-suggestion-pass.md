# Macron Candidate Case Suggestion Pass

## Purpose

Improve candidate suggestion quality after the deferred real OCR review showed:

```text
BUDO -> budō
```

was semantically right, but the reviewed export value needed to preserve title capitalization:

```text
BUDŌ
```

This pass adds advisory casing metadata only. It does not apply corrections or mutate source text.

## Implementation

Updated:

```text
utils/text/macron_candidates.py
tests/test_macron_normalization_candidates.py
```

Added candidate metadata:

```text
case_pattern
reviewed_value_suggestion
```

Examples:

| Observed | Candidate | case_pattern | reviewed_value_suggestion |
|---|---|---|---|
| `BUDO` | `budō` | `uppercase` | `BUDŌ` |
| `budo` | `budō` | `lowercase` | `budō` |
| `Daito-ryu` | `Daitō-ryū` | `titlecase` | `Daitō-ryū` |
| `KoRyU` | `koryū` | `mixed` | `koryū` |

The canonical `candidate` remains unchanged. The suggestion is a reviewer aid, not an automatic replacement.

## Review Artifact Check

After rerunning:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --filter edited \
  --source-filter real_ocr \
  --sort source
```

The real OCR `BUDO` candidates now include:

```json
{
  "observed": "BUDO",
  "candidate": "budō",
  "case_pattern": "uppercase",
  "reviewed_value_suggestion": "BUDŌ"
}
```

This matches the manually reviewed value from the deferred real OCR review:

```text
reviewed_value: BUDŌ
```

## Decision

- [x] Case-aware suggestion improves reviewer ergonomics.
- [x] `candidate` remains canonical glossary form.
- [x] `reviewed_value_suggestion` is advisory only.
- [x] All candidates still require review.
- [x] Source text remains unmodified.
- [x] No runtime/OCR/extraction/serialization behavior changed.

## Limits

This does not infer publication policy.

For example:

```text
BUDO
```

may still be intentionally ASCII in the source. The system can suggest `BUDŌ`, but a reviewer must decide whether to use it in an export.

The suggestion also does not solve title casing for multi-word phrases beyond preserving simple observed case patterns.

## Recommended Next Pass

Improve duplicate grouping and source labels in review queues:

```text
- group repeated candidate rows across related summaries
- include sample_id and source image path when available
- show occurrence count
- preserve all candidate IDs for decision export
```

Do not add automatic normalization.
