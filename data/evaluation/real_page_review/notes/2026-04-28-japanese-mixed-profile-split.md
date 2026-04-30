# Japanese Mixed Profile Split

## Purpose

Split the experiment-only `mixed_english_japanese` routing idea into two goal-specific profiles.

The routing-profile validation showed that a single mixed profile conflated two different review goals:

```text
readable blended English/Japanese text
Japanese term recovery inside English context
```

This pass keeps the change experiment-only. It does not alter runtime OCR defaults, canonical model fields, extraction behavior, serialization, or schema.

## Profiles Added

### `mixed_english_japanese_page_text`

Goal:

```text
Optimize for blended readable page text across English and Japanese.
```

Routes:

```text
eng+jpn, PSM 6, none/upscale_2x, primary
eng+jpn, PSM 11, none/upscale_2x, sparse_text_diagnostic
jpn, PSM 6, none/upscale_2x, japanese_term_diagnostic
```

Use when the question is whether a mixed-language crop can become readable OCR text.

### `mixed_japanese_parentheticals`

Goal:

```text
Optimize for recovering Japanese terms embedded inside mostly English context.
```

Routes:

```text
jpn, PSM 6, none/upscale_2x, primary_term_recovery
eng+jpn, PSM 6, none/upscale_2x, blended_readability_comparison
eng+jpn, PSM 11, none/upscale_2x, sparse_text_diagnostic
```

Use when the question is whether Japanese parentheticals, labels, or term-list characters can be recovered.

## Compatibility

The previous profile name remains available:

```text
mixed_english_japanese
```

It aliases to:

```text
mixed_english_japanese_page_text
```

This keeps existing local manifests and commands working while allowing new manifests to be more explicit.

## Tests

Updated:

```text
tests/test_japanese_region_ocr_experiment.py
```

Coverage:

- `mixed_english_japanese_page_text` prioritizes `eng+jpn`.
- `mixed_japanese_parentheticals` prioritizes `jpn`.
- Legacy `mixed_english_japanese` remains an alias for the page-text profile.

## Recommendation

Use `mixed_japanese_parentheticals` for the Corpus 2 parenthetical term crop in the next local validation manifest.

Do not promote either profile into runtime yet. The next evidence step should compare both mixed profiles on a larger set of manually labeled mixed regions.
