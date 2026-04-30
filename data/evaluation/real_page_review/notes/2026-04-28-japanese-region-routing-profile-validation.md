# Japanese Region Routing Profile Validation

## Purpose

Validate that the experiment-only Japanese region OCR helper applies explicit routing profiles as intended and that the profile output matches observed OCR quality.

This pass does not change runtime OCR behavior, extraction behavior, canonical model fields, serialization, or schema.

## Inputs

Routing helper:

```text
experiments/review_japanese_region_ocr.py
```

Local validation manifest:

```text
data/corpora/modern_japanese_ocr/manifests/japanese_region_routing_validation.local.json
```

The manifest is ignored and contains the same four regions from the previous region-specific evaluation, with explicit `japanese_region_type` values added:

```text
manual_japtext2_horizontal_body -> horizontal_modern_japanese
dfd_3336_calligraphy_block -> stylized_calligraphy
corpus2_185434_term_parentheticals -> mixed_english_japanese
manual_s34193423_vertical_sidebar -> vertical_modern_japanese
```

Comparison baseline:

```text
data/evaluation/real_page_review/notes/2026-04-28-region-specific-japanese-ocr-eval.md
```

Generated outputs, crops, and JSON summaries were written only under ignored output directories:

```text
data/notebook_outputs/japanese_region_routing_profile_validation/
```

## Environment

- Tesseract version: 5.5.2
- Relevant installed languages: `eng`, `jpn`, `jpn_vert`
- EasyOCR tested: no
- PaddleOCR tested: no
- Runtime defaults changed: no

Command:

```bash
.venv/bin/python experiments/review_japanese_region_ocr.py \
  --manifest data/corpora/modern_japanese_ocr/manifests/japanese_region_routing_validation.local.json \
  --output-dir data/notebook_outputs/japanese_region_routing_profile_validation
```

## Summary Table

| Sample ID | Expected Profile | Selected Profile | Intended Route Check | Best OCR Route | Quality Judgment | Expected Terms Recovered | Matches Prior Broad Matrix? | Notes |
|---|---|---|---|---|---|---|---|---|
| `manual_japtext2_horizontal_body` | `horizontal_modern_japanese` | `horizontal_modern_japanese` | yes: `jpn`, PSM 6 primary | `jpn`, PSM 6, `upscale_2x` | partial | `日本語`, `漢文`, `文字`, `表記`, `縦書き` | yes | The routed helper chooses the same useful horizontal route found previously. Missing `横書き` keeps judgment partial. |
| `dfd_3336_calligraphy_block` | `stylized_calligraphy` | `stylized_calligraphy` | yes: diagnostics only, review-needed | `jpn_vert`, PSM 5, `contrast_sharpen` | partial | `有`, `人` | yes | Correctly remains diagnostic/review-only; calligraphy is not reliably solved. |
| `corpus2_185434_term_parentheticals` | `mixed_english_japanese` | `mixed_english_japanese` | yes: `eng+jpn` primary plus `jpn` diagnostic | `jpn`, PSM 6, `upscale_2x` | meaningful | `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬` | yes | Important nuance: Japanese term recovery still comes from `jpn`, not the `eng+jpn` primary route. |
| `manual_s34193423_vertical_sidebar` | `vertical_modern_japanese` | `vertical_modern_japanese` | yes: `jpn_vert`, PSM 5 primary | `jpn_vert`, PSM 5, `none` | meaningful | `忍者`, `伊賀`, `甲賀` | yes | Confirms vertical routing behavior. |

## Per-Sample Notes

### `manual_japtext2_horizontal_body`

Selected profile: `horizontal_modern_japanese`

Selected routes:

```text
jpn, PSM 6, none/upscale_2x/threshold, primary
jpn, PSM 7, none/upscale_2x, single_line_diagnostic
```

Best result:

```text
jpn, PSM 6, upscale_2x
quality_judgment: partial
expected terms recovered: 日本語, 漢文, 文字, 表記, 縦書き
```

Assessment:

The helper selected the intended horizontal route. The output matches the prior broad-matrix finding that `jpn`, PSM 6 is useful for clean horizontal Japanese. The quality judgment is technically partial because `横書き` was not recovered in the best crop result.

### `dfd_3336_calligraphy_block`

Selected profile: `stylized_calligraphy`

Selected routes:

```text
jpn_vert, PSM 5, none/contrast_sharpen, diagnostic
jpn, PSM 6, none/contrast_sharpen, diagnostic
```

Best result:

```text
jpn_vert, PSM 5, contrast_sharpen
quality_judgment: partial
expected terms recovered: 有, 人
needs_human_review: true
```

Assessment:

The helper correctly treats stylized calligraphy as diagnostic/review-needed. The output still recovers only isolated characters, consistent with the previous evaluation. This profile should not become an automatic canonical Japanese extraction path.

### `corpus2_185434_term_parentheticals`

Selected profile: `mixed_english_japanese`

Selected routes:

```text
eng+jpn, PSM 6, none/upscale_2x, primary
eng+jpn, PSM 11, none/upscale_2x, sparse_text_diagnostic
jpn, PSM 6, none/upscale_2x, japanese_term_diagnostic
```

Best result:

```text
jpn, PSM 6, upscale_2x
quality_judgment: meaningful
expected terms recovered: 術, 剣, 刀, 弓, 火, 水, 馬
```

Assessment:

The helper selected the intended mixed profile and included `eng+jpn` primary routes. However, the best result again came from the `jpn` diagnostic route, not from `eng+jpn`. This confirms the earlier observation that `eng+jpn` is not automatically better for mixed English/Japanese regions when the review goal is Japanese term recovery.

Design implication:

`mixed_english_japanese` may need to be split or clarified in a later experiment:

```text
mixed_english_japanese_page_text:
  prioritize eng+jpn for blended readability

mixed_japanese_parentheticals:
  prioritize jpn for Japanese term recovery
```

No helper behavior was changed in this validation pass.

### `manual_s34193423_vertical_sidebar`

Selected profile: `vertical_modern_japanese`

Selected routes:

```text
jpn_vert, PSM 5, none/upscale_2x/threshold, primary
jpn, PSM 6, none/upscale_2x, horizontal_comparison
```

Best result:

```text
jpn_vert, PSM 5, none
quality_judgment: meaningful
expected terms recovered: 忍者, 伊賀, 甲賀
```

Assessment:

The helper selected the intended vertical route, and the quality judgment matches visual review. The route also preserved the previous finding that `jpn_vert`, PSM 5 is decisive for this region.

## Decision Questions

### Does each region sample select the intended profile?

Yes. All four manifest labels selected the intended routing profiles.

### Does vertical Japanese route to `jpn_vert` + PSM 5?

Yes. `manual_s34193423_vertical_sidebar` selected `vertical_modern_japanese`, and its primary route was `jpn_vert`, PSM 5.

### Does horizontal Japanese route to `jpn` + PSM 6?

Yes. `manual_japtext2_horizontal_body` selected `horizontal_modern_japanese`, and its primary route was `jpn`, PSM 6.

### Does mixed English/Japanese route to `eng+jpn`?

Yes, as primary routes. But the best Japanese-term recovery still came from the `jpn` diagnostic route. This should be treated as a design signal: mixed regions may need separate routing depending on whether the goal is blended page text or Japanese term recovery.

### Does stylized calligraphy stay diagnostics/review-only?

Yes. `dfd_3336_calligraphy_block` selected `stylized_calligraphy`, used diagnostic routes, and kept `needs_human_review=true`.

### Do `quality_judgment` values match observed OCR quality?

Mostly yes:

- `meaningful` matched full expected-term recovery for the vertical sidebar and Corpus 2 parentheticals.
- `partial` matched missing expected terms on the horizontal crop and calligraphy crop.
- The labels are useful as lightweight review signals, not quantitative OCR metrics.

## Cross-Profile Findings

- Explicit routing profiles reproduce the useful routes from the previous broad matrix while running fewer OCR combinations.
- `vertical_modern_japanese` is well-supported by current evidence.
- `horizontal_modern_japanese` is reasonable for clean horizontal modern Japanese.
- `stylized_calligraphy` is correctly treated as review-only.
- `mixed_english_japanese` needs refinement: `eng+jpn` is useful to compare, but `jpn` remains the better term-recovery route on the current Corpus 2 sample.

## Recommendation

Do not integrate Japanese region OCR into runtime yet.

Next narrow pass:

```text
Refine experiment-only mixed-region routing labels.
```

Specifically, consider splitting:

```text
mixed_english_japanese
```

into:

```text
mixed_english_japanese_page_text
mixed_japanese_parentheticals
```

Then rerun the same validation manifest. This is still experiment-only and should not affect canonical fields or runtime defaults.

## Git Hygiene

Do not stage:

```text
data/corpora/modern_japanese_ocr/manifests/japanese_region_routing_validation.local.json
data/notebook_outputs/japanese_region_routing_profile_validation/
```

Only this review note should be committed from the validation pass.
