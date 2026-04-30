# Japanese OCR Experimental State — 2026-04-28

## Status

Modern Japanese OCR work is currently evidence-driven and experiment-only.

Current accepted state:

- Modern Japanese OCR resources have a dedicated corpus area.
- Real page sampling confirmed visible Japanese in selected project-native and manual samples.
- Full-page Japanese OCR is noisy on mixed layouts.
- Region-specific OCR can recover useful modern Japanese terms.
- Japanese OCR routing profiles are explicit and testable in `experiments/review_japanese_region_ocr.py`.
- Runtime OCR behavior is unchanged.
- Canonical Japanese fields are not promoted.
- Database schema and document serialization are unchanged.

## Current Experiment Stack

```text
modern_japanese_ocr corpus area
  -> local/manual region manifests
  -> experiments/review_japanese_region_ocr.py
  -> explicit region profiles
  -> Tesseract language / PSM / preprocessing route
  -> ignored OCR outputs
  -> review notes
```

This stack is for evaluation only. It is not part of normal document processing.

## Routing Profiles

### `horizontal_modern_japanese`

Optimizes for:

```text
clean horizontal kana/kanji body or line text
```

Primary route:

```text
jpn, PSM 6, none/upscale_2x/threshold
```

Evidence:

- The clean horizontal `JapText2` sample recovered useful terms with `jpn`, PSM 6.
- Upscaling was useful but not always materially better than the original crop.

### `vertical_modern_japanese`

Optimizes for:

```text
vertical Japanese columns, sidebars, headers, and panels
```

Primary route:

```text
jpn_vert, PSM 5, none/upscale_2x/threshold
```

Evidence:

- The vertical sidebar sample recovered `忍者`, `伊賀`, and `甲賀` with `jpn_vert`, PSM 5.
- Horizontal `jpn` was not enough for the same region.

### `mixed_english_japanese_page_text`

Optimizes for:

```text
blended readability across English and Japanese in the same crop
```

Primary route:

```text
eng+jpn, PSM 6
```

Diagnostics:

```text
eng+jpn, PSM 11
jpn, PSM 6
```

Evidence:

- The profile split exists because mixed regions have different goals.
- This profile is for readable mixed text, not necessarily Japanese term recall.

### `mixed_japanese_parentheticals`

Optimizes for:

```text
recovering Japanese terms embedded inside mostly English context
```

Primary route:

```text
jpn, PSM 6, none/upscale_2x
```

Diagnostics:

```text
eng+jpn, PSM 6/11
```

Evidence:

- The Corpus 2 parenthetical crop recovered `術`, `剣`, `刀`, `弓`, `火`, `水`, and `馬` with `jpn`, PSM 6, `upscale_2x`.
- `eng+jpn` did not outperform targeted `jpn` for term recovery on that sample.

### `mixed_english_japanese`

Compatibility alias for:

```text
mixed_english_japanese_page_text
```

This keeps older local manifests working while newer manifests can use goal-specific profiles.

### `romanized_japanese_macrons`

Optimizes for:

```text
Latin romanized Japanese with macrons such as koryū, budō, Daitō-ryū, jūjutsu
```

Current evidence:

- Synthetic cleanup tests preserve macrons.
- Real OCR sampling has not yet proven macronized romanization recovery.

Primary route remains experimental:

```text
eng, PSM 6
eng+jpn, PSM 6 as comparison
```

Do not design canonical Japanese fields around macron recovery until targeted real samples exist.

### `stylized_calligraphy`

Optimizes for:

```text
diagnostics only / review-needed handling
```

Routes:

```text
jpn_vert, PSM 5, diagnostic
jpn, PSM 6, diagnostic
```

Evidence:

- The DFD calligraphy region recovered only isolated characters such as `有` and `人`.
- Current Tesseract crop/preprocessing is not reliable for stylized calligraphy.

### `unknown_japanese_like`

Fallback exploratory profile.

Use when a region might contain Japanese but orientation, source quality, or content type is unclear.

## Evidence Trail

Focused full-page evaluation:

- Visible Japanese was confirmed in selected samples.
- Full-page `eng+jpn` and `jpn` produced noisy Japanese-like output on mixed layouts.
- Reliable Japanese terms were not recovered consistently.
- Macrons were not present in that focused sample set.

Region-specific evaluation:

- Cropping materially improved the Corpus 2 parenthetical term region.
- Vertical Japanese needed `jpn_vert`, PSM 5.
- Clean horizontal modern Japanese can work with `jpn`, PSM 6.
- Stylized calligraphy remained weak.

Routing-profile validation:

- All validation samples selected the intended profiles.
- `vertical_modern_japanese` and `horizontal_modern_japanese` behaved as expected.
- `stylized_calligraphy` stayed diagnostic/review-only.
- The original mixed profile was too broad.

Mixed-profile split:

- `mixed_english_japanese_page_text` now represents blended readability.
- `mixed_japanese_parentheticals` now represents Japanese term recovery.
- The old `mixed_english_japanese` name remains a compatibility alias.

## What Remains Experiment-Only

- Japanese region manifests.
- Crop coordinates.
- Routing profiles.
- Preprocessing profile selection.
- Region OCR output.
- Quality judgments.
- `needs_human_review` decisions for Japanese OCR regions.

None of this is currently part of normal runtime OCR processing.

## Why Runtime Integration Is Deferred

Runtime integration is premature because:

- Region selection is still manual.
- Orientation classification is not automated.
- The routing profiles are validated on a very small set.
- Mixed-region goals differ between readable page text and term recovery.
- Stylized calligraphy is not solved.
- Alternate engines have not been compared on the same regions.
- Macrons have not been proven in real OCR output.

## Why Canonical Japanese Fields Are Deferred

Canonical Japanese fields should wait until the OCR stream is reliable enough to populate them.

Current blockers:

- Full-page OCR does not reliably recover Japanese text.
- Region OCR works only when the region is manually selected and routed.
- Real macronized romanization has not been observed in sampled OCR output.
- Japanese analysis would currently amplify OCR noise if promoted globally.

The current correct place for Japanese OCR output is experiment/review diagnostics.

## Known Gaps

- Broader modern Japanese region sample set is needed.
- Macron-bearing real samples are still missing.
- Vertical routing has one strong sample, not a corpus-level validation.
- Mixed parenthetical routing needs more samples.
- Stylized calligraphy likely needs manual transcription, a different engine, or separate treatment.
- EasyOCR has not been seriously evaluated for these regions.
- PaddleOCR/other modern OCR engines have not been compared on the same region manifest.
- No review/export workflow exists for correcting Japanese OCR regions.

## Do Not Change Right Now

- Do not promote Japanese analysis into canonical model fields.
- Do not change runtime OCR defaults.
- Do not route all pages through `eng+jpn`.
- Do not assume full-page Japanese OCR is sufficient.
- Do not add automatic Japanese region detection yet.
- Do not treat stylized calligraphy OCR as reliable.
- Do not use unknown-provenance datasets for benchmark claims.

## Future Backlog

- Curate macron-bearing real samples.
- Add more manually labeled Japanese region samples.
- Validate `mixed_japanese_parentheticals` on more Corpus 2 pages.
- Compare EasyOCR on the same region manifest without making it a runtime dependency.
- Compare PaddleOCR or other OCR engines in an isolated eval environment.
- Add orientation metadata to local region manifests.
- Design a review/export workflow for Japanese region crops and OCR results.
- Revisit canonical Japanese fields only after real OCR evidence supports them.

## Recommended Next Branch

Macron-focused sample curation.

Reason:

```text
Japanese region routing is now testable,
but real OCR has still not proven macronized romanization recovery.
```

The next evidence pass should find or create a small, trustworthy sample set with visible macronized romanized Japanese:

```text
koryū
budō
Daitō-ryū
jūjutsu
ō
ū
```

Then compare OCR configs and preprocessing on those crops before designing any canonical Japanese/romanization fields.

## Relevant Files

- `docs/modern-japanese-ocr-corpus-plan.md`
- `docs/japanese-region-ocr-routing-design.md`
- `experiments/review_japanese_region_ocr.py`
- `tests/test_japanese_region_ocr_experiment.py`
- `data/corpora/modern_japanese_ocr/README.md`

## Relevant Review Notes

- `data/evaluation/real_page_review/notes/2026-04-28-modern-japanese-ocr-focused-eval.md`
- `data/evaluation/real_page_review/notes/2026-04-28-region-specific-japanese-ocr-eval.md`
- `data/evaluation/real_page_review/notes/2026-04-28-japanese-region-routing-helper-pass.md`
- `data/evaluation/real_page_review/notes/2026-04-28-japanese-region-routing-profile-validation.md`
- `data/evaluation/real_page_review/notes/2026-04-28-japanese-mixed-profile-split.md`
