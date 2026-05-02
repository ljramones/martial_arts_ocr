# Curated Japanese Sample OCR Review

Run date: 2026-05-01

## Purpose

Run the curated real Japanese sample set through existing OCR review tools to validate whether the current experiment-only routing profiles remain useful on better-selected samples.

This pass does not change runtime OCR defaults, extraction behavior, serialization, canonical model fields, schema, or corpus data.

## Scope

Inputs came from:

- `data/evaluation/real_page_review/notes/2026-04-28-real-japanese-sample-curation-plan.md`
- existing project-native DFD and Corpus 2 manifests;
- tracked manual modern Japanese samples;
- an ignored local region manifest created for this review:

```text
data/corpora/modern_japanese_ocr/manifests/curated_japanese_sample_regions.local.json
```

Generated outputs were written under ignored paths:

```text
data/notebook_outputs/curated_japanese_sample_ocr_review/
```

## Commands

Region-specific routing review:

```bash
.venv/bin/python experiments/review_japanese_region_ocr.py \
  --manifest data/corpora/modern_japanese_ocr/manifests/curated_japanese_sample_regions.local.json \
  --output-dir data/notebook_outputs/curated_japanese_sample_ocr_review/region_routes
```

Full-page contrast run:

```bash
.venv/bin/python experiments/review_real_ocr_text_quality.py \
  --output-dir data/notebook_outputs/curated_japanese_sample_ocr_review/page_jpn_psm6 \
  --manifest data/corpora/modern_japanese_ocr/project_native/manifests/manifest.local.json \
  --manifest data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/manifests/manifest.local.json \
  --language jpn \
  --psm 6 \
  --sample-id original_img_3288 \
  --sample-id original_img_3330 \
  --sample-id original_img_3336 \
  --sample-id original_img_3344 \
  --sample-id corpus2_new_doc_2026_04_28_16_56_38 \
  --sample-id corpus2_new_doc_2026_04_28_18_29_28 \
  --sample-id corpus2_new_doc_2026_04_28_18_54_34 \
  --sample-id corpus2_new_doc_2026_04_28_19_40_50 \
  --sample-id corpus2_new_doc_2026_04_28_19_41_28 \
  --sample-id manual_modern_japanese_japtext2 \
  --sample-id manual_modern_japanese_s_34193423 \
  --sample-id manual_modern_japanese_istock_calligraphy
```

Macron candidate scan over the new full-page summary:

```bash
.venv/bin/python experiments/review_macron_candidates.py \
  --summary-json data/notebook_outputs/curated_japanese_sample_ocr_review/page_jpn_psm6/summary.json \
  --output-dir data/notebook_outputs/curated_japanese_sample_ocr_review/macron_candidates_page_jpn_psm6 \
  --no-fixtures \
  --filter all \
  --source-filter all \
  --sort source
```

## Region Routing Results

| Sample ID | Region Type | Best Route | Expected Terms Recovered | Quality | Decision |
|---|---|---|---|---|---|
| `manual_japtext2_horizontal_body` | `horizontal_modern_japanese` | `jpn`, PSM 6, `upscale_2x` | `日本語`, `漢文`, `文字`, `表記`, `縦書き` | partial | Profile remains useful; clean horizontal Japanese works, but not all expected terms were recovered. |
| `corpus2_185434_term_parentheticals` | `mixed_japanese_parentheticals` | `jpn`, PSM 6, `upscale_2x` | `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬` | meaningful | Profile is validated; Japanese term recovery is better served by targeted `jpn` than blended `eng+jpn`. |
| `manual_s34193423_vertical_sidebar` | `vertical_modern_japanese` | `jpn_vert`, PSM 5, `none` | `忍者`, `伊賀`, `甲賀` | meaningful | Profile remains strongly supported; vertical Japanese should route to `jpn_vert` + PSM 5. |
| `dfd_3336_calligraphy_block` | `stylized_calligraphy` | `jpn_vert`, PSM 5, `contrast_sharpen` | `有`, `人` | partial/noisy | Correctly stays diagnostic/review-only; current Tesseract route is not reliable for calligraphy. |

## Full-Page `jpn` PSM 6 Contrast

This run intentionally used one Japanese full-page config as a contrast, not as a full OCR matrix. It confirms that full-page Japanese OCR remains too noisy for mixed layouts and visual pages.

| Sample ID | Words | Lines | Readable Text Preview | Assessment |
|---|---:|---:|---|---|
| `original_img_3288` | 63 | 14 | `THE DRAEGER LECTURES`; `1. BUJUTSU AND BUD0...` | English title/list remains readable; Japanese config adds small artifacts and misreads `BUDO` as `BUD0`/`BUDo`. |
| `original_img_3330` | 310 | 59 | `ト 。`; `内 内 e w い` | Mostly noisy; figure/caption page is not useful under full-page `jpn`. |
| `original_img_3336` | 528 | 57 | English prose with Japanese-like noise | Full page remains mixed/noisy; calligraphy still needs region diagnostics or manual review. |
| `original_img_3344` | 584 | 57 | Japanese-like fragments mixed with English OCR errors | Diagram/label case remains noisy and not review-ready as full-page Japanese OCR. |
| `corpus2_new_doc_2026_04_28_16_56_38` | 113 | 15 | `SI 国 際`; noisy English/Japanese mix | Useful as a noisy false-positive/control page, not a reliable Japanese sample. |
| `corpus2_new_doc_2026_04_28_18_29_28` | 438 | 52 | `SATO KINBEI SENSEI`; Japanese-like artifacts | Caption/article page is partially readable for English names but noisy for Japanese. |
| `corpus2_new_doc_2026_04_28_18_54_34` | 273 | 30 | Parenthetical line with isolated `刀`, `火`-like signal | Full page recovers fragments; cropped parenthetical route recovers far more expected terms. |
| `corpus2_new_doc_2026_04_28_19_40_50` | 9 | 4 | `上 日 < お の` | Large Japanese cover/title page still fails as full-page OCR. |
| `corpus2_new_doc_2026_04_28_19_41_28` | 232 | 30 | `BANSENSHUKALvolume`; noisy mixed output | Full page does not recover Japanese blocks reliably. |
| `manual_modern_japanese_japtext2` | 28 | 5 | Mostly incorrect Japanese fragments | Full-page `jpn` PSM 6 failed on this image, while the selected crop worked better. |
| `manual_modern_japanese_s_34193423` | 552 | 53 | Noisy horizontal treatment of vertical text | Full-page horizontal `jpn` is inappropriate; `jpn_vert` cropped route is needed. |
| `manual_modern_japanese_istock_calligraphy` | 40 | 9 | `人生` plus noisy fragments | Stylized calligraphy remains unsuitable for current Tesseract route. |

## Macron Candidate Result

Scanning the new full-page `jpn` PSM 6 summary produced:

```text
sources scanned: 256
sources with candidates: 1
candidate count: 1
```

Candidate:

```json
{
  "observed": "BUDo",
  "candidate": "budō",
  "context": "1. BUJUTSU AND BUDo. し 主 ae",
  "match_type": "variant_exact",
  "requires_review": true,
  "case_pattern": "mixed",
  "reviewed_value_suggestion": "budō"
}
```

This is a useful review target, but it does not prove OCR macron recognition. It reinforces the existing macron workflow finding:

```text
OCR emits ASCII or corrupted romanization
  -> candidate layer proposes review-required macron value
  -> source text remains unchanged
```

The previous source-image review for this page found the better reviewed export value was `BUDŌ`, not lowercase `budō`.

## Decision Questions

### Do the existing routing profiles still work on the curated set?

Yes, for the four region samples with explicit crop boxes:

- `horizontal_modern_japanese` remains useful.
- `mixed_japanese_parentheticals` is the right goal-specific profile for the Corpus 2 term-list crop.
- `vertical_modern_japanese` remains strongly supported.
- `stylized_calligraphy` correctly stays diagnostic/review-only.

### Does horizontal Japanese route well to `jpn` / PSM 6?

Yes, when the region is clean and cropped. The clean horizontal crop recovered five expected terms with `jpn`, PSM 6, `upscale_2x`.

Full-page `jpn` PSM 6 on the same manual image was not reliable, which suggests crop selection is important even for apparently clean samples.

### Does vertical Japanese route well to `jpn_vert` / PSM 5?

Yes. The vertical sidebar route recovered all expected terms:

```text
忍者
伊賀
甲賀
```

Full-page horizontal `jpn` OCR on the same source was noisy and not suitable.

### Are parenthetical Japanese terms better served by `jpn` than `eng+jpn`?

Yes for the current evidence. The `mixed_japanese_parentheticals` profile recovered all expected target characters with:

```text
jpn, PSM 6, upscale_2x
```

This supports the profile split: blended page readability and Japanese term recovery should remain separate review goals.

### Are caption/label cases readable enough for review?

Partially, but not for Japanese extraction yet.

`original_img_3330`, `original_img_3344`, and `corpus2_new_doc_2026_04_28_18_29_28` remain noisy under full-page `jpn` PSM 6. These pages are useful as review stress cases, but they need either:

- manually selected caption/label regions,
- better OCR config comparison,
- or human review context.

### Which sample types remain unsuitable for Tesseract?

Currently unsuitable as reliable OCR sources:

- stylized calligraphy;
- large cover/title Japanese without a good crop;
- mixed diagram/label pages as full-page OCR;
- vertical pages without `jpn_vert` region routing;
- noisy scan/photo layouts where Japanese-like glyphs appear as false positives.

## Findings

- Better sample curation confirms the existing region profile design.
- Crops remain the main difference between useful Japanese OCR and noisy full-page output.
- Full-page `jpn` PSM 6 is not a general solution.
- `jpn_vert` + PSM 5 should remain the preferred route for vertical Japanese regions.
- `mixed_japanese_parentheticals` should remain distinct from blended mixed page text.
- Macron work remains review-candidate based; visible source macrons are still not confirmed.

## Recommendation

Do not promote Japanese canonical fields yet.

Recommended next branch:

```text
Create more curated region boxes for the current 12-sample set.
```

Priority:

1. Add caption/label crop boxes for `original_img_3330`, `original_img_3344`, and `corpus2_new_doc_2026_04_28_18_29_28`.
2. Add Japanese block/title crop boxes for `corpus2_new_doc_2026_04_28_19_40_50` and `corpus2_new_doc_2026_04_28_19_41_28`.
3. Re-run `review_japanese_region_ocr.py` with explicit region profiles.
4. Keep macron review candidate workflow unchanged until real visible macron source images are found.

Do not add runtime routing or schema fields until the region-box evidence set is larger.
