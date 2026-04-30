# Region-Specific Japanese OCR Evaluation

## Purpose

Determine whether Japanese OCR improves when OCR is run on selected Japanese text regions instead of full pages.

This pass does not change runtime OCR defaults, canonical model fields, extraction behavior, or serialization behavior. It uses manually selected local crop regions and ignored experiment output only.

## Environment

- Tesseract version: 5.5.2
- Installed languages: `eng`, `jpn`, `jpn_vert` available, along with many others in `/opt/homebrew/share/tessdata/`
- OCR configs tested:
  - Tesseract `jpn`, `eng+jpn`, `eng` with `--psm 6`
  - Tesseract `jpn_vert`, `jpn` with `--psm 5` and `--psm 6`
- EasyOCR tested: no
- PaddleOCR tested: no
- Preprocessing variants:
  - `original_crop`
  - `grayscale`
  - `threshold`
  - `upscale_2x`
  - `upscale_3x`
  - `contrast_or_sharpen`
- Output directory:
  - `data/notebook_outputs/japanese_region_ocr_eval/`
  - `data/notebook_outputs/japanese_region_ocr_eval_jpn_vert/`

Commands used:

```bash
.venv/bin/python experiments/review_japanese_region_ocr.py \
  --manifest data/corpora/modern_japanese_ocr/manifests/japanese_region_manifest.local.json \
  --output-dir data/notebook_outputs/japanese_region_ocr_eval \
  --language jpn --language eng+jpn --language eng --psm 6

.venv/bin/python experiments/review_japanese_region_ocr.py \
  --manifest data/corpora/modern_japanese_ocr/manifests/japanese_region_manifest.local.json \
  --output-dir data/notebook_outputs/japanese_region_ocr_eval_jpn_vert \
  --language jpn_vert --language jpn --psm 5 --psm 6
```

The region manifest is local and ignored:

```text
data/corpora/modern_japanese_ocr/manifests/japanese_region_manifest.local.json
```

## Region Samples

| Sample ID | Source Image | BBox | Visible Text | Expected Terms | Notes |
|---|---|---:|---|---|---|
| `manual_japtext2_horizontal_body` | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/JapText2.jpg` | `[0, 45, 650, 245]` | Horizontal modern Japanese body text | `日本語`, `漢文`, `文字`, `表記`, `縦書き`, `横書き` | Clean manual sample; provenance unclear but already tracked. |
| `dfd_3336_calligraphy_block` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg` | `[910, 440, 420, 390]` | Stylized Japanese calligraphy block | `有`, `無`, `構`, `殺`, `活`, `人`, `剣` | Hard project-native sample; handwritten/calligraphic style. |
| `corpus2_185434_term_parentheticals` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg` | `[60, 80, 370, 380]` | English martial arts terms with Japanese parentheticals | `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬` | Mixed English/Japanese text region. |
| `manual_s34193423_vertical_sidebar` | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/S__34193423.jpg` | `[365, 60, 190, 245]` | Vertical Japanese sidebar/header text | `忍者`, `伊賀`, `甲賀` | Tests vertical Japanese routing. |

## Summary Table

| Sample ID | OCR Config | Preprocess | Output Quality | Expected Terms Recovered | Notes |
|---|---|---|---|---|---|
| `manual_japtext2_horizontal_body` | `jpn`, PSM 6 | `upscale_2x` | meaningful | `日本語`, `漢文`, `文字`, `表記`, `縦書き` | Full-page `jpn` PSM 6 already recovered all expected terms; crop remained useful but did not improve the clean sample. |
| `dfd_3336_calligraphy_block` | `jpn_vert`, PSM 5 | `contrast_or_sharpen` | partial/noisy | `有`, `人` | Cropping did not make stylized calligraphy reliably readable. |
| `corpus2_185434_term_parentheticals` | `jpn`, PSM 6 | `upscale_2x` | meaningful but noisy | `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬` | Region crop recovered all target Japanese term characters; full page only recovered a small subset. |
| `manual_s34193423_vertical_sidebar` | `jpn_vert`, PSM 5 | `original_crop` | meaningful/partial | `忍者`, `伊賀`, `甲賀` | Vertical language model and PSM 5 were decisive. |

## Per-Region Notes

### `manual_japtext2_horizontal_body`

Source: `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/JapText2.jpg`

BBox: `[0, 45, 650, 245]`

Visible expected text: modern horizontal Japanese body text, including terms such as `日本語`, `漢文`, `文字`, `表記`, and `縦書き`.

Full-page OCR:
- `jpn` PSM 6 recovered all expected terms in this sample.
- `eng+jpn` recovered only some Japanese terms.
- `eng` produced Latin OCR noise.

Region OCR:
- `jpn` PSM 6 with `upscale_2x` recovered `日本語`, `漢文`, `文字`, `表記`, and `縦書き`.
- The crop result included some bottom-line noise from the selected crop.

Best result:
- config: `jpn`, PSM 6
- preprocess: `upscale_2x`
- recovered terms: `日本語`, `漢文`, `文字`, `表記`, `縦書き`
- quality: meaningful
- notes: This clean horizontal sample shows that Tesseract Japanese can work when the page is simple and the correct language is used. It is not evidence that full-page mixed-layout OCR is solved.

### `dfd_3336_calligraphy_block`

Source: `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg`

BBox: `[910, 440, 420, 390]`

Visible expected text: stylized calligraphy block associated with `katsujin no ken` / `satsujin no to`.

Full-page OCR:
- `jpn`, `jpn_vert`, and `eng+jpn` produced mostly noisy output.
- Some runs recovered isolated expected characters such as `人` and `有`.

Region OCR:
- `jpn_vert` PSM 5 with `contrast_or_sharpen` recovered `有` and `人`.
- Other variants were noisy and did not recover enough expected characters to be useful.

Best result:
- config: `jpn_vert`, PSM 5
- preprocess: `contrast_or_sharpen`
- recovered terms: `有`, `人`
- quality: partial/noisy
- notes: Region cropping is not enough for stylized calligraphy. This likely needs separate handling, better ground truth, or a different OCR strategy.

### `corpus2_185434_term_parentheticals`

Source: `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg`

BBox: `[60, 80, 370, 380]`

Visible expected text: mixed English romanized martial arts terms with Japanese parentheticals.

Full-page OCR:
- `jpn` PSM 6 recovered only a small subset of expected terms, including `刀` and `火`.
- `eng+jpn` and `eng` did not recover useful Japanese terms.

Region OCR:
- `jpn` PSM 6 with `upscale_2x` recovered `術`, `剣`, `刀`, `弓`, `火`, `水`, and `馬`.
- Output was still noisy and mixed with English OCR errors, but the Japanese term signal became useful.

Best result:
- config: `jpn`, PSM 6
- preprocess: `upscale_2x`
- recovered terms: `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬`
- quality: meaningful but noisy
- notes: This is the strongest evidence that full-page layout noise is a blocker and region-specific OCR can recover useful Japanese terms.

### `manual_s34193423_vertical_sidebar`

Source: `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/S__34193423.jpg`

BBox: `[365, 60, 190, 245]`

Visible expected text: vertical Japanese sidebar/header text related to ninja origins.

Full-page OCR:
- `jpn_vert` PSM 5 recovered `忍者` but was very noisy.
- `jpn`, `jpn_vert` PSM 6, `eng+jpn`, and `eng` were not reliable for the target sidebar.

Region OCR:
- `jpn_vert` PSM 5 on `original_crop` recovered `忍者`, `伊賀`, and `甲賀`.
- Horizontal `jpn` PSM 6 did not recover the expected terms from the crop.

Best result:
- config: `jpn_vert`, PSM 5
- preprocess: `original_crop`
- recovered terms: `忍者`, `伊賀`, `甲賀`
- quality: meaningful/partial
- notes: Vertical Japanese needs explicit vertical-language routing. Full-page OCR may contain the signal, but the cropped vertical region makes it much easier to isolate.

## Cross-Region Findings

- Region-specific OCR materially improved 2 of 4 samples:
  - Corpus 2 mixed English/Japanese parentheticals.
  - Manual vertical Japanese sidebar.
- The clean horizontal Japanese sample already worked with full-page `jpn` PSM 6, so cropping was not necessary there.
- The DFD calligraphy sample remained weak after cropping and preprocessing.
- `jpn` is the strongest language for horizontal modern Japanese.
- `jpn_vert` with PSM 5 is important for vertical Japanese.
- `eng+jpn` was not the best config in any tested region. It can recover some Japanese glyphs, but the mixed language model did not outperform a targeted Japanese language choice.
- Upscaling helped on horizontal region crops. It was not universally useful, and the vertical sidebar worked best without preprocessing.

## Failure Patterns

- Full-page OCR loses Japanese signal when Japanese appears inside mixed English/Japanese or visual-heavy page regions.
- Wrong orientation/language selection causes severe noise, especially for vertical Japanese.
- Stylized/calligraphic Japanese is not solved by simple crop preprocessing.
- `eng+jpn` is not a reliable replacement for choosing `jpn` or `jpn_vert` based on region orientation.
- Crop quality matters. Overly broad crops can introduce bottom-line or neighboring text noise.

## Decision

- [x] Region-specific preprocessing makes Tesseract Japanese usable
- [ ] Tesseract remains insufficient even on clean crops
- [ ] EasyOCR should be evaluated more seriously
- [ ] PaddleOCR Japanese OCR should be evaluated in `.venv-eval`
- [x] Better sample/ground truth curation is needed
- [x] Do not promote Japanese fields yet

## Recommendation

Do not promote Japanese analysis into canonical model fields yet.

The next useful branch is a region-routing and preprocessing design pass:

```text
Japanese-bearing region selection
  -> orientation classification: horizontal vs vertical
  -> language routing: jpn vs jpn_vert
  -> small preprocessing set: original / upscale_2x / contrast
  -> compact diagnostics
```

This should remain review/evaluation mode until a larger curated region set with expected terms or ground truth exists. The current evidence says Tesseract is usable for some modern Japanese regions, but not with blind full-page OCR and not for stylized calligraphy.
