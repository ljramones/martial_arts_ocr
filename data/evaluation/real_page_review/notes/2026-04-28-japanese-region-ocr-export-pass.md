# Japanese Region OCR Export Pass

Run date: 2026-05-01

## Purpose

Add experiment-only structured review exports for Japanese region OCR results.

This pass keeps Japanese region OCR outside runtime processing. It does not change OCR defaults, extraction behavior, serialization, canonical model fields, schema, or `WorkflowOrchestrator`.

## What Changed

Updated:

```text
experiments/review_japanese_region_ocr.py
tests/test_japanese_region_ocr_experiment.py
```

The helper still writes the existing broad summary:

```text
summary.json
```

It now also writes review-focused artifacts:

```text
region_ocr_results.json
region_ocr_results.csv
region_ocr_review.md
```

These files are generated under the selected ignored experiment output directory.

## Output Structure

`region_ocr_results.json` uses:

```json
{
  "schema_version": "japanese_region_ocr_review.v1",
  "manifest": "path/to/manifest.local.json",
  "result_count": 4,
  "results": []
}
```

Each result records the best Tesseract route for a manually selected region:

```json
{
  "sample_id": "manual_s34193423_vertical_sidebar",
  "source_image": ".../S__34193423.jpg",
  "bbox": [365, 60, 190, 245],
  "region_type": "vertical_modern_japanese",
  "route": {
    "language": "jpn_vert",
    "psm": 5,
    "preprocess_profile": "none",
    "role": "primary"
  },
  "ocr_output": "...",
  "ocr_output_preview": ["..."],
  "expected_terms": ["忍者", "伊賀", "甲賀"],
  "terms_recovered": ["忍者", "伊賀", "甲賀"],
  "terms_missing": [],
  "quality_judgment": "meaningful",
  "needs_review": false,
  "notes": "..."
}
```

The existing `summary.json` remains the place for the broader route matrix and full-page diagnostics.

## CSV Export

`region_ocr_results.csv` includes compact review columns:

```text
sample_id
source_image
bbox
region_type
language
psm
preprocess_profile
quality_judgment
expected_terms
terms_recovered
terms_missing
needs_review
```

Terms are serialized with ` | ` separators for simple spreadsheet inspection.

## Markdown Export

`region_ocr_review.md` provides:

- a summary table;
- per-sample details;
- source path and bbox;
- selected route;
- expected/recovered/missing terms;
- quality judgment;
- OCR output text.

This is intended for human review, not canonical document output.

## Expected-Term Matching

Expected-term matching is exact substring matching only:

```text
term in ocr_output
```

No fuzzy matching, Japanese normalization, macron normalization, translation, or romanization is applied.

This keeps the export auditable and avoids accidentally treating noisy OCR as a correct recovery.

## Quality Judgment

The helper preserves the existing deterministic quality labels:

```text
meaningful:
  all expected terms are recovered

partial:
  one or more expected terms are recovered, but at least one is missing

noisy:
  Japanese-like output appears, but expected terms are missing

fail:
  empty/error/non-Japanese output
```

`needs_review` is true when the manifest/profile requests review or when the selected best result is `partial`, `noisy`, or `fail`.

## Dry Run

Command:

```bash
.venv/bin/python experiments/review_japanese_region_ocr.py \
  --manifest data/corpora/modern_japanese_ocr/manifests/curated_japanese_sample_regions.local.json \
  --output-dir data/notebook_outputs/japanese_region_ocr_export_pass
```

Generated files:

```text
data/notebook_outputs/japanese_region_ocr_export_pass/summary.json
data/notebook_outputs/japanese_region_ocr_export_pass/region_ocr_results.json
data/notebook_outputs/japanese_region_ocr_export_pass/region_ocr_results.csv
data/notebook_outputs/japanese_region_ocr_export_pass/region_ocr_review.md
```

The output directory is ignored.

Dry-run result summary:

| Sample | Region Type | Route | Quality | Terms Recovered | Needs Review |
|---|---|---|---|---|---|
| `manual_japtext2_horizontal_body` | `horizontal_modern_japanese` | `jpn / PSM 6 / upscale_2x` | partial | `日本語`, `漢文`, `文字`, `表記`, `縦書き` | true |
| `corpus2_185434_term_parentheticals` | `mixed_japanese_parentheticals` | `jpn / PSM 6 / upscale_2x` | meaningful | `術`, `剣`, `刀`, `弓`, `火`, `水`, `馬` | false |
| `manual_s34193423_vertical_sidebar` | `vertical_modern_japanese` | `jpn_vert / PSM 5 / none` | meaningful | `忍者`, `伊賀`, `甲賀` | false |
| `dfd_3336_calligraphy_block` | `stylized_calligraphy` | `jpn_vert / PSM 5 / contrast_sharpen` | partial | `有`, `人` | true |

## Tests

Updated:

```text
tests/test_japanese_region_ocr_experiment.py
```

Coverage added:

- review result JSON shape;
- route metadata in review result;
- exact expected-term recovery and missing-term lists;
- CSV row shape;
- Markdown review output content;
- `_process_sample()` populates `review_result`;
- existing broad matrix mode remains available;
- no OCR engine is required for the new tests.

## What Remains Experiment-Only

- region manifests;
- crop coordinates;
- routing profiles;
- preprocessing selection;
- OCR output;
- quality judgment;
- review/export artifacts.

None of this is integrated into runtime document processing.

## Recommended Next Use

Use the export artifacts to review more manually selected Japanese regions:

1. Add caption/label crop boxes for `original_img_3330`, `original_img_3344`, and `corpus2_new_doc_2026_04_28_18_29_28`.
2. Add Japanese block/title crop boxes for `corpus2_new_doc_2026_04_28_19_40_50` and `corpus2_new_doc_2026_04_28_19_41_28`.
3. Run `experiments/review_japanese_region_ocr.py`.
4. Inspect `region_ocr_review.md` first, then use JSON/CSV for comparison or downstream review workflows.

Do not add runtime Japanese routing or canonical Japanese fields until more reviewed region evidence exists.
