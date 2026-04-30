# Japanese Region OCR Routing Design

## Status

Japanese OCR should not be promoted into canonical model fields yet.

The current evidence supports a narrower conclusion:

- Full-page Japanese OCR is weak on mixed pages.
- Region-specific OCR can recover useful modern Japanese terms.
- Vertical Japanese requires explicit vertical routing.
- Stylized/calligraphic Japanese remains unreliable with the current Tesseract path.

This design is for review/evaluation mode only. It does not change runtime OCR defaults, extraction behavior, serialization behavior, or canonical model fields.

## Evidence Summary

The focused full-page review showed that visible Japanese exists in the selected samples, but blind full-page `eng`, `eng+jpn`, and `jpn` runs did not reliably recover meaningful Japanese text.

The region-specific review changed the picture:

- Clean horizontal modern Japanese worked with `jpn` and PSM 6.
- Corpus 2 mixed English/Japanese parentheticals improved materially when cropped and run with `jpn`, PSM 6, and `upscale_2x`.
- A vertical Japanese sidebar recovered `忍者`, `伊賀`, and `甲賀` only when routed to `jpn_vert`, PSM 5.
- A DFD calligraphy block remained poor even after cropping, orientation changes, and contrast/sharpen preprocessing.

The strongest operational result is that Japanese OCR quality depends on region selection, orientation, and language/PSM routing. `eng+jpn` is not a reliable substitute for choosing `jpn` or `jpn_vert` on a region.

## Problem Statement

The current OCR pipeline is optimized around a selected/best full-page OCR result. That is appropriate for English-heavy document output, but it is not enough for Japanese-bearing regions embedded inside:

- English prose.
- martial arts term lists.
- figure captions.
- diagram labels.
- vertical sidebars.
- stylized calligraphy.

The next problem is not schema design. It is determining whether Japanese-bearing regions can be identified, cropped, oriented, preprocessed, and routed to the right OCR configuration consistently enough to produce reliable text.

## Region Types

### `horizontal_modern_japanese`

Visual characteristics:

- Clean horizontal kana/kanji body text.
- Usually a rectangular text block.
- Low figure or caption interference.

Likely OCR config:

- Tesseract `jpn`
- PSM 6 for block text
- PSM 7 for single-line regions, to be evaluated

Preprocessing profile:

- `none` for clean images
- `upscale_2x` when text is small
- `threshold` only if contrast is weak and noise is controlled

Expected quality:

- Medium on clean samples.
- The manual `JapText2` sample produced meaningful output with `jpn`, PSM 6.

Fallback behavior:

- Keep best candidate as diagnostic.
- Mark uncertain if expected terms are missing or output contains high noise.

### `vertical_modern_japanese`

Visual characteristics:

- Tall/narrow text regions.
- Columns of Japanese text.
- Often sidebars, panels, or covers.

Likely OCR config:

- Tesseract `jpn_vert`
- PSM 5

Preprocessing profile:

- `none` first.
- `upscale_2x` only if text is small.
- Avoid threshold unless source contrast is weak.

Expected quality:

- Medium/high on the tested vertical sidebar.
- `jpn_vert`, PSM 5 recovered `忍者`, `伊賀`, and `甲賀` from the cropped manual sample.

Fallback behavior:

- If `jpn_vert` fails, try `jpn` PSM 6 only as a diagnostic comparison.
- Do not treat full-page `jpn_vert` noise as reliable text.

### `mixed_english_japanese`

Visual characteristics:

- English body text or term lists with Japanese parentheticals.
- Small Japanese spans inside a primarily English page.
- OCR can be harmed by surrounding English and line layout.

Likely OCR config:

- Tesseract `jpn` for the Japanese crop.
- `eng+jpn` as a diagnostic comparison, not the default winner.
- Keep English full-page OCR separate.

Preprocessing profile:

- `upscale_2x` for small parentheticals.
- `none` and `grayscale` as controls.
- Avoid broad crops that include too much English paragraph text.

Expected quality:

- Medium for term recovery, not necessarily clean full sentence transcription.
- The Corpus 2 parenthetical crop recovered `術`, `剣`, `刀`, `弓`, `火`, `水`, and `馬` with `jpn`, PSM 6, `upscale_2x`.

Fallback behavior:

- Preserve English canonical OCR from the normal best-result path.
- Store Japanese region OCR as review diagnostics until a larger region set validates it.

### `romanized_japanese_macrons`

Visual characteristics:

- Latin text with macrons such as `ō` and `ū`.
- Terms such as `koryū`, `budō`, `Daitō-ryū`, and `jūjutsu`.

Likely OCR config:

- Tesseract `eng` or `eng+jpn`, to be evaluated.
- Japanese language packs are not necessarily helpful for Latin macron text.

Preprocessing profile:

- `none`
- `upscale_2x` for small print
- avoid destructive thresholding that can damage diacritics

Expected quality:

- Unknown. Synthetic cleanup tests preserve macrons, but real OCR sampling has not yet produced reliable macron-bearing output.

Fallback behavior:

- Treat real macron recovery as unproven.
- Do not design canonical Japanese fields around macron detection until targeted samples exist.

### `stylized_calligraphy`

Visual characteristics:

- Brush-like or calligraphic Japanese.
- Figure-like decorative text.
- Often vertical or semi-vertical.

Likely OCR config:

- None reliable yet.

Preprocessing profile:

- `contrast_sharpen` can be tried as diagnostics.
- Tesseract results should be treated as low-confidence.

Expected quality:

- Low. The DFD calligraphy crop recovered only isolated characters such as `有` and `人`.

Fallback behavior:

- Mark as difficult / needs review.
- Preserve visual crop rather than pretending OCR output is trustworthy.
- Consider future engine comparison or manual transcription workflow.

### `unknown_japanese_like`

Visual characteristics:

- OCR or visual detector suspects Japanese, but orientation/content is unclear.
- Could be noisy English, diagram marks, seals, calligraphy, or actual Japanese.

Likely OCR config:

- Diagnostic matrix only:
  - `jpn`, PSM 6
  - `jpn_vert`, PSM 5
  - `eng+jpn`, PSM 6

Preprocessing profile:

- `none`
- `upscale_2x`

Expected quality:

- Unknown.

Fallback behavior:

- Keep diagnostics compact.
- Mark `needs_human_review`.

## Proposed Routing Matrix

| Region Type | OCR Language | PSM | Preprocessing | Confidence | Notes |
|---|---|---:|---|---|---|
| `horizontal_modern_japanese` | `jpn` | 6 / 7 | crop + `none` or `upscale_2x` | medium | Clean horizontal sample worked with `jpn`, PSM 6. PSM 7 needs line-region testing. |
| `vertical_modern_japanese` | `jpn_vert` | 5 | crop + `none`, optional `upscale_2x` | medium/high on tested sample | Decisive for the vertical sidebar. |
| `mixed_english_japanese` | `jpn`; compare `eng+jpn` | 6 / 11 | crop + `upscale_2x`; avoid broad English context | low/medium | Region crop recovered Japanese parenthetical terms where full-page OCR did not. |
| `romanized_japanese_macrons` | `eng` or `eng+jpn` | 6 / 7 | crop + `none` or `upscale_2x` | unknown | Real macron output has not been observed yet. |
| `stylized_calligraphy` | none reliable yet | n/a | review/manual/future engine | low | Tesseract recovered only isolated characters after crop/preprocessing. |
| `unknown_japanese_like` | diagnostic matrix | 5 / 6 | crop + `none` / `upscale_2x` | low | Use diagnostics only; do not promote. |

## Preprocessing Profiles

### `none`

Use first for clean crops and vertical Japanese. The vertical sidebar performed best with the original crop.

### `grayscale`

Useful as a conservative control. It should not alter text geometry and is unlikely to damage source text, but it was not the best profile in the region-specific review.

### `threshold`

Potentially useful for low-contrast scans, but risky for small diacritics, punctuation, and degraded text. Do not make it a default profile before more samples are tested.

### `upscale_2x`

Most useful profile in the current region review for small horizontal Japanese and mixed English/Japanese parentheticals. This should be the first enhancement profile for modern Japanese crops.

### `upscale_3x`

Useful as a diagnostic variant, but more expensive and not clearly better than `upscale_2x` in current results.

### `contrast_sharpen`

Helped the DFD calligraphy sample slightly, but did not make the output reliable. Treat as a diagnostic profile for difficult regions rather than a general default.

## Output / Metadata Design

Do not add these fields to the canonical model yet. For future review-mode output, region OCR diagnostics could use metadata such as:

```text
region_ocr_attempts
region_ocr_best_result
region_ocr_language
region_ocr_psm
region_ocr_preprocess_profile
region_orientation
japanese_region_type
needs_human_review
```

Recommended diagnostic shape:

```json
{
  "japanese_region_type": "vertical_modern_japanese",
  "region_orientation": "vertical",
  "region_ocr_best_result": {
    "engine": "tesseract",
    "language": "jpn_vert",
    "psm": 5,
    "preprocess_profile": "none",
    "text": "...",
    "expected_terms_recovered": ["忍者", "伊賀", "甲賀"],
    "confidence": null
  },
  "region_ocr_attempts": [
    {
      "language": "jpn_vert",
      "psm": 5,
      "preprocess_profile": "none",
      "text_length": 97,
      "japanese_char_count": 59
    }
  ],
  "needs_human_review": true
}
```

Keep this compact. Do not dump large crop images or every OCR candidate into `data.json` by default.

## Runtime Integration Proposal

### Phase 1: Experiment-only region OCR runner

Keep Japanese region OCR in `experiments/`.

The runner reads a local region manifest, applies explicit routing profiles, writes ignored crops/results, and produces a review note. This avoids changing normal runtime behavior while building evidence.

### Phase 2: Optional review-mode region OCR on manually selected regions

Add an optional review-mode path that accepts manually selected Japanese region manifests. It should attach compact diagnostics to review output only.

This phase should still avoid automatic region detection and should not promote Japanese analysis into canonical fields.

### Phase 3: Automatic routing after evaluated regions exist

Only after enough labeled/evaluated regions exist should the system consider automatic routing from detected Japanese-bearing regions.

Automatic routing needs evidence for:

- region selection precision,
- horizontal vs vertical orientation,
- preprocessing profile choice,
- false-positive rate,
- and downstream artifact usefulness.

## What Not To Automate Yet

- Do not auto-detect every Japanese region yet.
- Do not promote Japanese analysis into canonical model fields yet.
- Do not rely on full-page `eng+jpn` for Japanese extraction.
- Do not treat stylized calligraphy as reliable OCR output with the current Tesseract path.
- Do not change normal OCR defaults for English-heavy pages.
- Do not add OCR engine dependencies to normal tests/runtime.
- Do not use unknown-provenance datasets for benchmark claims.

## Test Strategy

Future implementation should use synthetic fixtures and fake routing inputs. It should not require OCR binaries in normal tests.

Proposed tests:

- Routing matrix selects `jpn_vert` + PSM 5 for `vertical_modern_japanese`.
- Horizontal Japanese routes to `jpn` + crop profile.
- Mixed English/Japanese routes to a Japanese-region crop path while preserving English full-page OCR separately.
- Stylized calligraphy routes to `needs_review`.
- Romanized macron regions do not get forced through `jpn_vert`.
- Preprocessing profile selection is deterministic.
- Diagnostics are compact.
- Normal runtime OCR defaults remain unchanged.

## Future Backlog

- Build a larger local Japanese region manifest with expected terms.
- Add manifest examples for horizontal, vertical, mixed, macron, and calligraphy regions.
- Compare EasyOCR on the same region manifest without making it a runtime dependency.
- Compare PaddleOCR or another modern Japanese OCR path in an isolated eval environment.
- Add a small review UI/export loop for manually adjusting Japanese region crops.
- Add ground-truth transcription for selected modern Japanese regions.
- Evaluate whether romanized macron OCR is an English OCR problem, a preprocessing problem, or a sample-selection problem.

## Recommended Next Implementation Pass

Refactor `experiments/review_japanese_region_ocr.py` to make routing profiles explicit and reusable for experiments only.

The next helper should accept region type/orientation metadata from the local manifest and apply the routing matrix automatically:

```text
horizontal_modern_japanese -> jpn, PSM 6, none/upscale_2x
vertical_modern_japanese   -> jpn_vert, PSM 5, none/upscale_2x
mixed_english_japanese     -> jpn, PSM 6, none/upscale_2x, compare eng+jpn
stylized_calligraphy       -> diagnostics only, mark needs_review
unknown_japanese_like      -> small diagnostic matrix
```

Keep it experiment-only. Do not wire this into runtime or canonical fields until a larger evaluated region set supports it.
