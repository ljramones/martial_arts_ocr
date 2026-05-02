# Real Japanese Sample Curation Plan

## Purpose

Curate real project samples for modern Japanese OCR, romanized Japanese, macron, and mixed-layout evaluation before adding more pipeline logic.

The current tools are ahead of the evidence. This plan identifies a small set of real pages/images that should be used to evaluate Japanese OCR routing, macron candidate review, line ordering, and review/export behavior before any runtime OCR changes or canonical Japanese fields are added.

## Scope

This pass covers sample selection only.

In scope:

- project-native DFD and Corpus 2 pages already present under `data/corpora/`;
- tracked manual modern Japanese samples under `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/`;
- local ignored manifests that reference existing source paths.

Out of scope:

- broad OCR runs;
- new OCR engine work;
- schema/model changes;
- runtime OCR routing;
- copying or moving private image payloads;
- using `img_ocr_jp_cn` for benchmark claims while provenance/license remain unknown.

## Sources Inspected

| Source | Inspected | Notes |
|---|---|---|
| `data/corpora/donn_draeger/dfd_notes_master/` | yes | Project-native DFD pages and manifest references. |
| `data/corpora/ad_hoc/corpus2/` | yes | Project-native Corpus 2 pages and manifest references. |
| `data/corpora/modern_japanese_ocr/` | yes | Organized corpus area, manual samples, existing ignored local manifests. |
| `data/evaluation/real_page_review/notes/` | yes | Used prior OCR reviews as source evidence instead of re-running OCR. |
| `docs/japanese-ocr-experimental-state-2026-04-28.md` | yes | Routing profile evidence and known gaps. |
| `docs/macron-candidate-workflow-state-2026-04-28.md` | yes | Real macron candidate evidence and review workflow state. |
| `docs/modern-japanese-ocr-corpus-plan.md` | yes | Dataset handling rules and provenance cautions. |

Existing ignored local manifests were inspected:

```text
data/corpora/modern_japanese_ocr/project_native/manifests/manifest.local.json
data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/manifests/manifest.local.json
data/corpora/modern_japanese_ocr/manifests/japanese_region_manifest.local.json
data/corpora/modern_japanese_ocr/manifests/japanese_region_routing_validation.local.json
```

No new source images were moved or copied.

## Selection Method

Samples were selected from prior visual/OCR review evidence using these priorities:

1. Confirmed visible Japanese from focused Japanese OCR review.
2. Region-specific evidence where expected Japanese terms are already recorded.
3. Project-native pages with romanized Japanese terms or all-caps martial arts terms that exercise macron candidate review.
4. Pages with captions, labels, diagrams, or mixed layouts that are likely to expose reading order and region-routing issues.
5. Difficult qualitative stress cases such as stylized calligraphy and noisy visual layouts.

The curated set intentionally remains small. It is not a benchmark. It is a practical next evaluation set.

## Curated Sample Set

| Sample ID | Corpus | Path | Visible Japanese | Visible Macrons | Romanized Terms | Challenge Type | Why Selected |
|---|---|---|---|---|---|---|---|
| `original_img_3288` | DFD | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3288.jpg` | no | no | yes | `macron`, `mixed_layout` | Source image contains `BUJUTSU AND BUDO`; useful for real OCR macron-candidate review and casing behavior. |
| `original_img_3330` | DFD | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg` | uncertain | no | yes | `caption_or_label`, `mixed_layout` | Caption/figure page with terms such as `ryu` and `Kamakura`; useful for captions and visual-page reading order. |
| `original_img_3336` | DFD | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3336.jpg` | yes | no confirmed | yes | `stylized_calligraphy`, `romanized_japanese` | Large Japanese calligraphy plus romanized terms such as `katsujin no ken`, `satsujin no to`, and `Yagyu Shinkage Ryu`. |
| `original_img_3344` | DFD | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | yes/uncertain | no | yes | `caption_or_label`, `mixed_layout` | Diagram/label page with Japanese-like labels and romanized terms; useful difficult full-page OCR case. |
| `corpus2_new_doc_2026_04_28_16_56_38` | Corpus 2 | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.56.38.jpg` | uncertain | no | yes/uncertain | `noisy_scan`, `mixed_layout` | Prior `eng+jpn` run produced noisy kana-like output; useful as a noisy false-positive/control page. |
| `corpus2_new_doc_2026_04_28_18_29_28` | Corpus 2 | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg` | uncertain | no | yes | `caption_or_label`, `mixed_layout` | Article/photo page with Japanese martial arts names such as `SATO KINBEI SENSEI`; useful for caption and mixed article layout review. |
| `corpus2_new_doc_2026_04_28_18_54_34` | Corpus 2 | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.34.jpg` | yes | no | yes | `parenthetical_japanese`, `mixed_layout` | Dense martial arts term list with Japanese parentheticals; region crop recovered `術`, `剣`, `刀`, `弓`, `火`, `水`, and `馬`. |
| `corpus2_new_doc_2026_04_28_19_40_50` | Corpus 2 | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.40.50.jpg` | yes | no | uncertain | `clean_horizontal_japanese`, `noisy_scan` | Large Japanese cover/title text; prior full-page OCR failed and may need crop/orientation/preprocessing. |
| `corpus2_new_doc_2026_04_28_19_41_28` | Corpus 2 | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 19.41.28.jpg` | yes | no | uncertain | `mixed_layout`, `clean_horizontal_japanese` | Bansenshukai page with Japanese blocks and English translation; likely needs region-specific OCR. |
| `manual_modern_japanese_japtext2` | Manual modern Japanese | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/JapText2.jpg` | yes | no | no | `clean_horizontal_japanese` | Clean horizontal modern Japanese sample; useful smoke test despite unclear provenance. |
| `manual_modern_japanese_s_34193423` | Manual modern Japanese | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/S__34193423.jpg` | yes | no | no | `vertical_japanese`, `mixed_layout` | Vertical Japanese sidebar/header sample; `jpn_vert` + PSM 5 recovered `忍者`, `伊賀`, and `甲賀` in region evaluation. |
| `manual_modern_japanese_istock_calligraphy` | Manual modern Japanese | `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/original/istockphoto-1015263364-612x612.jpg` | yes | no | no | `stylized_calligraphy` | Stylized vertical calligraphy-like sample; useful as a hard qualitative stress case, not quantitative evidence. |

## Challenge Coverage

| Challenge Type | Count | Samples |
|---|---:|---|
| `macron` | 1 | `original_img_3288` |
| `parenthetical_japanese` | 1 | `corpus2_new_doc_2026_04_28_18_54_34` |
| `vertical_japanese` | 1 | `manual_modern_japanese_s_34193423` |
| `caption_or_label` | 3 | `original_img_3330`, `original_img_3344`, `corpus2_new_doc_2026_04_28_18_29_28` |
| `mixed_layout` | 7 | `original_img_3288`, `original_img_3330`, `original_img_3344`, `corpus2_new_doc_2026_04_28_16_56_38`, `corpus2_new_doc_2026_04_28_18_29_28`, `corpus2_new_doc_2026_04_28_18_54_34`, `manual_modern_japanese_s_34193423` |
| `stylized_calligraphy` | 2 | `original_img_3336`, `manual_modern_japanese_istock_calligraphy` |
| `clean_horizontal_japanese` | 3 | `corpus2_new_doc_2026_04_28_19_40_50`, `corpus2_new_doc_2026_04_28_19_41_28`, `manual_modern_japanese_japtext2` |
| `noisy_scan` | 2 | `corpus2_new_doc_2026_04_28_16_56_38`, `corpus2_new_doc_2026_04_28_19_40_50` |
| `unknown` | 0 | none |

Coverage notes:

- Confirmed visible Japanese appears in 8 of 12 samples.
- Confirmed visible macronized romanization still has not been found.
- `original_img_3288` is the only real source image so far that has produced a reviewed macron candidate from OCR text (`BUDO -> BUDŌ`), but the source itself does not visibly contain a macron.
- Manual modern Japanese samples are useful qualitative cases, but provenance remains unclear.

## Known Gaps

- No confirmed real image sample with visible macronized romanization.
- No broad set of project-native clean horizontal Japanese regions.
- Only one validated vertical Japanese crop.
- Only one validated Japanese parenthetical crop.
- Stylized/calligraphic Japanese remains difficult and lacks reliable ground truth.
- `img_ocr_jp_cn` may contain useful image+annotation material, but provenance/license remain unresolved.
- Existing project-native manifests reference pages, not reviewed region boxes for every challenge type.

## Recommended Evaluation Order

1. `clean_horizontal_japanese`
   - Start with `manual_modern_japanese_japtext2`, then Corpus 2 Japanese block/title pages.
   - Goal: verify `jpn`, PSM 6 behavior on simpler modern Japanese text.

2. `parenthetical_japanese`
   - Start with `corpus2_new_doc_2026_04_28_18_54_34`.
   - Goal: validate `mixed_japanese_parentheticals` routing and term recovery.

3. `vertical_japanese`
   - Start with `manual_modern_japanese_s_34193423`.
   - Goal: validate `jpn_vert`, PSM 5 and crop sensitivity.

4. `mixed_layout`
   - Use DFD and Corpus 2 mixed pages after simple/cropped cases.
   - Goal: identify when line ordering or region routing is the blocker.

5. `caption_or_label`
   - Use DFD diagram/caption pages and Corpus 2 photo/article pages.
   - Goal: test whether labels/captions need separate OCR/review handling.

6. `stylized_calligraphy`
   - Use `original_img_3336` and the manual calligraphy sample only as hard diagnostics.
   - Goal: avoid treating stylized calligraphy as solved by current Tesseract routing.

7. `macron`
   - Re-run only when a source image with visible macronized romanization is found.
   - Current real source evidence supports candidate review, not OCR macron recognition.

## Suggested Tooling

Use existing tools only:

```text
experiments/review_real_ocr_text_quality.py
experiments/review_japanese_region_ocr.py
experiments/review_macron_candidates.py
experiments/compare_macron_ocr_engines.py
```

Suggested local manifest usage:

- Use `data/corpora/modern_japanese_ocr/project_native/manifests/manifest.local.json` for project-native page references.
- Use `data/corpora/modern_japanese_ocr/external/manual_modern_japanese_samples/manifests/manifest.local.json` for the three tracked manual samples.
- Use `data/corpora/modern_japanese_ocr/manifests/japanese_region_manifest.local.json` for manually selected region boxes.
- Keep all `*.local.json` manifests ignored and uncommitted.

Recommended next local manifest change:

```text
Add original_img_3288, original_img_3330, corpus2_new_doc_2026_04_28_16_56_38, and corpus2_new_doc_2026_04_28_18_29_28 to the project_native local manifest if the next evaluator wants one command over this full curation set.
```

This pass did not update local manifests because the existing ignored manifests already cover the highest-confidence Japanese samples and region boxes.

## Do Not Do Yet

- Do not reopen image-region extraction.
- Do not add runtime Japanese OCR routing.
- Do not promote Japanese or romanization fields into canonical models.
- Do not add automatic macron normalization.
- Do not treat external/provenance-unknown datasets as benchmark evidence.
- Do not train or fine-tune models.
- Do not commit private source images, generated OCR outputs, or local manifests.

## Next Step

Run a focused evaluation over this curated set, using the existing experiment tools:

```text
page-level review:
  review_real_ocr_text_quality.py

region-level Japanese review:
  review_japanese_region_ocr.py

macron candidate review:
  review_macron_candidates.py
```

Primary decision question:

```text
Do better real samples change the blocker from sample availability to OCR routing, preprocessing, engine choice, or review/export workflow?
```

If this curation set still does not expose real macron-bearing source images, keep macron work in the review-only candidate workflow and prioritize Japanese region OCR evidence instead.
