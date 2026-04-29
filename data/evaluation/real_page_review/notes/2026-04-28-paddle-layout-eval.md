# Paddle Layout Evaluation

Date: 2026-04-28

## Purpose

Evaluate whether PaddleOCR PP-Structure / PP-DocLayout can split broad mixed
text/figure regions better than the current classical/OCR-aware/mixed-triage
path.

This is an isolated evaluation. It does not change runtime defaults, does not
add PaddleOCR to normal project dependencies, and does not make layout models
required for tests.

## Setup

- Python version: `3.12.13` in `.venv-eval`
- Platform: `macOS-26.3.1-arm64-arm-64bit`
- Architecture: `arm64`
- PaddleOCR version: `3.5.0`
- PaddlePaddle version: `3.3.0`
- CPU/GPU: CPU-only intended
- API used: `PPStructureV3`
- Installation issues:
  - The first sandboxed package install retry failed with the same DNS error as
    the prior attempt.
  - Escalated package install/network access succeeded.
  - PaddlePaddle installed from Paddle's CPU wheel index:
    `.venv-eval/bin/python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`
  - PaddleOCR installed with:
    `.venv-eval/bin/python -m pip install paddleocr`
  - `PPStructureV3` construction required the document-parser extra:
    `.venv-eval/bin/python -m pip install "paddleocr[doc-parser]"`
  - Importing `paddleocr` tried to create `~/.paddlex`; the evaluator now sets
    `PADDLE_PDX_CACHE_HOME` under ignored project output before import.
- Model download/setup notes:
  - `PPStructureV3` downloaded official models into:
    `data/notebook_outputs/paddle_layout_eval/paddlex_cache/official_models/`
  - Model cache size after the run: roughly `1.7G`.
  - The cache is ignored and must not be committed.

Generated run metadata:

```text
data/notebook_outputs/paddle_layout_eval/comparison.json
data/notebook_outputs/paddle_layout_eval/overlays/
data/notebook_outputs/paddle_layout_eval/paddlex_cache/
```

These output paths are gitignored and should not be committed.

## Retry Attempt

- Date/time: 2026-04-29
- Network available: yes, only with escalated command execution
- Python version: `3.12.13`
- Platform: `macOS-26.3.1-arm64-arm-64bit`
- Architecture: `arm64`
- PaddlePaddle install command:
  `.venv-eval/bin/python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`
- PaddlePaddle installed: yes
- PaddlePaddle version: `3.3.0`
- PaddleOCR install command:
  `.venv-eval/bin/python -m pip install paddleocr`
- PaddleOCR installed: yes
- PaddleOCR version: `3.5.0`
- Layout/structure API available: yes, `PPStructureV3`
- Inference ran: yes
- Model download location:
  `data/notebook_outputs/paddle_layout_eval/paddlex_cache/official_models/`
- Setup issues:
  - Requires network/model downloads on first run.
  - Creates many model components for full `PPStructureV3`, including document
    orientation, layout, OCR, table, and formula models.
  - CPU inference is usable but slow enough to remain evaluation-only.

## Result

- [x] Paddle layout ran and produced comparison output
- [ ] PaddleOCR installed, but layout/structure API was unavailable or unclear
- [ ] PaddlePaddle/PaddleOCR install failed
- [ ] Platform architecture appears unsupported
- [ ] Network failure persisted

## Pages Evaluated

| Corpus | Sample ID | Path | Expected Outcome |
|---|---|---|---|
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 16.55.48.jpg` | Photo/visual content should be separated from the surrounding paragraph/body text; success means a figure/photo bbox without the broad mixed parent crop. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_10_58` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 17.10.58.jpg` | Photographic/visual material should be retained without accepting heading or paragraph fragments as figures. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_19_36` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 17.19.36.jpg` | Broad mixed candidate should split into text and visual roles if a real photo/visual region is present. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.29.28.jpg` | Existing broad visual/mixed crop should become a tighter visual bbox. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_00` | `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.00.jpg` | Photo-grid or panel content should be detected; success can be one useful enclosing region or separate useful photo regions. |
| DFD | `original_img_3335` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3335.jpg` | Known large visual should remain detectable without swallowing unrelated body text. |
| DFD | `original_img_3344` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3344.jpg` | Labeled diagram should be retained; labels near the diagram are acceptable, unrelated paragraph text is not. |
| DFD | `original_img_3397` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3397.jpg` | Sparse arrows/symbols should remain detectable as visual layout content. |
| DFD | `original_img_3352` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3352.jpg` | Known visual/annotation areas should remain detectable; label-heavy crops are acceptable if not overly broad. |
| DFD | `original_img_3330` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3330.jpg` | Tall/narrow visual strips should be retained where they are not plain vertical text. |
| DFD known-good | `original_img_3292` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3292.jpg` | Known-good diagram detection should not regress. |
| DFD known-good | `original_img_3340` | `data/corpora/donn_draeger/dfd_notes_master/original/IMG_3340.jpg` | Known-good diagram detection should not regress. |

## Summary Table

| Corpus | Sample ID | Classical/Mixed-Triage Result | Paddle Layout Result | Score 0-2 | Regression? | Notes |
|---|---|---|---|---:|---|---|
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | broad/mixed, `needs_review` | two clean `image` regions plus separate text/title regions | 2 | no | Strong split of two photos from body text. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_10_58` | broad/mixed, `needs_review` | one very broad `image` region over most of page | 1 | no | Finds visual material but still swallows text. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_19_36` | broad/mixed, `needs_review` | right-side photo isolated as `image`; text separated | 2 | no | Clean improvement over broad parent crop. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | broad/mixed, `needs_review` | top-right visual isolated as `image`; text separated | 2 | no | Clean split. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_00` | photo-grid partial, `needs_review` | one broad `image` region around grid | 1 | no | Useful grid retention, but not split into individual panels and includes header text. |
| DFD | `original_img_3335` | known visual retained | broad `image` region retains visual but includes adjacent text | 1 | no | Useful but not clean. |
| DFD | `original_img_3344` | labeled diagram retained | small diagram/visual region partially detected; much of rotated page text is classified as text | 1 | no | Partial; not a clean labeled-diagram crop. |
| DFD | `original_img_3397` | sparse symbols retained | sparse symbols retained, but in broad/partial visual regions | 1 | no | Useful but noisy. |
| DFD | `original_img_3352` | known candidates retained | two broad `image` regions retained | 1 | no | Retains visual content; crop granularity still broad. |
| DFD | `original_img_3330` | tall/narrow strips retained | tall/narrow visual region retained as broad `image` region | 1 | no | Useful recall, not tight. |
| DFD known-good | `original_img_3292` | known-good retained | two main diagram `image` regions retained plus figure-title boxes | 2 | no | Good retention. |
| DFD known-good | `original_img_3340` | known-good retained | diagram/photo region retained with separate text regions | 2 | no | Good retention. |

## Corpus 2 Broad/Mixed Results

Total score: `8/10`.

The five Corpus 2 broad/mixed cases are the strongest result from this
evaluation. PPStructureV3 cleanly separated visual regions from text on
`16_55_48`, `17_19_36`, and `18_29_28`. It partially helped on `17_10_58`,
where it found the photographic area but as one broad page-level `image` crop.
It also partially helped on `18_54_00`, where the photo grid was retained as
one useful region but not split into separate photo panels.

This is materially better than the current classical/OCR-aware/mixed-triage
path for the broad/mixed Corpus 2 problem.

## DFD Hard Page Results

The DFD hard pages were mostly partial. Visual content was retained, but crop
granularity is still broad on rotated or sparse pages:

- `3335`: large visual retained, but text is included in a broad region.
- `3344`: labeled diagram page remains difficult; PPStructureV3 separates many
  text regions but does not produce a clean labeled-diagram crop.
- `3397`: sparse symbol content is retained, but boxes are broad/fragmented.
- `3352`: visual candidates retained, but broad.
- `3330`: tall/narrow visual content retained, but broad.

No serious known-visual regression was observed, but DFD quality is not clean
enough to replace the classical detector outright.

## Known-Good Regression Checks

Known-good pages did not regress:

- `original_img_3292`: two main diagram regions retained cleanly.
- `original_img_3340`: diagram/photo region retained with text separated.

## Runtime Practicality

- Install friction: moderate to high. Needs Paddle's CPU wheel index,
  `paddleocr`, `paddleocr[doc-parser]`, network escalation, and cache redirection
  away from `~/.paddlex`.
- Inference time per page: roughly `5-25s` per page after model download/cache
  warmup on this CPU run.
- Model files required: many official Paddle models, cached under
  `data/notebook_outputs/paddle_layout_eval/paddlex_cache/official_models/`.
- Offline repeatability: likely possible after cache is populated, but not yet
  verified in a network-disabled rerun.
- Memory/CPU pain: acceptable for evaluation, too heavy for default runtime.

## Decision

- [ ] Paddle layout works well enough to become an optional strategy later
- [x] Paddle layout partially helps; fusion should be evaluated
- [ ] Paddle layout unavailable or setup failed; try DocLayout-YOLO next
- [ ] Paddle layout does not help; annotation/fine-tuning path is justified

## Recommendation

PPStructureV3 is promising specifically for Corpus 2 broad/mixed parent crops.
It should not replace the current detector directly, but it is strong enough to
justify a later optional strategy/fusion experiment.

Recommended next step: add a comparison/fusion analysis that uses Paddle layout
`image` boxes as optional review-mode proposals alongside the current
classical/OCR-aware detector. Keep runtime defaults unchanged and keep Paddle
out of normal dependencies/tests.
