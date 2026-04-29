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
- PaddleOCR version: unavailable; package could not be installed
- PaddlePaddle version: unavailable; package could not be installed
- CPU/GPU: CPU-only intended
- API used: none; evaluator would prefer `PPStructureV3` if exposed, then fall
  back to older `PPStructure`
- Installation issues:
  - `.venv-eval` was created with the project Python 3.12 interpreter.
  - `pip install paddleocr` failed because the environment could not reach
    PyPI:
    `Failed to establish a new connection: [Errno 8] nodename nor servname provided, or not known`.
  - The same install command was retried with escalated permissions and failed
    with the same network/DNS error.
- Model download/setup notes: no PaddleOCR package or model files were
  downloaded.

Generated skipped-run metadata:

```text
data/notebook_outputs/paddle_layout_eval/comparison.json
```

That output path is gitignored and should not be committed.

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
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | broad/mixed, `needs_review` | not run; PaddleOCR unavailable | n/a | n/a | |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_10_58` | broad/mixed, `needs_review` | not run; PaddleOCR unavailable | n/a | n/a | |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_19_36` | broad/mixed, `needs_review` | not run; PaddleOCR unavailable | n/a | n/a | |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | broad/mixed, `needs_review` | not run; PaddleOCR unavailable | n/a | n/a | |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_00` | photo-grid partial, `needs_review` | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD | `original_img_3335` | known visual retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD | `original_img_3344` | labeled diagram retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD | `original_img_3397` | sparse symbols retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD | `original_img_3352` | known candidates retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD | `original_img_3330` | tall/narrow strips retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD known-good | `original_img_3292` | known-good retained | not run; PaddleOCR unavailable | n/a | n/a | |
| DFD known-good | `original_img_3340` | known-good retained | not run; PaddleOCR unavailable | n/a | n/a | |

## Corpus 2 Broad/Mixed Results

Not run. PaddleOCR could not be installed in `.venv-eval` because PyPI was not
reachable from this environment.

## DFD Hard Page Results

Not run. PaddleOCR could not be installed in `.venv-eval`.

## Known-Good Regression Checks

Not run. PaddleOCR could not be installed in `.venv-eval`.

## Runtime Practicality

- Install friction: blocking in this environment because package resolution
  could not reach PyPI. No Paddle API could be observed locally.
- Inference time per page: not measured
- Model files required: unknown; no models downloaded
- Offline repeatability: not established
- Memory/CPU pain: not measured

## Decision

- [ ] Paddle layout works well enough to become an optional strategy later
- [ ] Paddle layout partially helps; fusion should be evaluated
- [x] Paddle layout unavailable or setup failed; try DocLayout-YOLO next
- [ ] Paddle layout does not help; annotation/fine-tuning path is justified

## Recommendation

This pass did not produce a PP-Structure / PP-DocLayout quality result because
PaddleOCR could not be installed. The evaluation harness is in place and skips
cleanly when PaddleOCR is unavailable.

Recommended next step: either rerun this evaluator in an environment with
PaddleOCR 3.x and its inference dependencies already installed, or pivot to the
DocLayout-YOLO setup/evaluation path. Do not add more classical detector
heuristics based on this failed Paddle setup.
