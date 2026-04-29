# Paddle Fusion Failure Analysis

## Purpose

Analyze why containment-only Paddle fusion improved only 2/5 Corpus 2
broad/mixed cases.

## Source Data

- Validation note:
  `data/evaluation/real_page_review/notes/2026-04-28-paddle-fusion-validation.md`
- Layout evaluation note:
  `data/evaluation/real_page_review/notes/2026-04-28-paddle-layout-eval.md`
- Comparison JSON used:
  `data/notebook_outputs/paddle_layout_eval/comparison.json`
- Was diagnostic export added? no
- Notes: existing comparison JSON contained Paddle bboxes, raw labels,
  normalized labels, and confidence scores. Fusion metrics were computed from
  those bboxes and the recorded classical bboxes; no detector/fusion decisions
  were changed.

## Summary

| Case | Sample ID | Failure Type | Classical Issue | Paddle Output | Fusion Issue | Recommended Next Rule |
|---|---|---|---|---|---|---|
| 1 | `corpus2_new_doc_2026_04_28_17_10_58` | B/F | Classical parent is a narrow crop inside a larger visual/photo area. | One broad `image` region mostly covering the page. | Paddle region fully covers the classical crop but is larger, so V1 tightening correctly refuses it. | Evaluate larger-but-better replacement only with text-separation evidence; otherwise leave as review metadata. |
| 2 | `corpus2_new_doc_2026_04_28_17_19_36` | A/F | Classical parent only partially covers the right-side photo region. | Useful `image` bbox with separated text regions. | Paddle bbox is only `0.684` inside the classical parent, below the `0.75` containment gate. | Add independent Paddle visual region or allow near-overlap addition when Paddle is high-confidence and text is separately classified. |
| 3 | `corpus2_new_doc_2026_04_28_18_54_00` | B/F | Classical parent is a partial photo-grid crop. | One broad `image` region around the full grid plus header area. | Paddle region contains the classical crop but is much larger, so it is not a tightening replacement. | Evaluate larger-but-better replacement for partial parents, but preserve `needs_review` because Paddle crop includes header text. |

## Per-Case Analysis

### `corpus2_new_doc_2026_04_28_17_10_58`

- Input path:
  `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 17.10.58.jpg`
- Classical bbox(es): `(108, 102, 355, 148)`
- Paddle bbox(es):
  - `text`: `(346, 33, 91, 16)`, confidence `0.568`
  - `image`: `(74, 60, 407, 567)`, confidence `0.507`
  - `paragraph_title`: `(77, 8, 272, 32)`, confidence `0.321`
- Current fused bbox(es): unchanged `(108, 102, 355, 148)`
- Paddle raw labels: `text`, `image`, `paragraph_title`
- Paddle normalized labels: `text`, `image`, `title`
- Was Paddle useful? partial
- Failure type: B/F
- Why did containment-only fusion fail or underperform? The Paddle image region
  covers the full classical crop (`classical_covered_by_paddle_ratio=1.0`) but
  is much larger than the classical parent (`area_tightness_ratio=4.392`) and
  only `0.228` of the Paddle region lies inside the parent. V1 correctly rejects
  this as a tightening replacement.
- Recommended next rule: consider larger-but-better replacement only when
  Paddle separately identifies surrounding text/title and the resulting crop is
  useful enough for review. This rule needs annotation or a stronger visual/text
  separation gate before implementation.

### `corpus2_new_doc_2026_04_28_17_19_36`

- Input path:
  `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 17.19.36.jpg`
- Classical bbox(es): `(96, 10, 359, 400)`
- Paddle bbox(es):
  - `image`: `(322, 224, 160, 226)`, confidence `0.916`
  - multiple `text` regions around the page
  - multiple `paragraph_title` regions near the top
- Current fused bbox(es): unchanged `(96, 10, 359, 400)`
- Paddle raw labels: `image`, `text`, `paragraph_title`
- Paddle normalized labels: `image`, `text`, `title`
- Was Paddle useful? yes
- Failure type: A/F
- Why did containment-only fusion fail or underperform? The Paddle image region
  appears to isolate the right-side visual content, but the recorded classical
  parent does not contain enough of it. The measured
  `paddle_inside_classical_ratio` is `0.684`, below the `0.75` gate, with
  `area_tightness_ratio=0.252` and `iou=0.160`. Lowering containment might make
  this one pass, but it would also weaken the safety guarantee globally.
- Recommended next rule: add independent high-confidence Paddle visual regions
  near or overlapping mixed classical regions when Paddle also separates text.
  This is not a replacement rule; it is additive layout evidence.

### `corpus2_new_doc_2026_04_28_18_54_00`

- Input path:
  `data/corpora/ad_hoc/corpus2/original/new doc 2026-04-28 18.54.00.jpg`
- Classical bbox(es): `(53, 44, 139, 170)`
- Paddle bbox(es):
  - `image`: `(31, 38, 400, 595)`, confidence `0.962`
  - `text`: `(41, 15, 390, 24)`, confidence `0.430`
- Current fused bbox(es): unchanged `(53, 44, 139, 170)`
- Paddle raw labels: `image`, `text`
- Paddle normalized labels: `image`, `text`
- Was Paddle useful? partial
- Failure type: B/F
- Why did containment-only fusion fail or underperform? The Paddle image region
  covers the full classical crop (`classical_covered_by_paddle_ratio=1.0`) but
  is much larger (`area_tightness_ratio=10.072`) and only `0.099` of the Paddle
  region lies inside the classical parent. This looks like a partial classical
  parent plus a broader Paddle photo-grid crop, not a tight replacement.
- Recommended next rule: evaluate larger-but-better replacement for partial
  classical parents only when the Paddle visual region is semantically useful
  and clearly separated from text. Keep `needs_review` because the Paddle crop
  still includes header/text material.

## Cross-Case Findings

- Dominant failure type: F, with B in two cases and A in one case.
- Does Paddle provide useful signal in the failed cases? yes, but the useful
  signal is not always a tighter bbox inside the classical parent.
- Is V2 fusion justified? yes, but not as a simple threshold tweak. The failed
  cases need additive or larger-region logic, not weaker containment-only
  replacement.
- If yes, which specific V2 rule should be implemented first? The safest first
  rule is additive Paddle visual regions near mixed classical parents when
  Paddle also separates text. Larger-but-better replacement is riskier and
  should wait for annotation or explicit page-level success criteria.
- If no, should we pivot to annotation / ground truth? Annotation is still the
  best way to decide whether larger Paddle crops are better or just differently
  broad. The current data is enough to design a narrow V2 experiment, not enough
  to make Paddle fusion preferred.

## Proposed Next Step

- [x] Implement V2 fusion with Paddle visual additions
- [ ] Implement V2 fusion with multi-region support
- [ ] Implement V2 fusion with larger-but-better replacement
- [ ] Improve Paddle label mapping
- [ ] Keep V1 fusion experimental; gate was too strict for V1
- [ ] Do not expand fusion; move to annotation/ground truth
- [ ] Need more diagnostics before implementation

V2 should be a small, explicitly gated experiment. It should add Paddle visual
regions as additional review candidates only when they are high-confidence,
near/overlapping an unresolved mixed classical region, and Paddle has separate
text/title evidence on the page. It should not silently replace classical
regions or enable Paddle by default.
