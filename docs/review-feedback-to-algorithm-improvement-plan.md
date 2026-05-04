# Review Feedback to Algorithm Improvement Plan

## Purpose

The local research review workbench is not only a way to correct pages. It is also a way to collect reviewed examples that can later improve region detection, OCR routing, and layout models.

The immediate workbench goal remains practical review: load a page, inspect suggestions, draw or correct meaningful regions, and save clean review state. The longer-term goal is to preserve those decisions as evidence for safer algorithm improvement.

## Core Idea

Machine recognition is advisory. Reviewer decisions are the ground truth source.

Reviewed regions should be preserved in a way that can later become evaluation or training data. The system should keep both sides of the review:

- what the machine proposed;
- what the reviewer accepted, resized, retyped, rejected, ignored, or manually added.

This avoids tuning detectors from isolated anecdotes. A reviewed workbench history can become a concrete corpus of positives, negatives, boundary corrections, and missed-region examples.

## Feedback Events Captured by the Workbench

Useful feedback events include:

- `machine_region_accepted`
- `machine_region_resized`
- `machine_region_retyped`
- `machine_region_rejected`
- `manual_region_added`
- `region_ignored`
- `region_split_needed`
- `region_duplicate_created`
- `ocr_result_accepted`
- `ocr_result_edited`
- `ocr_result_rejected`
- `translation_result_accepted`
- `translation_result_rejected`

Not every event needs a separate UI control in the first version. Many can be inferred from preserved detector fields plus reviewer fields.

## Region Review Outcomes

Suggested outcome categories:

- `true_positive`: machine proposed region was accepted.
- `resized_positive`: machine proposed region was useful, but the boundary was corrected.
- `false_positive`: machine proposed region was rejected or ignored.
- `missed_positive`: reviewer manually added a region not proposed by the machine.
- `type_error`: machine proposed a region, but reviewer changed its type.
- `hard_negative`: machine proposed an image/diagram/photo region that was actually paragraph text or another non-target.
- `uncertain`: reviewer marked the region as needs-review or deferred judgment.

For the current page-review workflow, a manually drawn missing side figure would become a `missed_positive`; a rejected paragraph box proposed as an image would become a `hard_negative` or `false_positive`.

## Suggested Feedback Metadata

For missed positives:

```json
{
  "feedback_type": "missed_positive",
  "target_type": "diagram",
  "source_region_id": null,
  "reviewed_region_id": "manual_003",
  "detector": "review_mode_extraction",
  "page_id": "page_025",
  "bbox": [123, 853, 340, 455],
  "notes": "Left figure missed by recognition",
  "reviewed_by": "local_user",
  "reviewed_at": "2026-05-04T00:00:00"
}
```

For rejected machine regions:

```json
{
  "feedback_type": "false_positive",
  "source_region_id": "det_009",
  "detected_type": "diagram",
  "reviewed_type": "ignore",
  "reason": "paragraph_text",
  "bbox": [91, 75, 784, 758],
  "page_id": "page_025"
}
```

For edited OCR attempts:

```json
{
  "feedback_type": "ocr_result_edited",
  "attempt_id": "ocr_001",
  "region_id": "r_004",
  "raw_text": "Le opmgageet I'm glad you're frank...",
  "reviewed_text": "[Question.]\nAaaa yes. I'm glad you're frank...",
  "review_status": "edited",
  "source_text_mutated": false,
  "reason": "dirty_typewritten_first_line",
  "page_id": "page_025"
}
```

Short-term workbench state should preserve enough information to derive these records:

- `source`: `machine_detection`, `reviewer_manual`, `reviewer_manual_duplicate`, or `reviewer_override`;
- `detected_bbox`;
- `reviewed_bbox`;
- `effective_bbox`;
- `detected_type`;
- `reviewed_type`;
- `effective_type`;
- `review_status`;
- OCR attempt `text` / `cleaned_text`;
- OCR attempt `reviewed_text`;
- OCR attempt `source_text_mutated=false`;
- `training_feedback`;
- `notes`.

## Export Formats for Future Training/Evaluation

Future export targets:

- `reviewed_regions.json`
- `detector_feedback.json`
- `hard_negatives.json`
- `image_region_training_examples.json`
- COCO-style annotations
- YOLO-style labels
- regression-fixture manifests

These should be explicit exports. Do not commit private source images or local review projects by default.

## How Feedback Improves Algorithms

Reviewed feedback can support:

- regression tests for pages where the detector previously failed;
- precision/recall measurement against reviewed regions;
- safer threshold tuning based on batches, not one-off pages;
- identification of common false-positive patterns;
- identification of common missed-positive patterns;
- comparison of Paddle, DocLayout, YOLO-style, or other layout models against human-reviewed boxes;
- future fine-tuning if enough reviewed annotations accumulate.

The loop should keep algorithm work evidence-driven: detector changes should improve reviewed-page metrics and should not reintroduce known false positives.

## What Not To Automate Yet

- Do not train live from reviewer changes.
- Do not change detector thresholds from one page.
- Do not auto-correct regions without review.
- Do not treat unreviewed machine regions as ground truth.
- Do not export private source images unless explicitly intended.
- Do not silently rewrite source images or original OCR text.

## Short-Term Workbench Implications

The workbench UI should remain centered on direct review:

- draw/select region modes;
- right-panel region inventory;
- quick type assignment;
- editable boundaries;
- notes;
- source/evidence audit fields;
- duplicate/nudge controls available as advanced helpers.

The state model should preserve feedback metadata, but the UI should not become a training-console. The reviewer’s job is to produce clean reviewed page state.

## Medium-Term Algorithm Improvement Loop

1. Run recognition.
2. Reviewer corrects regions.
3. Export feedback.
4. Analyze false positives, missed positives, type errors, and boundary deltas.
5. Add regression tests or detector improvements.
6. Re-run reviewed pages.
7. Compare new results against previous results and reviewed ground truth.

This creates a disciplined improvement loop:

```text
manual review now
  -> better reviewed data
  -> safer algorithm improvements later
```

## Future Backlog

- Export reviewed region feedback from `project_state.json`.
- Build a reviewed-page regression runner.
- Add metrics for region precision/recall and bbox overlap.
- Export hard-negative text/paragraph examples.
- Export missed-positive image/diagram/photo examples.
- Compare alternate layout engines against reviewed regions.
- Add optional COCO/YOLO export after enough pages are reviewed.
- Track reviewer identity locally when useful for trusted-collaborator workflows.

## Recommended Next Implementation Pass

Workbench region review simplification:

- draw/select region modes;
- right-panel region inventory;
- duplicate/nudge controls moved into Advanced;
- preserved feedback metadata for accepted, resized, rejected, manually added, and ignored regions.

After enough pages are reviewed, the next evidence pass should be a small export utility that reads local `project_state.json` files and writes `detector_feedback.json` without copying private images.
