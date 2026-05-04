# Review Workbench Direct Region Drawing

## Purpose

Make visual region creation the primary workbench workflow. Numeric bbox entry should remain available for fine adjustment, but a reviewer should be able to draw boxes directly on the scanned page.

## Why This Was Added

The workbench is meant for research review, not operating every detector control manually. Pages with missed figures or labels should be handled by drawing the real regions quickly:

```text
choose drawing tool
drag rectangle on page
release mouse
new reviewed region is created
repeat as needed
```

This also captures the strongest feedback signal for future detector work: regions the machine missed but the reviewer added.

## Workflow

- `Select / Move`: select, move, and resize existing regions.
- `Draw Image Region`: draw image/diagram/photo candidates.
- `Draw Text Region`: draw English text regions.
- `Draw Japanese Region`: draw horizontal modern Japanese regions by default.
- `Draw Ignore Region`: draw page areas to skip.

After mouse release, the region is created and selected. The current draw mode remains active so the reviewer can draw multiple regions in sequence.

## State Metadata

Manually drawn regions are stored as reviewer-created evidence:

```json
{
  "source": "reviewer_manual",
  "detected_bbox": null,
  "reviewed_bbox": [10, 20, 100, 60],
  "effective_bbox": [10, 20, 100, 60],
  "metadata": {
    "feedback_type": "missed_positive",
    "manually_added": true
  },
  "training_feedback": {
    "label": "manually_added",
    "feedback_type": "missed_positive"
  }
}
```

Ignore regions are created as reviewed ignore regions with `status=ignored`.

## Feedback-Loop Significance

Manual regions are not just corrections for the current page. They can later become:

- missed-positive detector examples;
- reviewed image/text/Japanese annotations;
- regression fixtures;
- layout-model comparison data.

No live training is performed from these records.

## Manual Verification Steps

1. Open `/review`.
2. Load a local project and select a page.
3. Choose `Draw Image Region`.
4. Drag a rectangle on the page and release.
5. Confirm a new region appears and remains selected.
6. Draw a second image region without reselecting the tool.
7. Choose `Draw Ignore Region` and draw an ignored area.
8. Confirm numeric bbox fields reflect the drawn bbox and can still fine-tune it.
9. Reload the project and confirm regions persist.

## Non-Goals

- No OCR behavior changes.
- No recognition/detector changes.
- No orientation changes.
- No OCR-all or translation.
- No generated/private review projects committed.
