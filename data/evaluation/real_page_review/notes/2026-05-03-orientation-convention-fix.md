# Orientation Convention Fix

## Purpose

Record the workbench orientation convention bug and the fix.

## Observed Failure

A page that was visibly rotated `270°` was correctly identified by the NN as `270°`, but the workbench applied `270°` as the display rotation. The page ended up at `180°` instead of upright.

## Root Cause

The NN output is the current orientation of the source image:

```text
current_orientation_degrees
```

The workbench incorrectly treated that value as the correction rotation to apply.

For a `270°` source image, applying another `270°` gives:

```text
270 + 270 = 540
540 mod 360 = 180
```

That matches the observed failure.

## Correct Mapping

The workbench now computes the correction rotation as the inverse of the NN output:

```text
correction_rotation_degrees = (360 - current_orientation_degrees) % 360
```

Mapping:

| NN current orientation | Workbench correction |
|---:|---:|
| 0 | 0 |
| 90 | 270 |
| 180 | 180 |
| 270 | 90 |

## State Convention

The page orientation state keeps both concepts visible:

```json
{
  "detected_rotation_degrees": 270,
  "reviewed_rotation_degrees": 90,
  "effective_rotation_degrees": 90,
  "metadata": {
    "model_output_convention": "current_orientation_degrees",
    "rotation_convention": "clockwise_correction_to_apply"
  }
}
```

`detected_rotation_degrees` remains the model's current-orientation class. `effective_rotation_degrees` is the clockwise correction used for display, recognition, and later OCR crops.

## Manual Verification

Use a page visibly rotated `270°`:

1. Load the page in the local review workbench.
2. Click `Run Orient`.
3. Confirm the UI shows detected current `270°`.
4. Confirm the UI shows correction `90°`.
5. Confirm the displayed page is upright.
6. Run recognition only after the orientation looks correct.

## Non-Changes

- No model retraining.
- No checkpoint movement.
- No heuristic replacement.
- No OCR behavior change.
- No extraction behavior change.
- Original source images remain unmodified.
