# Workbench Recognition Diagnostics Pass

## Purpose

Add observability for the review workbench recognition candidate lifecycle.

## Observed Issue

A corrected/upright page contained multiple figure/photo regions in a row. The logs showed several contour `diagram` candidates and `ContourDetector found 6 regions`, but the workbench overlays showed only a subset.

This is not an orientation issue. Orientation is already corrected before recognition.

## Diagnostics Added

Workbench recognition now stores compact diagnostics on page state:

```text
pages[].recognition_diagnostics
```

The diagnostics include:

- raw candidate count;
- accepted count;
- rejected count;
- suppressed count;
- merged count;
- imported count;
- candidate records with stage, reason, bbox, type, confidence, and metadata.

Candidate stages include:

```text
raw
accepted
rejected
suppressed
merged
imported
```

Detector-level diagnostics preserve contour top-k suppressions when available.

## UI

The local review workbench now shows a `Recognition Diagnostics` panel with lifecycle counts and a candidate list.

This is diagnostic only. Imported-region behavior is unchanged.

## How To Inspect Candidate Loss

After `Run Recognition`, inspect the diagnostics panel or `project_state.json`.

Use the candidate stages to determine:

- raw candidate missing: improve candidate generation;
- rejected: inspect text-like filtering;
- suppressed: inspect top-k or consolidation policy;
- merged: inspect consolidation/NMS;
- imported missing despite accepted: inspect workbench import mapping.

## Non-Changes

- Orientation code was not changed.
- OCR behavior was not changed.
- Extraction defaults were not changed globally.
- Detector thresholds were not tuned.
- Consolidation/NMS behavior was not changed.
- Generated runtime outputs were not committed.

## Next Recommended Fix Path

Run recognition on the problematic page again and inspect `recognition_diagnostics`.

The next implementation should target the exact loss point:

```text
raw candidate absent -> candidate generation
rejected -> text-like rejection
suppressed/merged -> consolidation/top-k/NMS
accepted but not imported -> workbench import mapping
```
