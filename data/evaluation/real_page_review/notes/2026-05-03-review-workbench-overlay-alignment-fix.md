# Review Workbench Overlay Alignment Fix

## Purpose

Record the coordinate-mapping issue found while testing the local review workbench recognition overlays.

## Observed Issue

Detected boxes appeared offset from the scanned page image. The visible symptom was that boxes were drawn relative to the dark viewer area rather than the rendered page image.

## Cause

`project_state.json` stores bboxes in natural image coordinates:

```text
[x, y, width, height]
```

The frontend previously rendered SVG rectangles using those coordinates through an SVG sized to the viewer container. When the rendered image size differed from the viewer container, the overlay mapping could drift.

## Fix

The frontend now computes the rendered image layout from the actual image element:

```text
rendered width
rendered height
image offset within the viewer
scaleX
scaleY
```

Overlay rectangles are drawn in rendered-image coordinates, with the SVG positioned over the actual image. Drag/resize edits convert rendered coordinates back to natural image coordinates before updating `reviewed_bbox`.

## BBox Storage Invariant

Bboxes remain stored in natural image coordinates. No backend bbox format changed.

## Orientation Assumption

The workbench assumes the browser-displayed image orientation matches the backend image coordinate orientation. If overlays are rotated or mirrored rather than offset/scaled, EXIF orientation should be reviewed before saving corrected boxes.

## Manual Verification Steps

1. Start the local Flask app.
2. Open `/review`.
3. Load a local image folder.
4. Select a page with detected regions.
5. Run recognition.
6. Confirm boxes align with the displayed page image, not the dark viewer container.
7. Drag or resize a box.
8. Confirm the side-panel bbox remains in natural image coordinates.

## Runtime Behavior

No OCR, extraction-default, or runtime behavior was changed. This was a frontend coordinate-mapping fix for the local review workbench.
