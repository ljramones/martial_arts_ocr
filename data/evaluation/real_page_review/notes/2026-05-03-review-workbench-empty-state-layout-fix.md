# Review Workbench Empty-State Layout Fix

## Purpose

Record the viewer layout issue found while manually testing the local review workbench.

## Observed Issue

After selecting a page, the viewer could still reserve a large dark empty-state area above the scanned page image. The message:

```text
Create or load a project, then select a page.
```

should only appear when no page is selected.

## Cause

The page stage used a fixed minimum height for both empty and loaded states. The placeholder was hidden by JavaScript, but the loaded viewer still retained empty-state layout space. That made the page image appear below a large dark area and made overlay alignment harder to reason about.

## Fix

The viewer stage now has an explicit `is-empty` state:

- empty state: centered placeholder with minimum height;
- loaded state: no placeholder and no fixed empty-state height;
- page image replaces the placeholder rather than being pushed below it.

The overlay remains positioned against the rendered image area.

## Manual Verification Steps

1. Start the local Flask app.
2. Open `/review`.
3. Confirm the empty placeholder appears before a page is selected.
4. Load a project and select a page.
5. Confirm the placeholder disappears.
6. Confirm the scanned page starts at the top of the viewer area rather than below a large dark block.
7. Run recognition and confirm overlays align with the rendered page image.

## Runtime Behavior

No OCR, recognition, extraction-default, or backend project-state behavior changed. This was a frontend layout fix.
