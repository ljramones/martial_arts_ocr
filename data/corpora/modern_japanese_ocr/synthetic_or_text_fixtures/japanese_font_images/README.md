# Japanese Font Images

## Status

Local synthetic/font-image dataset. Provenance is inferred from local metadata
and not fully confirmed.

## Source / Provenance

Moved from:

```text
data/P-font-image-dataset/
```

The `labels.csv` file references Kaggle-style paths such as
`/kaggle/input/jp-fonts/...`, which suggests a generated Japanese font-image
dataset. Confirm source and license before redistribution.

## Contents

Observed:

- Around 150,000 `.jpg` images.
- `labels.csv` with generated text, language, blur, orientation, background,
  skew, color, spacing, font, and related generation metadata.

## Intended Use

Useful for synthetic OCR text fixture generation, cleanup/serialization stress
tests, and possibly character/word OCR experiments.

## Not Intended For

This is not real scanned-page OCR data. Do not use it as proof that the
pipeline handles real document layout, mixed scans, captions, or page
reconstruction.

## Local Files

Payload files under `original/` are ignored by default.

## Manifest

Use `manifests/manifest.example.json` as the shape for a local manifest.

## Notes / Open Questions

- Confirm source and license.
- Decide whether a tiny generated subset should become a tracked test fixture.
- Keep the full dataset local unless explicitly approved.
