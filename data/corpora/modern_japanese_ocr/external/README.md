# External Modern Japanese OCR Resources

Third-party modern Japanese OCR/document datasets belong here.

Each dataset should live under its own slug:

```text
external/<dataset_slug>/
  README.md
  original/
  manifests/
    manifest.example.json
    manifest.local.json
```

## Commit Policy

Dataset payloads under `original/` are ignored by default. Track only README
files, manifest examples, and other small provenance notes unless a dataset is
explicitly approved for version control.

## Provenance

If source, license, or redistribution terms are unclear, mark the dataset as
provenance unknown and do not treat it as canonical.
