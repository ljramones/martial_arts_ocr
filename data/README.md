# Project Data Layout

This directory is the single home for corpora, runtime files, training data, evaluation metadata, and notebook outputs.

## Directory Roles

- `corpora/`: source document collections and corpus-specific manifests.
- `runtime/`: Flask uploads, processed artifacts, SQLite databases, and local runtime state.
- `training/`: ML training datasets and derived training material.
- `evaluation/`: review manifests, notes, rubrics, and quality-check metadata.
- `notebook_outputs/`: temporary crops, overlays, and diagnostics created by notebooks.

## Source Data Rules

Original corpus pages belong under an `original/` directory and should not be edited in place. Derived pages created by rotation, skew, noise, cropping, or tiling belong under `augmented/` or `training/`, depending on purpose.

Runtime output, notebook output, local manifests, private folders, and databases are local/generated and are ignored. Commit README files, manifest examples, curated reference metadata, and small intentional fixtures only.

## Donn Draeger Review Workflow

Generate a local real-page manifest from the original corpus:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input data/corpora/donn_draeger/dfd_notes_master/original \
  --output data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json
```

Then open `notebooks/05_real_page_extraction_review.ipynb` and review image regions, crop quality, text cleanup, and reading order. Notebook outputs go to `data/notebook_outputs/`.

## Adding New Corpora

Create a new folder under `data/corpora/<collection>/<corpus>/` with at least:

```text
README.md
original/
augmented/
manifests/manifest.example.json
```

Keep corpus-specific provenance in that corpus README.
