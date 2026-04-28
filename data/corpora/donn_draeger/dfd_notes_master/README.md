# Donn Draeger DFD Notes Master Corpus

This corpus contains Donn Draeger lecture page images used for extraction-quality review.

## Layout

- `original/`: original source pages. Do not modify these files directly.
- `augmented/`: derived pages produced from originals by rotation, skew, noise, cropping, tiling, or other transforms.
- `manifests/`: manifest examples and local review manifests for Notebook 05.

`manifest.local.json` is local/generated and ignored. `manifest.example.json` documents the expected schema.

## Review Workflow

Generate a local manifest:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input data/corpora/donn_draeger/dfd_notes_master/original \
  --output data/corpora/donn_draeger/dfd_notes_master/manifests/manifest.local.json
```

Open `notebooks/05_real_page_extraction_review.ipynb` and inspect detected image regions, saved crops, text cleanup, and reading order. Record threshold or failure notes before integrating extraction into runtime.

Review original pages first. Use `augmented/` and training datasets only after the original pages are understood.
