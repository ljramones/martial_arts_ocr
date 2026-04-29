# Corpus 2

This ad hoc corpus contains a second local review set added after the Donn Draeger DFD extraction reviews.

## Layout

- `original/`: local source JPEG pages. Do not modify these files directly.
- `augmented/`: derived pages produced from originals by rotation, skew, noise, cropping, tiling, or other transforms.
- `manifests/`: manifest examples and local review manifests for Notebook 05.

The original images are local/private and are ignored by git. Commit only documentation, manifest examples, and intentionally curated metadata.

## Manifest Workflow

Generate a local manifest:

```bash
.venv/bin/python scripts/generate_real_page_manifest.py \
  --input data/corpora/ad_hoc/corpus2/original \
  --output data/corpora/ad_hoc/corpus2/manifests/manifest.local.json \
  --collection-name corpus2 \
  --source-kind original
```

Then open `notebooks/05_real_page_extraction_review.ipynb` and point it at the local manifest to review image-region extraction, crop quality, and text cleanup.

Review `original/` before creating or evaluating any augmented variants.
