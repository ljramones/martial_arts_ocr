# Project-Native Modern Japanese OCR Samples

## Status

Placeholder for selected pages from this repository's own DFD and Corpus 2
material that visibly contain modern Japanese or romanized Japanese terms.

## Source / Provenance

Project-native only:

- `data/corpora/donn_draeger/dfd_notes_master/`
- `data/corpora/ad_hoc/corpus2/`

Do not copy unrelated external datasets here.

## Contents

- `original/`: optional local copies or symlinks for selected project-native
  pages.
- `manifests/`: tracked examples and ignored local manifests.

## Intended Use

Use for small, curated OCR comparisons against project material, especially
when evaluating whether OCR configs recover Japanese characters or romanized
martial arts terms.

## Not Intended For

This is not a third-party dataset bucket and not a training-data area.

## Local Files

Dataset payloads under `original/` are ignored by default. Prefer manifests
that point back to the existing DFD / Corpus 2 source paths.

## Manifest

Use `manifests/manifest.example.json` as the shape for local manifests.

## Notes / Open Questions

The focused Japanese OCR review found candidate project-native pages, but a
final project-native modern Japanese OCR manifest has not been created yet.
