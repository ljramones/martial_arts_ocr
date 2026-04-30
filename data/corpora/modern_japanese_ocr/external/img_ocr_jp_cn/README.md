# IMG OCR JP/CN

## Status

Local external dataset. Provenance is not fully confirmed from the local files.
Do not redistribute or commit payload files until source and license are
confirmed.

## Source / Provenance

Inferred from folder/file names only:

- Original local folder: `data/IMG_OCR_JP_CN`
- Current local payload folder:
  `data/corpora/modern_japanese_ocr/external/img_ocr_jp_cn/original/`
- Looks like a Japanese/Chinese OCR document-image dataset with paired image
  files and LabelMe-style JSON annotations.

Provenance unknown — do not treat as canonical yet.

## Contents

Observed categories include:

- Badges and passes
- Bills/receipts
- Book contents or covers
- Contracts
- Forms
- Identity cards
- Newspapers
- Notes
- Papers/thesis pages
- Trade documents
- Whiteboard/blackboard images

The local inspection found image files paired with JSON files. Sample JSON uses
`shapes[].label` text and rectangle `points`, which can serve as OCR ground
truth and region geometry if licensing permits.

## Intended Use

Potentially useful for modern Japanese OCR evaluation because it appears to
include real document images plus text-region annotations.

## Not Intended For

Do not use for training or redistribute in repository commits until source,
license, and ground-truth semantics are confirmed.

## Local Files

Payload files are ignored:

```text
original/
```

## Manifest

Use `manifests/manifest.example.json` as the shape for a local manifest.

## Notes / Open Questions

- Confirm dataset source and license.
- Confirm whether JSON labels are transcription ground truth or layout labels.
- Decide whether subsets can be used for quantitative OCR accuracy checks.
