# Real-Page Evaluation Samples

This directory defines the local sample manifest format for extraction review.

Do not commit private scanned research pages by default. Place local scans under:

```text
samples/private/
```

For local review:

1. Copy `samples/manifest.example.json` to `samples/manifest.local.json`.
2. Put private page images under `samples/private/`.
3. Edit `manifest.local.json` with paths, notes, and expected behavior.
4. Open `notebooks/05_real_page_extraction_review.ipynb`.

`samples/private/` and `samples/manifest.local.json` are gitignored so local review data stays out of commits.

The manifest is intentionally descriptive, not a strict benchmark. Use it to record representative pages, expected extraction behavior, OCR-like sample text, and failure notes before runtime integration.
