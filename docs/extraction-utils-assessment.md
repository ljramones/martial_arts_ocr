# Extraction Utilities Assessment

## Modules Found

`utils/image` contains IO helpers, preprocessing, layout detection, region geometry/filtering, crop extraction, and CLI/API wrappers. The most relevant runtime pieces are `ImageProcessor`, `LayoutAnalyzer`, `ImageRegion`, and `extract_region`/`save_region_crops`.

`utils/text` currently centers on `text_utils.py`, which provides `TextCleaner`, `LanguageDetector`, `TextFormatter`, `TextStatistics`, and Japanese terminology helpers.

## Current Runtime Use

`OCRProcessor` uses `ImageProcessor`, `LayoutAnalyzer`, `ImageRegion`, `TextCleaner`, `LanguageDetector`, and `TextStatistics`. `ContentExtractor` also uses `ImageProcessor`, `LayoutAnalyzer`, `ImageRegion`, image crop helpers, and save helpers. `PageReconstructor` uses `TextFormatter` and `TextStatistics`.

## Overlap

`src/martial_arts_ocr/imaging` mostly wraps or builds on `utils/image`. OCR postprocessing overlaps with `TextCleaner`, but `TextCleaner` is still used directly by OCR and Japanese processing. Reconstruction overlaps with `TextFormatter`.

## Obsolete Or Risky Areas

`utils/image/layout/_legacy_image_layout.py` and `utils/image/regions/image_regions.py` are compatibility/legacy surfaces. Several docs reference older `ImageRegion(x, y, width, height)` behavior while the refactored dataclass had moved to `bbox`; this was a live compatibility bug for layout detectors and JSON serialization.

The text cleaner had two risky defaults: whitespace normalization used `\s`, which could collapse useful line breaks, and a Japanese correction rewrote `ー` as `一`, which corrupts katakana and mixed Japanese text.

## Minimal Fixes Applied

`ImageRegion` now accepts both `bbox` and legacy `x/y/width/height`, exposes `x`, `y`, `confidence`, and `to_dict()`, and remains bbox-backed. `save_region_crops()` was added for stable crop metadata. `TextCleaner` now preserves line breaks and no longer rewrites Japanese long-vowel marks. Extraction adapters now map utility outputs into canonical `TextRegion`, `ImageRegion`, `PageResult`, and `DocumentResult`.
