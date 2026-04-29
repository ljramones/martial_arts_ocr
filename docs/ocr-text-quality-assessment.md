# OCR/Text Quality Assessment

## Status

Extraction architecture is frozen for this phase. The next project risk is OCR
text quality and the usefulness of canonical `DocumentResult` output.

Current OCR/text plumbing is functional and better structured than the legacy
prototype path: Tesseract/EasyOCR wrappers can emit real OCR boxes, adapters can
promote boxes into canonical `TextRegion` objects, and `WorkflowOrchestrator`
writes `data.json`, `page_data.json`, `page_1.html`, and `text.txt` from
`DocumentResult`.

The main gap is quality and hierarchy. The pipeline preserves many raw details,
but it does not yet provide a clean line/paragraph text-region hierarchy,
review-friendly `data.json` shape, or strong evidence that cleanup is safe on
real DFD/Corpus 2 OCR text.

## Scope

Reviewed the current OCR engine wrappers, post-processing, text utilities,
Japanese processing, pipeline adapters, canonical document models,
orchestration, reconstruction, and tests. This is an assessment only; no
runtime behavior was changed.

## Current OCR Engine Output

Supported engines:

- Tesseract via `TesseractEngine`
- EasyOCR via `EasyOCREngine`

Tesseract:

- Uses configured languages from `Config.OCR_ENGINES["tesseract"]`.
- Runs `pytesseract.image_to_data()` and `image_to_string()`.
- Emits `OCRResult.text`, confidence, engine name, metadata, and word-level
  `bounding_boxes`.
- Bounding boxes contain `text`, normalized confidence, `x`, `y`, `width`,
  `height`, `source="ocr_engine"`, `engine="tesseract"`, and
  `ocr_level="word"`.
- `OCRProcessor._run_full_page_ocr()` tries PSM `11`, `3`, and `6`, but it
  overwrites `result.metadata` with only `{"full_page": True, "psm": psm}`.
  That loses the original Tesseract metadata such as language/config unless the
  legacy `OCRResult` itself remains available elsewhere.

EasyOCR:

- Disabled by default in config.
- Emits accepted detections joined with newlines.
- Converts polygon boxes to rectangular `x/y/width/height`.
- Tags boxes with `source="ocr_engine"`, `engine="easyocr"`, and
  `ocr_level="line"` when paragraph mode is used.

Selection:

- `_select_best_ocr_result()` uses confidence, text length, and word count.
- There is no evidence yet that the selected engine/PSM is best on DFD or
  Corpus 2 real pages.

## OCR Box Flow into Canonical Model

Box adapters are in `src/martial_arts_ocr/pipeline/adapters.py`.

Supported shapes:

- `OCRResult.bounding_boxes`
- `ocr_text_boxes`
- `words`
- `lines`
- `blocks`
- Tesseract TSV-style dicts
- EasyOCR polygon tuples
- existing `TextRegion`-like objects

Canonical flow:

```text
OCR engine output
  -> OCRResult.bounding_boxes / generic box shape
  -> ocr_text_regions_from_ocr_output()
  -> TextRegion(...)
  -> PageResult.text_regions
  -> PageResult.metadata["ocr_text_boxes"]
  -> DocumentResult.to_dict()
```

Coordinate convention:

- Canonical `BoundingBox` is `x, y, width, height`.
- `utils/text/geometry.py` explicitly documents `(x, y, width, height)`.
- Adapter metadata sets `bbox_convention="xywh"` on normalized OCR boxes.

Provenance:

- `TextRegion.metadata` preserves `source`, `engine`, and `ocr_level`.
- Tesseract/EasyOCR wrappers tag true OCR engine boxes.
- `ExtractionService` prefers true OCR-engine boxes over layout-derived text
  boxes for OCR-aware image filtering.

Risk:

- `ocr_text_boxes_from_ocr_output()` collects boxes from `best_ocr_result` and
  every item in `ocr_results`. In the current `ProcessingResult`, the selected
  `best_ocr_result` is usually also present in `ocr_results`, so duplicate
  boxes may be possible.
- If OCR word boxes are promoted directly into `PageResult.text_regions`, the
  canonical page may contain many word regions rather than line/paragraph
  regions. That is useful for geometry, but poor as the primary readable text
  structure.

## Text Cleanup Behavior

There are two cleanup layers:

1. `OCRPostProcessor.clean_text()`
2. `utils.TextCleaner.clean_text()`

`OCRPostProcessor`:

- Applies Unicode NFKC normalization.
- Applies typewriter corrections when indicators match.
- Processes lines for hyphenation and soft-wrap merging.
- Applies broad OCR correction regexes.
- Applies domain corrections if present.
- May run SymSpell for low-confidence text.
- Normalizes whitespace at the end.

`TextCleaner`:

- Removes control characters except newlines/tabs.
- Removes leading/trailing per-line whitespace.
- Removes some OCR artifacts and standalone punctuation lines.
- Removes isolated single ASCII-character lines.
- Applies common OCR substitutions such as `rn -> m` and `vv -> w`.
- Normalizes repeated spaces and excessive blank lines.

Line breaks:

- `TextCleaner` preserves useful line breaks and reduces `\n\n\n` to `\n\n`.
- `OCRPostProcessor` has one risky regex in general corrections:
  `r'\s+' -> ' '`, which can collapse newlines before the later
  `\n{3,}` rule can preserve paragraph boundaries.
- `_process_lines()` can merge soft wraps, but later whitespace cleanup may
  reduce layout signal.

Risk:

- Cleanup is English/typewriter-heavy and may be too aggressive for mixed
  English/Japanese scholarly pages.
- Some replacements are context-free (`rn -> m`, `vv -> w`) and can damage
  names or romanized terms.
- There is no real-page fixture proving the postprocessor preserves line,
  heading, caption, and paragraph structure from DFD/Corpus 2 OCR output.

## Japanese, Macron, and Punctuation Preservation

Existing test coverage in `tests/test_text_utils_extraction.py` verifies that
`TextCleaner` preserves:

```text
武道
柔術
koryū
budō
Daitō-ryū
ō
ū
—
・
「」
```

It also verifies useful line breaks are preserved by `TextCleaner`.

Important limitation:

- These tests cover `TextCleaner`, not the full `OCRPostProcessor -> TextCleaner
  -> DocumentResult -> data.json` path.
- `OCRPostProcessor` uses NFKC normalization. That is usually helpful, but it
  should be explicitly tested against macrons, Japanese punctuation, long vowel
  marks, and mixed English/Japanese martial arts text.
- There is no current test proving `Daitō-ryū`, `koryū`, `budō`, Japanese
  quotes, or interpuncts survive the complete `OCRProcessor.process_document()`
  cleanup chain.

## Japanese Processing Path

`JapaneseProcessor` currently:

- detects Japanese segments
- romanizes via pykakasi when available, with fallback character mapping
- attempts morphology through MeCab when available
- attempts translation through Argos when available
- extracts martial arts terms
- returns `JapaneseProcessingResult`

`OCRProcessor.process_document()` calls `JapaneseProcessor.process_text()` only
when final cleaned text contains kana/kanji.

Where Japanese output lands:

- In legacy `ProcessingResult.japanese_result`
- In `ProcessingResult.language_segments`
- In generated HTML/markdown through legacy helper methods
- In `DocumentResult.metadata["legacy"]` after adapter conversion
- In DB persistence through legacy fields such as `japanese_segments`,
  `overall_romaji`, `overall_translation`, and `japanese_metadata`

Where it does not yet land cleanly:

- Not as first-class `TextRegion.metadata`
- Not as first-class `PageResult.metadata`
- Not as explicit canonical `DocumentResult` fields

Romanization/translation are therefore document-path behavior through the
legacy processor result and metadata, not a clean canonical model feature yet.
Japanese model promotion should remain a separate future pass.

## DocumentResult Serialization and data.json Shape

`DocumentResult.to_dict()` emits:

- `document_id`
- `source_path`
- `pages`
- `language_hint`
- `detected_languages`
- `confidence`
- `metadata`

`PageResult.to_dict()` emits:

- page dimensions
- `text_regions`
- `image_regions`
- `raw_text`
- confidence
- metadata

`TextRegion.to_dict()` includes bbox, confidence, language, reading order, and
metadata. `ImageRegion.to_dict()` does the same for image regions.

`WorkflowOrchestrator._write_artifacts()` writes `data.json` from
`DocumentResult.to_dict()` and adds legacy aliases:

- `processing_date`
- `legacy_processing_result`
- `raw_text`
- `cleaned_text`
- `text`
- optional `html_content`
- optional `overall_confidence`

Strength:

- Current `data.json` preserves canonical pages/regions and enough legacy data
  for compatibility.
- OCR boxes are visible through `pages[].text_regions[]` and
  `pages[].metadata["ocr_text_boxes"]` when adapters promote them.

Risk:

- `metadata["legacy"]` can be large and noisy.
- OCR boxes may appear twice: as `TextRegion` objects and as serialized
  `ocr_text_boxes` metadata.
- Word-level boxes as primary `text_regions` make `data.json` geometrically
  rich but not reader-friendly.
- There is not yet a compact review-oriented text summary separating raw OCR,
  cleaned text, lines, paragraphs, boxes, engine metadata, and Japanese
  analysis.

## Page Reconstruction Gaps

`PageReconstructor` supports both legacy `ProcessingResult` and canonical
`DocumentResult` / `PageResult`.

Canonical reconstruction:

- Converts each `TextRegion` into a positioned text element.
- Converts each `ImageRegion` into a positioned image element.
- Falls back to one text block from `page.combined_text()` if no elements exist.
- Sorts elements by `(y, x)`.

Gaps:

- If `TextRegion` objects are word-level OCR boxes, `page_1.html` becomes many
  small word elements rather than readable line/paragraph reconstruction.
- There is no line/paragraph grouping from OCR word boxes yet.
- OCR boxes do not currently influence paragraph reconstruction beyond direct
  region placement.
- Image regions are included, but extraction is review-mode only and disabled by
  default.
- The HTML is useful as a debug/review artifact, not yet as a faithful page
  reconstruction.

## Test Coverage

Covered:

- OCR processor contract adapts legacy output to `DocumentResult`.
- OCR output adapter handles dicts, objects, paged results, current processing
  shape, and bounding boxes.
- OCR box adapter tests cover Tesseract-like rows, EasyOCR polygons, `words`,
  `lines`, existing metadata, and provenance.
- Document model tests cover serialization and combined text behavior.
- Text utility tests cover Japanese/macron/punctuation preservation in
  `TextCleaner`.
- Orchestrator tests cover artifact writing, canonical processor preference,
  DB persistence, extraction injection, and data.json image-region output.

Gaps:

- No real OCR fixture captures DFD or Corpus 2 OCR text before/after cleanup.
- No test proves full `OCRPostProcessor + TextCleaner` preserves Japanese,
  macrons, punctuation, and line breaks.
- No test verifies duplicate OCR box handling when `best_ocr_result` also
  appears in `ocr_results`.
- Japanese processor tests only verify module imports, not behavior.
- No test validates `data.json` readability or expected compact shape for OCR
  review.
- No test checks page reconstruction quality from word/line OCR boxes.

## Risks / Unknowns

- Tesseract PSM selection may not be empirically best for DFD/Corpus 2.
- Cleanup may damage rare names, romanized Japanese, macrons, or line structure.
- Word-level `TextRegion` promotion is good for geometry but poor for readable
  canonical text hierarchy.
- Japanese output remains legacy metadata, making downstream consumers rely on
  old shapes.
- `data.json` may be too noisy for review workflows because it mixes canonical
  data, legacy data, OCR boxes, and extraction metadata.
- `page_1.html` is not yet a strong page reconstruction artifact for text-heavy
  pages.

## Recommended Next Implementation Pass

Top recommendation:

1. Add OCR text-quality fixtures and a normalization/serialization review pass.

Scope:

- Use synthetic and small local fixture outputs, not private corpus images.
- Exercise `OCRPostProcessor -> TextCleaner -> adapters -> DocumentResult`.
- Verify preservation of Japanese, macrons, punctuation, useful line breaks,
  OCR provenance, and box geometry.
- Identify whether duplicate OCR boxes occur and fix only if proven.

Second recommendation:

2. Define a clearer canonical text hierarchy.

Target:

- Preserve OCR word boxes for geometry.
- Add or derive line-level `TextRegion` objects for readable page structure.
- Keep raw/cleaned/full text easy to inspect in `data.json`.
- Avoid making word boxes the only canonical text-region view.

Implementation should not promote Japanese analysis yet. First make text and box
serialization reliable and reviewable.

## Implemented Follow-Up: OCR Text Hierarchy Fixtures and Normalization

The first follow-up added synthetic OCR text-quality fixtures and a small
canonical text hierarchy helper.

Implemented behavior:

- OCR word boxes remain available as `TextRegion` objects with
  `metadata["source"]="ocr_engine"` and `metadata["ocr_level"]="word"`.
- Derived line regions are added with
  `metadata["source"]="ocr_normalization"` and
  `metadata["ocr_level"]="line"`.
- `PageResult.metadata["readable_text"]` provides compact line-oriented text.
- `PageResult.metadata["ocr_word_count"]` and
  `PageResult.metadata["ocr_line_count"]` summarize the hierarchy.
- Fixtures cover English word boxes, mixed English/Japanese martial arts text,
  macrons, Japanese punctuation, Tesseract-like rows, and EasyOCR-like polygons.

This does not solve multi-column layout, vertical Japanese, or final page
reconstruction. It gives the next OCR/text passes stable fixtures and a clear
word-vs-line distinction.

## Files Reviewed

- `src/martial_arts_ocr/ocr/processor.py`
- `src/martial_arts_ocr/ocr/engines.py`
- `src/martial_arts_ocr/ocr/models.py`
- `src/martial_arts_ocr/ocr/postprocessor.py`
- `src/martial_arts_ocr/pipeline/adapters.py`
- `src/martial_arts_ocr/pipeline/document_models.py`
- `src/martial_arts_ocr/pipeline/orchestrator.py`
- `src/martial_arts_ocr/pipeline/extraction_service.py`
- `src/martial_arts_ocr/japanese/processor.py`
- `src/martial_arts_ocr/reconstruction/page_reconstructor.py`
- `utils/text/text_utils.py`
- `utils/text/geometry.py`
- `tests/test_ocr_processor_contract.py`
- `tests/test_ocr_output_adapter.py`
- `tests/test_ocr_box_adapters.py`
- `tests/test_document_models.py`
- `tests/test_pipeline_orchestrator.py`
- `tests/test_text_utils_extraction.py`
- `docs/review-mode-extraction-guide.md`
- `docs/extraction-architecture-freeze-2026-04-28.md`
