# Japanese Region Routing Helper Pass

## Purpose

Refactor the experiment-only Japanese region OCR helper so it reflects the routing design without changing runtime OCR behavior, canonical model fields, extraction, or serialization.

## Scope

Changed:

- `experiments/review_japanese_region_ocr.py`
- `tests/test_japanese_region_ocr_experiment.py`

Not changed:

- runtime OCR defaults
- canonical Japanese fields
- extraction behavior
- document serialization
- database schema
- required dependencies

## Helper Changes

The helper now defines explicit routing profile data structures:

```text
RegionProfile
OcrRoute
```

Available region profiles:

```text
horizontal_modern_japanese
vertical_modern_japanese
mixed_english_japanese
romanized_japanese_macrons
stylized_calligraphy
unknown_japanese_like
```

Available preprocessing profiles:

```text
none
grayscale
threshold
upscale_2x
upscale_3x
contrast_sharpen
```

Backward-compatible aliases remain available for older snippets/tests:

```text
original_crop -> none
contrast_or_sharpen -> contrast_sharpen
```

## Routing Profiles

| Region Profile | Route Summary | Review Behavior |
|---|---|---|
| `horizontal_modern_japanese` | `jpn`, PSM 6; diagnostic PSM 7 | Modern horizontal Japanese crop route. |
| `vertical_modern_japanese` | `jpn_vert`, PSM 5; diagnostic `jpn`, PSM 6 | Vertical routing path from the successful sidebar sample. |
| `mixed_english_japanese` | `eng+jpn`, PSM 6/11; diagnostic `jpn`, PSM 6 | Tests mixed-language routing while preserving the earlier evidence that `jpn` can recover term characters. |
| `romanized_japanese_macrons` | `eng`, PSM 6; diagnostic `eng+jpn`, PSM 6 | Keeps macron text in a Latin OCR path until real samples prove otherwise. |
| `stylized_calligraphy` | `jpn_vert`/`jpn` diagnostics only | Defaults to `needs_review`; Tesseract is not reliable here. |
| `unknown_japanese_like` | small diagnostic matrix | Fallback profile for untyped manifest entries. |

## Manifest Behavior

The helper can now choose a profile from any of these sample fields:

```text
japanese_region_type
region_type
profile
```

It also supports a CLI override:

```bash
--profile vertical_modern_japanese
```

Routing profiles are enabled by default. The old broad matrix behavior remains available:

```bash
--no-use-routing-profiles
```

## Output Changes

Each sample result now includes:

```text
routing_profile
selected_routes
needs_human_review
```

Each OCR attempt now includes:

```text
language
psm
preprocess_profile
route_role
quality_judgment
expected_terms_recovered
```

Quality judgment is intentionally simple:

```text
meaningful: all expected terms recovered
partial: some expected terms recovered
noisy: Japanese characters present but expected terms missing
fail: no useful Japanese signal
```

This is a review aid, not a model-quality metric.

## Tests Added

Added:

```text
tests/test_japanese_region_ocr_experiment.py
```

Coverage:

- parser exposes routing flags
- profile matrix encodes expected routes
- manifest profile selection and CLI override
- vertical routed processing records route/profile metadata
- legacy matrix mode still works with requested languages/PSMs
- quality judgment distinguishes meaningful/partial/noisy/fail

The tests use synthetic images and monkeypatched OCR calls. They do not require OCR binaries.

## Validation

Targeted helper tests:

```text
.venv/bin/python -m pytest tests/test_japanese_region_ocr_experiment.py -q
```

Result:

```text
6 passed
```

Full verification is run separately before commit.

Full verification:

```text
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile app.py config.py database.py models.py processors/*.py src/martial_arts_ocr/**/*.py scripts/*.py experiments/*.py
find utils -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile
```

Result:

```text
179 passed
project py_compile passed
utils py_compile passed
```

## Known Limits

- Region profile selection still depends on manual manifest labels.
- No automatic Japanese region detection was added.
- No automatic orientation classifier was added.
- No Japanese canonical model fields were added.
- `mixed_english_japanese` routing remains exploratory because prior evidence favored `jpn` for term recovery while the current pass also needs to test `eng+jpn` behavior.
- EasyOCR and PaddleOCR comparison are still future work.

## Recommended Next Step

Run the routed helper against the existing local Japanese region manifest after adding explicit `japanese_region_type` values to local samples.

Then compare:

```text
routed helper output
vs previous broad matrix output
```

Do not wire Japanese region OCR into runtime until a larger evaluated region set supports the routing matrix.
