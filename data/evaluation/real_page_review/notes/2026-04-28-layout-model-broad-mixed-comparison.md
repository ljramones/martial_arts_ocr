# Layout Model Broad/Mixed Comparison

Date: 2026-04-28

## Purpose

Determine whether document-layout models separate text and visual regions better
than the current classical/OCR-aware detector on broad/mixed crop cases.

This was a focused availability and strategy comparison pass. No model was
installed, downloaded, or wired into runtime.

## Backends Available

| Backend | Available? | Notes |
|---|---|---|
| Classical baseline | yes | `ClassicalLayoutStrategy` runs the current OpenCV/scoring/consolidation detector. |
| PaddleOCR PP-Structure / PP-DocLayout | no | `paddleocr` is not installed. |
| DocLayout-YOLO | no | `ultralytics` is installed, but no DocLayout-YOLO model path is configured. |
| LayoutParser/Detectron2 | no | `layoutparser` and `detectron2` are not installed. |
| Generic YOLO | no for this comparison | `ultralytics` is installed and `experiments/image_layout_model/yolov8n.pt` exists, but that is not a configured/validated document-layout model. It was not used as a semantic layout backend. |

## Summary Table

| Corpus | Sample ID | Backend | Text/Figure Separation | Visual Retained? | Text FP? | Broad/Mixed Crop? | Notes |
|---|---|---|---|---|---|---|---|
| DFD | `original_img_3335` | Classical/OCR-aware | partial | yes | no | yes | Large visual retained, but broad crop remains. |
| DFD | `original_img_3344` | Classical/OCR-aware | partial | yes | no | yes | Labeled diagram retained as mixed via OCR rescue. |
| DFD | `original_img_3397` | Classical/OCR-aware | acceptable | yes | no | no | Sparse/symbol regions retained. |
| DFD | `original_img_3352` | Classical/OCR-aware | acceptable | yes | no | no | Known candidates retained. |
| DFD | `original_img_3330` | Classical/OCR-aware | acceptable | yes | no | no | Tall/narrow visual strips retained. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_16_55_48` | Classical/OCR-aware | partial | yes | yes | yes | OCR-aware pass removed one text-like child crop but retained broad/mixed parent crop. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_10_58` | Classical/OCR-aware | partial | yes | no | yes | Mixed crop retained with strong visual evidence. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_17_19_36` | Classical/OCR-aware | partial | yes | no | yes | Broad mixed crop retained. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_29_28` | Classical/OCR-aware | partial | yes | no | yes | Broad visual/mixed crop retained. |
| Corpus 2 | `corpus2_new_doc_2026_04_28_18_54_00` | Classical/OCR-aware | partial | partial | no | yes | Photo-grid recall remains partial; OCR geometry does not add missing visual candidates. |

Optional document-layout ML backends were skipped because they were unavailable
or not configured with local document-layout model weights. Therefore this pass
does not answer whether PP-Structure or DocLayout-YOLO would outperform the
classical detector; it only confirms they are not currently runnable in this
environment without an explicit install/model setup step.

## DFD Results

The classical/OCR-aware path remained stable on the five hard DFD pages:

- `3335`: large visual region retained; broad crop remains.
- `3344`: labeled diagram retained as mixed; no visual regression.
- `3397`: sparse arrows/symbols retained.
- `3352`: known visual candidates retained.
- `3330`: tall/narrow visual strips retained.

No optional layout backend was available to compare semantic region classes
against these pages.

## Corpus 2 Results

The broad/mixed cases remain the main limitation:

- `16_55_48`: one OCR-overlapping child crop was removed, but the broad parent
  remained as an uncertain/mixed crop.
- `17_10_58`, `17_19_36`, `18_29_28`: visual regions remained, but broad/mixed
  crops were not split into separate text and figure regions.
- `18_54_00`: photo-grid recall remains partial; OCR-aware suppression cannot
  recover visual candidates that the classical detector does not propose.

No optional layout backend was available to test whether semantic layout classes
would split these parent crops or recover missed photo-grid regions.

## Findings

- Did any layout backend split broad parent regions into separate text/figure
  regions? Not tested; no document-layout backend was available/configured.
- Did any backend recover missed photo-grid/visual content? Not tested; no
  document-layout backend was available/configured.
- Did any backend introduce new text false positives? Not tested.
- Did any backend clearly outperform the classical/OCR-aware baseline? No
  available backend could be compared.

## Recommendation

- [ ] Layout model comparison shows enough promise to prioritize ML strategy integration
- [x] Layout models were unavailable; install/evaluate one backend next
- [ ] Layout models did not help; mixed-region refinement is justified
- [ ] Results are inconclusive; improve evaluation/annotation first

Recommended next step: explicitly evaluate one document-layout backend in the
workbench, preferably PaddleOCR PP-Structure/PP-DocLayout or DocLayout-YOLO with
local model weights. If that is not practical, proceed to mixed-region
refinement, but keep it optional and require a corpus-level improvement gate.
