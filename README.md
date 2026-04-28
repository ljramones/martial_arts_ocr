# Martial Arts OCR

Experimental OCR and digitization tools for martial arts research materials, especially Donn Draeger lecture materials and mixed English/Japanese scans. The project is being stabilized around a package layout while preserving earlier Flask, OCR, Japanese-processing, layout, Qt, and model experiments.

## Current Working Scope

- Upload scanned image files through a local Flask UI.
- Run OCR through Tesseract and/or EasyOCR where those engines are installed.
- Detect and preserve basic image regions such as diagrams and illustrations.
- Process modern Japanese text using local tools where configured, including MeCab, pykakasi, and Argos Translate.
- Store processing metadata in SQLite.
- Export HTML/JSON-style artifacts for local viewing and inspection.
- Keep operation local/offline once dependencies and language data are installed.

## Experimental / Not Yet Productized

- Qt desktop UI in `experiments/qt_app/`.
- YOLO image layout model work in `experiments/image_layout_model/`.
- Orientation model work in `experiments/orientation_model/`.
- Classical Japanese, koryu densho, and makimono workflows.
- Seal detection, KuroNet integration, historical normalization, and grammar conversion.
- Batch workflow orchestration and scholarly three-panel output.

These areas are roadmap or research code unless tests and runtime docs explicitly say otherwise.

## Immediate Stabilization Goals

- Restore a clean package structure under `src/martial_arts_ocr/`.
- Add import and smoke tests that do not require OCR engines or model downloads.
- Move diagnostics and manual tools out of the repository root.
- Create one canonical pipeline API.
- Make documentation match the current implementation.

## Repository Layout

```text
src/martial_arts_ocr/      Package facade and new stable APIs
processors/                Legacy OCR, Japanese, extraction, and reconstruction code
utils/                     Shared image and text utilities
templates/, static/        Flask UI templates and assets
scripts/                   Setup, diagnostics, and manual OCR tools
experiments/               Qt UI and model experiments
tests/                     Pytest smoke and import tests
data/                      Local uploads, processed output, diagnostics, and runs
```

The legacy root modules remain while migration is in progress. New code should prefer imports from `martial_arts_ocr.*` where a package API exists.

## Setup

Install system OCR dependencies first. On macOS:

```bash
brew install tesseract tesseract-lang mecab mecab-ipadic
```

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/setup_japanese.py
```

## Running

Run the legacy Flask app:

```bash
python app.py
```

Then open `http://localhost:5000`.

The package CLI currently delegates to the same legacy app:

```bash
python -m martial_arts_ocr.cli
```

Run the experimental Qt UI with:

```bash
python -m experiments.qt_app.main
```

## Tests

The baseline tests are intentionally lightweight:

```bash
python -m pytest -q
```

They verify package imports, config loading, Flask factory creation, and the `WorkflowOrchestrator` seam with a fake processor. They do not require Tesseract, EasyOCR, Japanese dictionaries, GPU support, or model downloads.

## Pipeline API

New integrations should target the canonical pipeline seam:

```python
from pathlib import Path
from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

request = PipelineRequest(image_path=Path("scan.png"), language_hint="en")
result = WorkflowOrchestrator().process_document(request)
```

For tests, inject a fake processor into `WorkflowOrchestrator` to avoid heavy OCR dependencies.

## Data and Generated Files

Local inputs and generated output belong under `data/` and should not be committed. Large model checkpoints, training runs, OCR output, local databases, and private scans should remain local unless a change explicitly requires a small, reviewed fixture.
