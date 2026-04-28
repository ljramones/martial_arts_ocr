# Repository Guidelines

## Project Structure & Module Organization

This is a Python OCR system for scanned martial arts research materials. New stable APIs live under `src/martial_arts_ocr/`, including the Flask factory facade in `app/` and the pipeline seam in `pipeline/`. Legacy runtime code still lives in `processors/`, `utils/`, `app.py`, `database.py`, and `models.py` while migration continues. Web assets are in `templates/` and `static/`. Diagnostics and setup tools belong in `scripts/`; manual harnesses belong in `scripts/manual/`. Qt and model work are experiments under `experiments/qt_app/`, `experiments/image_layout_model/`, and `experiments/orientation_model/`. Local uploads, processed output, diagnostics, and training runs belong under `data/` and should not be committed.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/setup_japanese.py
```

Run the legacy Flask app with `python app.py`, then open `http://localhost:5000`. Use `python -m experiments.qt_app.main` for the experimental Qt UI. Run the smoke baseline with `python -m pytest -q`. Compile-check core files with `python -m py_compile app.py config.py database.py models.py processors/*.py`.

## Coding Style & Naming Conventions

Use 4-space indentation, snake_case for modules/functions/variables, and PascalCase for classes. Prefer package imports from `martial_arts_ocr.*` for new code, but preserve compatibility with legacy imports during migration. Keep pipeline stages small and avoid hard-coded local paths; route configurable paths through `config.py`.

## Testing Guidelines

Pytest tests live in `tests/` and should be named `test_*.py`. Baseline tests must not require OCR engines, Japanese dictionaries, GPUs, or model downloads; inject fakes into `WorkflowOrchestrator` for smoke coverage. Manual OCR/image experiments should stay in `scripts/manual/`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `fixed issues in processors`. Keep commits focused by subsystem. PRs should describe behavior changes, list verification commands, identify moved files or generated artifacts, and include screenshots for Flask or Qt UI changes.

## Security & Configuration Tips

Do not commit secrets, personal scans, local databases, checkpoints, or generated OCR output. Keep `master_key.txt`, `*.db`, `data/`, `tessdata/`, and model artifacts local unless a small fixture is explicitly reviewed.
