from __future__ import annotations

import importlib


def test_review_mode_runner_imports_without_paddle(monkeypatch):
    original_import = importlib.import_module
    imported: list[str] = []

    def tracking_import(name, package=None):
        imported.append(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    module = importlib.import_module("experiments.run_review_mode_extraction")

    assert module.build_parser() is not None
    assert "paddleocr" not in imported


def test_review_mode_runner_help_parser_has_expected_flags():
    from experiments.run_review_mode_extraction import build_parser

    help_text = build_parser().format_help()

    assert "--enable-image-extraction" in help_text
    assert "--enable-paddle-fusion" in help_text
    assert "--paddle-model-dir" in help_text
    assert "--no-crops" in help_text
