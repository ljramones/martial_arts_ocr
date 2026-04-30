from __future__ import annotations

from pathlib import Path

from experiments import review_macron_ocr as helper
from martial_arts_ocr.ocr.postprocessor import OCRPostProcessor
from utils.text.text_utils import TextCleaner


def test_macron_review_parser_exposes_language_and_psm_flags():
    help_text = helper.build_parser().format_help()

    assert "--language" in help_text
    assert "--psm" in help_text
    assert "--output-dir" in help_text


def test_cleanup_serialization_control_preserves_macrons(tmp_path):
    text = "koryū budō Daitō-ryū\njūjutsu dōjō ryūha sōke"

    control = helper._cleanup_serialization_control(
        "fixture",
        Path(tmp_path / "fixture.png"),
        text,
        postprocessor=OCRPostProcessor(domain="martial_arts"),
        cleaner=TextCleaner(),
    )

    for term in ["koryū", "budō", "Daitō-ryū", "jūjutsu", "dōjō", "ryūha", "sōke"]:
        assert term in control["readable_text"]
        assert term in control["serialized_text"]


def test_preservation_result_detects_stripped_macrons():
    assert helper._preservation_result("koryū budō", ("koryū budō",)) == "preserved"
    assert helper._preservation_result("koryu budo", ("koryū budō",)) == "macrons_stripped"
    assert helper._preservation_result("noise", ("koryū budō",)) == "missing_or_misread"
