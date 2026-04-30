from __future__ import annotations

from experiments import compare_macron_ocr_engines as helper


def test_parser_exposes_engine_skip_flags():
    help_text = helper.build_parser().format_help()

    assert "--skip-easyocr" in help_text
    assert "--skip-paddle" in help_text
    assert "--output-dir" in help_text


def test_term_classification_distinguishes_preserved_stripped_and_missing():
    assert helper._classify_term("Daitō-ryū", "Daitō-ryū") == "preserved"
    assert helper._classify_term("Daitō-ryū", "Daito-ryu") == "stripped"
    assert helper._classify_term("Daitō-ryū", "Dait6-rya") == "missing"
    assert helper._classify_term("kenjutsu", "kenjutsu") == "preserved"


def test_summary_tracks_macron_preservation_separately_from_plain_terms():
    results = [
        helper._result_payload(
            engine="fake",
            config="cfg",
            fixture="fixture",
            expected_terms=["Daitō-ryū", "kenjutsu"],
            output="Daito-ryu kenjutsu",
            elapsed=0.0,
        )
    ]

    summary = helper._summarize_results(results)[0]

    assert summary["preserved"] == 1
    assert summary["stripped"] == 1
    assert summary["macron_preserved"] == 0
    assert summary["macron_stripped"] == 1
