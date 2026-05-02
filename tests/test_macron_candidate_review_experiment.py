from __future__ import annotations

import json

from experiments import review_macron_candidates as helper


def test_parser_exposes_review_inputs():
    help_text = helper.build_parser().format_help()

    assert "--summary-json" in help_text
    assert "--output-dir" in help_text
    assert "--no-fixtures" in help_text


def test_load_text_sources_from_summary_json_extracts_relevant_text(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "readable_text": "Daito-ryu appears here.",
                        "metadata": {"ignored_notes": "Daito-ryu should not be extracted from arbitrary notes."},
                    }
                ],
                "best_result": {"raw_output": "koryG from OCR"},
            }
        ),
        encoding="utf-8",
    )

    sources = helper.load_text_sources_from_json_file(summary_path)

    assert [source.field_path for source in sources] == [
        "$.results[0].readable_text",
        "$.best_result.raw_output",
    ]
    assert [source.text for source in sources] == ["Daito-ryu appears here.", "koryG from OCR"]


def test_review_text_sources_emits_candidates_without_mutating_text():
    text = "Daito-ryu and koryu are ASCII OCR outputs."
    source = helper.TextSource(source_id="sample", source_type="fixture", text=text)

    reviewed = helper.review_text_sources([source])

    assert source.text == text
    assert reviewed[0]["candidate_count"] == 2
    assert [(candidate["observed"], candidate["candidate"]) for candidate in reviewed[0]["candidates"]] == [
        ("Daito-ryu", "Daitō-ryū"),
        ("koryu", "koryū"),
    ]
    assert all(candidate["requires_review"] is True for candidate in reviewed[0]["candidates"])


def test_canonical_fixture_text_does_not_emit_replacement_candidate():
    source = helper.TextSource(source_id="control", source_type="fixture", text="koryū budō Daitō-ryū")

    reviewed = helper.review_text_sources([source])

    assert reviewed[0]["candidate_count"] == 0
    assert reviewed[0]["candidates"] == []


def test_candidate_summary_groups_by_candidate_match_type_and_source_type():
    candidates = [
        {
            "candidate": "Daitō-ryū",
            "match_type": "variant_exact",
            "source_type": "fixture",
        },
        {
            "candidate": "Daitō-ryū",
            "match_type": "variant_hyphen_space",
            "source_type": "summary_json",
        },
        {
            "candidate": "koryū",
            "match_type": "observed_ocr_confusion",
            "source_type": "summary_json",
        },
    ]

    summary = helper.summarize_candidates(candidates)

    assert summary == {
        "by_candidate": {"Daitō-ryū": 2, "koryū": 1},
        "by_match_type": {
            "observed_ocr_confusion": 1,
            "variant_exact": 1,
            "variant_hyphen_space": 1,
        },
        "by_source_type": {"fixture": 1, "summary_json": 2},
    }
