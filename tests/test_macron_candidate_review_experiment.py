from __future__ import annotations

import json

from experiments import review_macron_candidates as helper


def test_parser_exposes_review_inputs():
    help_text = helper.build_parser().format_help()

    assert "--summary-json" in help_text
    assert "--output-dir" in help_text
    assert "--no-fixtures" in help_text
    assert "--decisions-file" in help_text
    assert "--reviewed-export" in help_text
    assert "--filter" in help_text
    assert "--source-filter" in help_text
    assert "--sort" in help_text
    assert "--limit" in help_text


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
    assert all(candidate["candidate_id"].startswith("sha256:") for candidate in reviewed[0]["candidates"])
    assert all(candidate["decision"] is None for candidate in reviewed[0]["candidates"])
    assert all(candidate["reviewed_value"] is None for candidate in reviewed[0]["candidates"])
    assert all(candidate["reviewer_notes"] == [] for candidate in reviewed[0]["candidates"])


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


def test_candidate_id_is_deterministic_and_source_sensitive():
    candidate = {
        "span": [10, 19],
        "observed": "Daito-ryu",
        "candidate": "Daitō-ryū",
        "match_type": "variant_exact",
    }

    first = helper.candidate_id_for(
        source_id="source-a",
        source_path="summary.json",
        field_path="$.readable_text",
        candidate=candidate,
    )
    second = helper.candidate_id_for(
        source_id="source-a",
        source_path="summary.json",
        field_path="$.readable_text",
        candidate=candidate,
    )
    different = helper.candidate_id_for(
        source_id="source-b",
        source_path="summary.json",
        field_path="$.readable_text",
        candidate=candidate,
    )

    assert first == second
    assert first.startswith("sha256:")
    assert first != different


def test_decisions_template_contains_review_placeholders():
    source = helper.TextSource(source_id="sample", source_type="fixture", text="Daito-ryu")
    reviewed_sources = [source for source in helper.review_text_sources([source]) if source["candidate_count"]]

    template = helper.build_decisions_template(reviewed_sources)

    assert template["schema_version"] == "macron_candidate_decisions.v1"
    assert len(template["decisions"]) == 1
    decision = template["decisions"][0]
    assert decision["candidate_id"].startswith("sha256:")
    assert decision["observed"] == "Daito-ryu"
    assert decision["candidate"] == "Daitō-ryū"
    assert decision["decision"] is None
    assert decision["reviewed_value"] is None
    assert decision["notes"] == []


def test_write_decisions_template_preserves_existing_file_by_default(tmp_path):
    path = tmp_path / "decisions.local.json"
    path.write_text('{"existing": true}', encoding="utf-8")

    wrote = helper.write_decisions_template(path, {"new": True})

    assert wrote is False
    assert path.read_text(encoding="utf-8") == '{"existing": true}'


def test_reviewed_export_separates_pending_reviewed_and_stale_decisions():
    source = helper.TextSource(source_id="sample", source_type="fixture", text="Daito-ryu koryu budo")
    reviewed_sources = [source for source in helper.review_text_sources([source]) if source["candidate_count"]]
    decisions_template = helper.build_decisions_template(reviewed_sources)
    decisions = decisions_template["decisions"]
    decisions[0]["decision"] = "accept"
    decisions[0]["reviewed_value"] = decisions[0]["candidate"]
    decisions[1]["decision"] = "reject"
    decisions.append({"candidate_id": "sha256:stale", "decision": "accept"})

    export = helper.build_reviewed_export(reviewed_sources, decisions_template)

    assert export["source_text_mutated"] is False
    assert export["counts"]["accept"] == 1
    assert export["counts"]["reject"] == 1
    assert export["counts"]["pending"] == 1
    assert export["counts"]["stale"] == 1
    assert len(export["reviewed_decisions"]) == 2
    assert export["reviewed_decisions"][0]["review_decision"]["reviewed_value"] == "Daitō-ryū"
    assert export["stale_decisions"] == [{"candidate_id": "sha256:stale", "decision": "accept"}]


def test_source_kind_distinguishes_fixture_synthetic_and_real_ocr():
    assert helper.source_kind_for({"source_type": "fixture", "source_path": None}) == "fixture"
    assert (
        helper.source_kind_for(
            {
                "source_type": "summary_json",
                "source_path": "data/notebook_outputs/macron_ocr_eval/summary.json",
            }
        )
        == "synthetic"
    )
    assert (
        helper.source_kind_for(
            {
                "source_type": "summary_json",
                "source_path": "data/notebook_outputs/ocr_text_quality_review/summary.json",
            }
        )
        == "real_ocr"
    )


def test_review_queue_filters_sorts_and_limits_candidates():
    sources = [
        source
        for source in helper.review_text_sources(
            [
                helper.TextSource(source_id="fixture", source_type="fixture", text="Daito-ryu budo"),
                helper.TextSource(
                    source_id="real",
                    source_type="summary_json",
                    source_path="data/notebook_outputs/ocr_text_quality_review/summary.json",
                    field_path="$.text",
                    text="BUDO",
                ),
            ]
        )
        if source["candidate_count"]
    ]
    decisions = helper.build_decisions_template(sources)
    decisions["decisions"][0]["decision"] = "accept"
    decisions["decisions"][0]["reviewed_value"] = decisions["decisions"][0]["candidate"]
    decisions["decisions"][1]["decision"] = "defer"

    queue = helper.build_review_queue(
        sources,
        decisions,
        decision_filter="reviewed",
        source_filter="fixture",
        sort_key="candidate",
        limit=1,
    )

    assert len(queue) == 1
    assert queue[0]["source_kind"] == "fixture"
    assert queue[0]["decision"] == "defer"
    assert queue[0]["candidate"] == "budō"


def test_review_queue_markdown_and_csv_include_context(tmp_path):
    rows = [
        {
            "candidate_id": "sha256:1",
            "decision": "pending",
            "source_kind": "real_ocr",
            "source_id": "source",
            "source_path": "summary.json",
            "field_path": "$.text",
            "observed": "BUDO",
            "candidate": "budō",
            "reviewed_value_suggestion": "BUDŌ",
            "case_pattern": "uppercase",
            "match_type": "variant_exact",
            "context": "BUJUTSU AND BUDO",
            "reviewed_value": None,
            "notes": ["needs image review"],
        }
    ]

    markdown_path = tmp_path / "queue.md"
    csv_path = tmp_path / "queue.csv"
    helper.write_review_queue_markdown(markdown_path, rows)
    helper.write_review_queue_csv(csv_path, rows)

    assert "BUJUTSU AND BUDO" in markdown_path.read_text(encoding="utf-8")
    assert "BUDŌ" in markdown_path.read_text(encoding="utf-8")
    assert "uppercase" in csv_path.read_text(encoding="utf-8")
    assert "needs image review" in csv_path.read_text(encoding="utf-8")
