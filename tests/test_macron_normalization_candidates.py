from __future__ import annotations

from utils.text.macron_candidates import (
    MacronGlossaryTerm,
    find_macron_normalization_candidates,
)


def test_exact_ascii_variant_creates_review_candidate():
    text = "The Daito-ryu tradition is discussed."

    candidates = find_macron_normalization_candidates(text)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.observed == "Daito-ryu"
    assert candidate.candidate == "Daitō-ryū"
    assert candidate.span == (4, 13)
    assert candidate.match_type == "variant_exact"
    assert candidate.requires_review is True
    assert candidate.confidence == "candidate"
    assert candidate.term_category == "style"


def test_candidate_to_dict_uses_json_friendly_shape():
    candidate = find_macron_normalization_candidates("Daito-ryu")[0].to_dict()

    assert candidate == {
        "observed": "Daito-ryu",
        "candidate": "Daitō-ryū",
        "span": [0, 9],
        "context": "Daito-ryu",
        "source": "martial_arts_macron_glossary",
        "match_type": "variant_exact",
        "requires_review": True,
        "confidence": "candidate",
        "term_category": "style",
        "ambiguous": False,
        "case_pattern": "titlecase",
        "reviewed_value_suggestion": "Daitō-ryū",
        "notes": [],
    }


def test_case_insensitive_match_preserves_observed_form():
    text = "KORYU and budo appear in OCR."

    candidates = find_macron_normalization_candidates(text)

    assert [(c.observed, c.candidate) for c in candidates] == [
        ("KORYU", "koryū"),
        ("budo", "budō"),
    ]
    assert [(c.case_pattern, c.reviewed_value_suggestion) for c in candidates] == [
        ("uppercase", "KORYŪ"),
        ("lowercase", "budō"),
    ]


def test_uppercase_observed_text_gets_uppercase_review_suggestion():
    candidate = find_macron_normalization_candidates("1. BUJUTSU AND BUDO.")[0]

    assert candidate.observed == "BUDO"
    assert candidate.candidate == "budō"
    assert candidate.case_pattern == "uppercase"
    assert candidate.reviewed_value_suggestion == "BUDŌ"
    assert candidate.requires_review is True


def test_mixed_case_observed_text_keeps_canonical_review_suggestion():
    candidate = find_macron_normalization_candidates("The OCR saw KoRyU.")[0]

    assert candidate.observed == "KoRyU"
    assert candidate.case_pattern == "mixed"
    assert candidate.reviewed_value_suggestion == "koryū"


def test_canonical_macron_text_does_not_create_replacement_candidate():
    text = "He studied koryū budō and Daitō-ryū."

    candidates = find_macron_normalization_candidates(text)

    assert candidates == []


def test_word_boundary_matching_avoids_false_positive_substrings():
    text = "The embudoed line and koryubase token are unrelated."

    candidates = find_macron_normalization_candidates(text)

    assert candidates == []


def test_hyphen_and_space_variants_match_known_terms_only():
    text = "Daito ryu and Daitoryu are OCR variants."

    candidates = find_macron_normalization_candidates(text)

    assert [(c.observed, c.candidate) for c in candidates] == [
        ("Daito ryu", "Daitō-ryū"),
        ("Daitoryu", "Daitō-ryū"),
    ]
    assert all(c.match_type == "variant_hyphen_space" for c in candidates)


def test_known_ocr_confusion_creates_review_required_candidate():
    text = "OCR saw koryG, bud6, and Dait6-rya."

    candidates = find_macron_normalization_candidates(text)

    assert [(c.observed, c.candidate, c.match_type) for c in candidates] == [
        ("koryG", "koryū", "observed_ocr_confusion"),
        ("bud6", "budō", "observed_ocr_confusion"),
        ("Dait6-rya", "Daitō-ryū", "observed_ocr_confusion"),
    ]
    assert all(c.requires_review for c in candidates)


def test_ambiguous_candidates_are_marked_and_not_suppressed():
    glossary = (
        MacronGlossaryTerm(canonical="ryū", variants=("ryu",), category="term"),
        MacronGlossaryTerm(canonical="Ryū", variants=("ryu",), category="name"),
    )

    candidates = find_macron_normalization_candidates("ryu", glossary=glossary)

    assert [candidate.candidate for candidate in candidates] == ["Ryū", "ryū"]
    assert all(candidate.ambiguous for candidate in candidates)
    assert all(candidate.requires_review for candidate in candidates)


def test_context_span_is_preserved():
    text = "before context Daito-ryu after context"

    candidate = find_macron_normalization_candidates(text, context_chars=7)[0]

    assert candidate.span == (15, 24)
    assert candidate.context == "ontext Daito-ryu after "


def test_source_text_is_not_mutated():
    text = "Daito-ryu Aiki-jujutsu"
    original = text[:]

    find_macron_normalization_candidates(text)

    assert text == original
