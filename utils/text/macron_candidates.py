"""Review-only macron normalization candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


SOURCE_NAME = "martial_arts_macron_glossary"


@dataclass(frozen=True)
class MacronGlossaryTerm:
    canonical: str
    variants: tuple[str, ...]
    category: str
    language: str = "ja-Latn"
    ocr_confusions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MacronCandidate:
    observed: str
    candidate: str
    span: tuple[int, int]
    context: str
    source: str = SOURCE_NAME
    match_type: str = "variant_exact"
    requires_review: bool = True
    confidence: str = "candidate"
    term_category: str | None = None
    ambiguous: bool = False
    case_pattern: str = "as_glossary"
    reviewed_value_suggestion: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "candidate": self.candidate,
            "span": [int(self.span[0]), int(self.span[1])],
            "context": self.context,
            "source": self.source,
            "match_type": self.match_type,
            "requires_review": self.requires_review,
            "confidence": self.confidence,
            "term_category": self.term_category,
            "ambiguous": self.ambiguous,
            "case_pattern": self.case_pattern,
            "reviewed_value_suggestion": self.reviewed_value_suggestion,
            "notes": list(self.notes),
        }


DEFAULT_GLOSSARY: tuple[MacronGlossaryTerm, ...] = (
    MacronGlossaryTerm(
        canonical="koryū",
        variants=("koryu",),
        ocr_confusions=("koryG",),
        category="tradition",
    ),
    MacronGlossaryTerm(
        canonical="budō",
        variants=("budo",),
        ocr_confusions=("bud6",),
        category="art",
    ),
    MacronGlossaryTerm(
        canonical="Daitō-ryū",
        variants=("Daito-ryu", "Daito ryu", "Daitoryu", "Daitō ryu"),
        ocr_confusions=("Dait6-rya", "Daito-ryt"),
        category="style",
    ),
    MacronGlossaryTerm(
        canonical="jūjutsu",
        variants=("jujutsu", "ju-jutsu"),
        ocr_confusions=("jGjutsu", "jajutsu"),
        category="art",
    ),
    MacronGlossaryTerm(
        canonical="dōjō",
        variants=("dojo",),
        ocr_confusions=("d6j6",),
        category="place",
    ),
    MacronGlossaryTerm(
        canonical="ryūha",
        variants=("ryuha",),
        ocr_confusions=("ryaha",),
        category="school",
    ),
    MacronGlossaryTerm(
        canonical="sōke",
        variants=("soke",),
        category="title",
    ),
    MacronGlossaryTerm(
        canonical="iaidō",
        variants=("iaido",),
        ocr_confusions=("iaid6", "iaidd"),
        category="art",
    ),
    MacronGlossaryTerm(
        canonical="aikijūjutsu",
        variants=("aikijujutsu", "aiki-jujutsu"),
        ocr_confusions=("aikijGjutsu", "aikiytjutsu", "aikijdjutsu"),
        category="art",
    ),
)


def find_macron_normalization_candidates(
    text: str,
    *,
    glossary: tuple[MacronGlossaryTerm, ...] | list[MacronGlossaryTerm] = DEFAULT_GLOSSARY,
    context_chars: int = 32,
) -> list[MacronCandidate]:
    """Find review-required macron normalization candidates without mutating text."""

    matches: list[tuple[tuple[int, int], MacronCandidate]] = []
    for term in glossary:
        for phrase, match_type in _phrases_for_term(term):
            for match in _find_phrase_matches(text, phrase):
                observed = match.group(0)
                if observed == term.canonical:
                    continue
                candidate = MacronCandidate(
                    observed=observed,
                    candidate=term.canonical,
                    span=(match.start(), match.end()),
                    context=_context(text, match.start(), match.end(), context_chars),
                    match_type=match_type,
                    term_category=term.category,
                    case_pattern=_case_pattern(observed),
                    reviewed_value_suggestion=_reviewed_value_suggestion(observed, term.canonical),
                    notes=term.notes,
                )
                matches.append(((match.start(), match.end()), candidate))

    return _mark_ambiguity(_deduplicate(matches))


def _phrases_for_term(term: MacronGlossaryTerm) -> list[tuple[str, str]]:
    phrases: list[tuple[str, str]] = []
    ascii_canonical = _strip_macrons(term.canonical)
    normalized_ascii_canonical = _normalized_hyphen_space(ascii_canonical)
    for variant in term.variants:
        match_type = "variant_exact"
        if (
            variant.casefold() != ascii_canonical.casefold()
            and _normalized_hyphen_space(variant) == normalized_ascii_canonical
        ):
            match_type = "variant_hyphen_space"
        phrases.append((variant, match_type))
    for confusion in term.ocr_confusions:
        phrases.append((confusion, "observed_ocr_confusion"))
    return phrases


def _find_phrase_matches(text: str, phrase: str) -> list[re.Match[str]]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return list(pattern.finditer(text))


def _context(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    return text[left:right]


def _deduplicate(matches: list[tuple[tuple[int, int], MacronCandidate]]) -> list[MacronCandidate]:
    seen: set[tuple[int, int, str, str]] = set()
    candidates: list[MacronCandidate] = []
    for _span, candidate in sorted(matches, key=lambda item: (item[0][0], item[0][1], item[1].candidate)):
        key = (candidate.span[0], candidate.span[1], candidate.observed.lower(), candidate.candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _mark_ambiguity(candidates: list[MacronCandidate]) -> list[MacronCandidate]:
    span_counts: dict[tuple[int, int, str], int] = {}
    for candidate in candidates:
        key = (candidate.span[0], candidate.span[1], candidate.observed.lower())
        span_counts[key] = span_counts.get(key, 0) + 1

    marked: list[MacronCandidate] = []
    for candidate in candidates:
        key = (candidate.span[0], candidate.span[1], candidate.observed.lower())
        if span_counts[key] <= 1:
            marked.append(candidate)
            continue
        marked.append(
            MacronCandidate(
                observed=candidate.observed,
                candidate=candidate.candidate,
                span=candidate.span,
                context=candidate.context,
                source=candidate.source,
                match_type=candidate.match_type,
                requires_review=True,
                confidence=candidate.confidence,
                term_category=candidate.term_category,
                ambiguous=True,
                case_pattern=candidate.case_pattern,
                reviewed_value_suggestion=candidate.reviewed_value_suggestion,
                notes=candidate.notes,
            )
        )
    return marked


def _normalized_hyphen_space(text: str) -> str:
    return re.sub(r"[-\s]+", "", text).casefold()


def _case_pattern(observed: str) -> str:
    letters = [char for char in observed if char.isalpha()]
    if not letters:
        return "as_glossary"
    if all(char.isupper() for char in letters):
        return "uppercase"
    if all(char.islower() for char in letters):
        return "lowercase"
    if observed[:1].isupper() and observed[1:].islower():
        return "titlecase"
    return "mixed"


def _reviewed_value_suggestion(observed: str, canonical: str) -> str:
    case_pattern = _case_pattern(observed)
    if case_pattern == "uppercase":
        return canonical.upper()
    if case_pattern == "titlecase":
        return canonical[:1].upper() + canonical[1:]
    return canonical


def _strip_macrons(text: str) -> str:
    return (
        text.replace("ā", "a")
        .replace("ē", "e")
        .replace("ī", "i")
        .replace("ō", "o")
        .replace("ū", "u")
        .replace("Ā", "A")
        .replace("Ē", "E")
        .replace("Ī", "I")
        .replace("Ō", "O")
        .replace("Ū", "U")
    )
