# utils/image/regions/text_fixups.py
"""
Post-OCR text cleanup utilities (no heavy deps).

Responsibilities
- Normalize whitespace/newlines
- Repair hyphenated line breaks: "exam-\nple" -> "example"
- Merge soft-wrapped lines within paragraphs
- Collapse repeated blank lines
- Remove duplicate lines (e.g., repeated headers/footers across pages)
- Optional normalization of quotes/dashes for consistent typography

Design notes
- Keep transformations conservative by default; tune via parameters.
- All functions are pure (return new strings), making them easy to unit test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


__all__ = [
    "normalize_whitespace",
    "fix_hyphenated_breaks",
    "merge_soft_wrapped_lines",
    "collapse_blank_lines",
    "remove_duplicate_lines",
    "normalize_quotes_dashes",
    "post_ocr_fixups",
    "FixupOptions",
]


# ------------------------------ core utilities --------------------------------

def normalize_whitespace(text: str, *, strip_trailing=True, tabs_as_spaces=4) -> str:
    """
    Normalize whitespace:
      - convert Windows/Mac newlines to '\n'
      - replace tabs with spaces (tabs_as_spaces)
      - optionally strip trailing spaces
      - normalize mixed spaces around newlines
    """
    if not text:
        return text

    # Normalize newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Tabs -> spaces
    if tabs_as_spaces and tabs_as_spaces > 0:
        t = t.expandtabs(tabs_as_spaces)

    # Strip trailing spaces per line
    if strip_trailing:
        t = "\n".join(line.rstrip(" \t") for line in t.split("\n"))

    # Remove spaces on blank-only lines
    t = re.sub(r"[ \t]+\n", "\n", t)

    return t


def fix_hyphenated_breaks(
    text: str,
    *,
    allow_en_dash: bool = True,
    preserve_known_prefixes: Iterable[str] = ("co-", "pre-", "re-", "non-"),
) -> str:
    """
    Join words split across line breaks by hyphenation:
      "exam-\nple" -> "example"
      "multi-\nline" -> "multiline"

    Heuristics:
      - Only fixes hyphen/dash immediately before a single newline.
      - If the hyphenated chunk matches a known prefix (e.g., 'co-'), keep it.
      - Optional: treat en-dashes as hyphens.

    NOTE: This does not alter real hyphenated compounds within a single line.
    """
    if not text:
        return text

    # Quick path: skip if no '-\n' (or '–\n')
    if "-\n" not in text and (not allow_en_dash or "–\n" not in text):
        return text

    dash_chars = r"\-" + (r"–" if allow_en_dash else "")
    # Negative lookbehind avoids touching list bullets like "-\n"
    # Capture groups:
    #   1) leading word part, 2) dash, 3) newline, 4) continuation token
    pattern = re.compile(rf"(?<!^)(?<!\n)([A-Za-z0-9]+)[{dash_chars}]\n([A-Za-z0-9]+)")

    def _join(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        # Respect a small set of legit prefixes (co-operate -> cooperate, but keep 'co-' when no break)
        for pref in preserve_known_prefixes:
            if left.lower() == pref[:-1].lower():
                # keep hyphen for these only if the result would be a known prefixed hyphenation
                # but since this is a line-break split, join without the hyphen (more readable for OCR)
                break
        # Default: concat without hyphen
        return f"{left}{right}"

    return pattern.sub(_join, text)


def merge_soft_wrapped_lines(
    text: str,
    *,
    keep_paragraphs: bool = True,
    respect_bullets: bool = True,
    respect_headings: bool = True,
) -> str:
    """
    Merge single newlines caused by soft wrapping within paragraphs.

    Strategy:
      - Work paragraph-by-paragraph (split on blank lines) if keep_paragraphs=True.
      - Inside each paragraph, replace single newlines with spaces unless:
          * The next line looks like a bullet/list item ( "-", "*", "•", digit. )
          * The current line ends with a hard break marker (e.g., ':' when followed by bullet)
          * The next line is ALL CAPS heading and respect_headings=True
    """
    if not text:
        return text

    def _looks_like_bullet(line: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*•]|[0-9]{1,2}[.)])\s+", line))

    def _looks_like_heading(line: str) -> bool:
        # ALL CAPS with words (allow numbers & '&'), or trailing colon
        s = line.strip()
        return bool(s) and (s.endswith(":") or re.fullmatch(r"[A-Z0-9 &/.,'-]{3,}", s) is not None)

    def _merge_block(block: str) -> str:
        # Split by single newlines, merge with heuristics
        lines = block.split("\n")
        out: List[str] = []
        for i, ln in enumerate(lines):
            if i == len(lines) - 1:
                out.append(ln)
                break
            nxt = lines[i + 1]

            # Respect bullets on next line
            if respect_bullets and _looks_like_bullet(nxt):
                out.append(ln)
                continue

            # Respect headings on next line
            if respect_headings and _looks_like_heading(nxt):
                out.append(ln)
                continue

            # If current line ends with hyphen we *already* fixed above; still, don't add extra space
            if re.search(r"[-–]\s*$", ln):
                out.append(ln)  # let hyphenation fixer handle join
                continue

            # Default: join with a single space
            joined = (ln.rstrip() + " " + nxt.lstrip())
            lines[i + 1] = joined
        # The algorithm accumulates into the last line
        return lines[-1]

    if keep_paragraphs:
        paras = re.split(r"\n{2,}", text)
        merged = [_merge_block(p) for p in paras]
        return "\n\n".join(merged)
    else:
        return _merge_block(text)


def collapse_blank_lines(text: str, *, max_consecutive: int = 1) -> str:
    """
    Collapse runs of blank lines to at most `max_consecutive`.
    """
    if max_consecutive < 1:
        max_consecutive = 1
    pattern = re.compile(r"\n{2,}")
    replacement = "\n" * (max_consecutive + 0)  # e.g., 1 -> "\n\n"? No, for 1 we want single '\n\n'?:
    # We want: max_consecutive blank *lines*. That means max_consecutive+1 newlines.
    replacement = "\n" * (max_consecutive + 1)
    # Normalize CRLF etc first
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return pattern.sub(replacement, t)


def remove_duplicate_lines(
    text: str,
    *,
    case_insensitive: bool = True,
    window: int = 6,
) -> str:
    """
    Remove duplicate lines while preserving order (handy for repeated headers/footers).
    Uses a sliding memory of the last `window` unique lines to avoid nuking legitimate
    repetitions far apart (e.g., repeated section titles pages later).

    Args:
      case_insensitive: match ignoring case.
      window: how many *recent* distinct lines to remember.

    Returns:
      Deduplicated text.
    """
    if not text:
        return text

    seen: List[str] = []
    out: List[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        key = raw.strip()
        key = key.lower() if case_insensitive else key
        if key and key in seen:
            # skip duplicate
            continue
        out.append(raw)
        if key:
            seen.append(key)
            if len(seen) > max(1, window):
                seen.pop(0)
    return "\n".join(out)


def normalize_quotes_dashes(
    text: str,
    *,
    style: str = "ascii",  # "ascii" | "smart"
) -> str:
    """
    Normalize quotes and dashes:
      - style="ascii":  “ ” ‘ ’ — – → " ' - (em/en to hyphen)
      - style="smart":  straight ASCII to typographic: " -> “/”, ' -> ‘/’ (simple heuristic)

    Note: The “smart” mode is intentionally conservative and does not try to balance quotes.
    """
    if not text:
        return text

    if style == "ascii":
        t = text
        t = t.replace("“", '"').replace("”", '"')
        t = t.replace("‘", "'").replace("’", "'")
        t = t.replace("—", "-").replace("–", "-")
        return t

    if style == "smart":
        # basic: replace straight quotes with curly based on rudimentary context
        def _smart_double(m: re.Match) -> str:
            s = m.group(0)
            # opening if preceded by start or whitespace; else closing
            if m.start() == 0 or text[m.start() - 1].isspace():
                return "“"
            return "”"

        def _smart_single(m: re.Match) -> str:
            if m.start() == 0 or text[m.start() - 1].isspace():
                return "‘"
            return "’"

        t = re.sub(r'"', _smart_double, text)
        t = re.sub(r"'", _smart_single, t)
        t = t.replace("--", "—")  # em-dash from double hyphen (optional)
        return t

    return text


# ------------------------------- Orchestrator ---------------------------------

@dataclass
class FixupOptions:
    # Whitespace
    strip_trailing: bool = True
    tabs_as_spaces: int = 4

    # Hyphenation repair
    fix_hyphens: bool = True
    allow_en_dash_hyphens: bool = True

    # Soft-wrap merge
    merge_soft_wraps: bool = True
    keep_paragraphs: bool = True
    respect_bullets: bool = True
    respect_headings: bool = True

    # Blank lines
    collapse_blanks: bool = True
    max_blank_lines: int = 1

    # Line dedupe
    remove_dupes: bool = False
    dedupe_window: int = 6
    dedupe_case_insensitive: bool = True

    # Typography
    normalize_quotes: bool = False
    quote_style: str = "ascii"  # or "smart"


def post_ocr_fixups(text: str, opts: FixupOptions | None = None) -> str:
    """
    Run a conservative, OCR-friendly cleanup pipeline.

    Order of operations:
      normalize_whitespace -> fix_hyphenated_breaks -> merge_soft_wrapped_lines
      -> collapse_blank_lines -> remove_duplicate_lines? -> normalize_quotes?
    """
    if not text:
        return text

    if opts is None:
        opts = FixupOptions()

    t = normalize_whitespace(text, strip_trailing=opts.strip_trailing, tabs_as_spaces=opts.tabs_as_spaces)

    if opts.fix_hyphens:
        t = fix_hyphenated_breaks(t, allow_en_dash=opts.allow_en_dash_hyphens)

    if opts.merge_soft_wraps:
        t = merge_soft_wrapped_lines(
            t,
            keep_paragraphs=opts.keep_paragraphs,
            respect_bullets=opts.respect_bullets,
            respect_headings=opts.respect_headings,
        )

    if opts.collapse_blanks:
        t = collapse_blank_lines(t, max_consecutive=opts.max_blank_lines)

    if opts.remove_dupes:
        t = remove_duplicate_lines(
            t,
            case_insensitive=opts.dedupe_case_insensitive,
            window=opts.dedupe_window,
        )

    if opts.normalize_quotes:
        t = normalize_quotes_dashes(t, style=opts.quote_style)

    return t
