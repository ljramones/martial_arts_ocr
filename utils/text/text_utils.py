"""
Text processing utilities for Martial Arts OCR.
Handles text cleaning, language detection, formatting, and OCR post-processing.
"""
import re
import string
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from pathlib import Path

# Language detection
try:
    from langdetect import detect, detect_langs

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available - language detection disabled")

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """Represents a segment of text with metadata."""
    text: str
    language: str = "unknown"
    confidence: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    segment_type: str = "text"  # "text", "japanese", "punctuation", "number"

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'type': self.segment_type
        }


@dataclass
class CleaningStats:
    """Statistics from text cleaning operations."""
    original_length: int
    cleaned_length: int
    lines_removed: int
    characters_removed: int
    words_before: int
    words_after: int

    @property
    def compression_ratio(self) -> float:
        return self.cleaned_length / self.original_length if self.original_length > 0 else 0

    def to_dict(self) -> Dict:
        return {
            'original_length': self.original_length,
            'cleaned_length': self.cleaned_length,
            'lines_removed': self.lines_removed,
            'characters_removed': self.characters_removed,
            'words_before': self.words_before,
            'words_after': self.words_after,
            'compression_ratio': self.compression_ratio
        }


class TextCleaner:
    """Clean and normalize OCR text output."""

    def __init__(self):
        # Common OCR errors and their corrections
        self.ocr_corrections = {
            # Common character substitutions
            'rn': 'm',  # common OCR error
            'vv': 'w',
            '0': 'o',  # in words (context-dependent)
            '1': 'l',  # in words (context-dependent)
            '|': 'I',  # vertical bar to capital I
            # Japanese-specific corrections
            'ー': '一',  # long vowel mark to kanji one
            '口': 'ロ',  # sometimes confused
        }

        # Regex patterns for cleaning
        self.patterns = {
            'multiple_spaces': re.compile(r'\s{2,}'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'trailing_spaces': re.compile(r'[ \t]+$', re.MULTILINE),
            'leading_spaces': re.compile(r'^[ \t]+', re.MULTILINE),
            'ocr_artifacts': re.compile(r'[■□▪▫◊◆◇○●△▲▼▽]'),  # OCR artifacts
            'stray_punctuation': re.compile(r'^[.,;:!?]+\s*$', re.MULTILINE),
            'isolated_chars': re.compile(r'^\s*[a-zA-Z]\s*$', re.MULTILINE),
        }

        # Japanese character ranges
        self.japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Kanji)
            (0x3400, 0x4DBF),  # CJK Extension A
            (0xFF65, 0xFF9F),  # Half-width Katakana
        ]

    def clean_text(self, text: str, aggressive: bool = False) -> Tuple[str, CleaningStats]:
        """Clean OCR text with various filters."""
        original_text = text
        stats = CleaningStats(
            original_length=len(text),
            cleaned_length=0,
            lines_removed=0,
            characters_removed=0,
            words_before=len(text.split()),
            words_after=0
        )

        # Basic cleaning
        cleaned = self._basic_cleaning(text)

        # Remove OCR artifacts
        cleaned = self._remove_ocr_artifacts(cleaned)

        # Fix common OCR errors
        cleaned = self._fix_ocr_errors(cleaned)

        # Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        if aggressive:
            # More aggressive cleaning
            cleaned = self._aggressive_cleaning(cleaned)

        # Update stats
        stats.cleaned_length = len(cleaned)
        stats.characters_removed = stats.original_length - stats.cleaned_length
        stats.words_after = len(cleaned.split())

        logger.debug(f"Text cleaning: {stats.original_length} -> {stats.cleaned_length} chars")
        return cleaned, stats

    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Remove null bytes and control characters (except newlines and tabs)
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # Remove trailing and leading spaces on each line
        cleaned = self.patterns['trailing_spaces'].sub('', cleaned)
        cleaned = self.patterns['leading_spaces'].sub('', cleaned)

        return cleaned

    def _remove_ocr_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts and noise."""
        # Remove OCR box drawing and shape artifacts
        cleaned = self.patterns['ocr_artifacts'].sub('', text)

        # Remove lines with only punctuation
        cleaned = self.patterns['stray_punctuation'].sub('', cleaned)

        # Remove isolated single characters (likely OCR errors)
        cleaned = self.patterns['isolated_chars'].sub('', cleaned)

        return cleaned

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR character recognition errors."""
        cleaned = text

        for error, correction in self.ocr_corrections.items():
            # Context-aware replacements
            if error in ['0', '1']:
                # Only replace in word contexts, not numbers
                pattern = rf'\b[a-zA-Z]*{error}[a-zA-Z]+\b|\b[a-zA-Z]+{error}[a-zA-Z]*\b'
                cleaned = re.sub(pattern, lambda m: m.group().replace(error, correction), cleaned)
            else:
                cleaned = cleaned.replace(error, correction)

        return cleaned

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single space
        cleaned = self.patterns['multiple_spaces'].sub(' ', text)

        # Replace multiple newlines with double newline (paragraph break)
        cleaned = self.patterns['multiple_newlines'].sub('\n\n', cleaned)

        # Remove trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def _aggressive_cleaning(self, text: str) -> str:
        """More aggressive cleaning for heavily corrupted OCR."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue

            # Skip lines with too many special characters
            special_char_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / len(line)
            if special_char_ratio > 0.5:
                continue

            # Skip lines that are mostly numbers (unless they look like dates/references)
            if re.match(r'^\d+\s*$', line) and len(line) < 10:
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


class LanguageDetector:
    """Detect and segment text by language."""

    def __init__(self):
        self.available = LANGDETECT_AVAILABLE

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the primary language of text."""
        if not self.available or not text.strip():
            return "unknown", 0.0

        try:
            detected_langs = detect_langs(text)
            if detected_langs:
                primary = detected_langs[0]
                return primary.lang, primary.prob
            return "unknown", 0.0
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown", 0.0

    def segment_by_language(self, text: str) -> List[TextSegment]:
        """Segment text into language-specific blocks."""
        segments = []

        # First, detect Japanese characters
        japanese_segments = self._extract_japanese_segments(text)

        if not japanese_segments:
            # No Japanese detected, treat as single segment
            lang, confidence = self.detect_language(text)
            segments.append(TextSegment(
                text=text,
                language=lang,
                confidence=confidence,
                start_pos=0,
                end_pos=len(text)
            ))
        else:
            # Mixed language text
            last_pos = 0

            for jp_segment in japanese_segments:
                # Add English segment before Japanese (if any)
                if jp_segment.start_pos > last_pos:
                    eng_text = text[last_pos:jp_segment.start_pos].strip()
                    if eng_text:
                        segments.append(TextSegment(
                            text=eng_text,
                            language="en",
                            confidence=0.8,
                            start_pos=last_pos,
                            end_pos=jp_segment.start_pos,
                            segment_type="text"
                        ))

                # Add Japanese segment
                segments.append(jp_segment)
                last_pos = jp_segment.end_pos

            # Add remaining English text (if any)
            if last_pos < len(text):
                remaining_text = text[last_pos:].strip()
                if remaining_text:
                    segments.append(TextSegment(
                        text=remaining_text,
                        language="en",
                        confidence=0.8,
                        start_pos=last_pos,
                        end_pos=len(text),
                        segment_type="text"
                    ))

        return segments

    def _extract_japanese_segments(self, text: str) -> List[TextSegment]:
        """Extract segments containing Japanese characters."""
        segments = []

        # Find all Japanese character sequences
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\uFF65-\uFF9F]+')

        for match in japanese_pattern.finditer(text):
            segments.append(TextSegment(
                text=match.group(),
                language="ja",
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end(),
                segment_type="japanese"
            ))

        return segments

    def is_japanese_text(self, text: str) -> bool:
        """Check if text contains Japanese characters."""
        for char in text:
            code = ord(char)
            for start, end in [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)]:
                if start <= code <= end:
                    return True
        return False


class TextFormatter:
    """Format text for different output types."""

    @staticmethod
    def to_html(text: str, preserve_formatting: bool = True) -> str:
        """Convert text to HTML format."""
        if not preserve_formatting:
            text = re.sub(r'\n+', ' ', text)

        # Escape HTML characters
        html = (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

        if preserve_formatting:
            # Convert newlines to <br> tags
            html = html.replace('\n', '<br>\n')

        return html

    @staticmethod
    def to_markdown(text: str, title: Optional[str] = None) -> str:
        """Convert text to Markdown format."""
        markdown = ""

        if title:
            markdown = f"# {title}\n\n"

        # Simple paragraph detection
        paragraphs = re.split(r'\n\s*\n', text.strip())

        for para in paragraphs:
            if para.strip():
                markdown += para.strip() + "\n\n"

        return markdown.strip()

    @staticmethod
    def to_plain_text(text: str, max_line_length: int = 80) -> str:
        """Format as plain text with line wrapping."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)


class TextStatistics:
    """Calculate statistics about text content."""

    @staticmethod
    def get_stats(text: str) -> Dict:
        """Get comprehensive text statistics."""
        lines = text.split('\n')
        words = text.split()

        # Character counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))

        # Language composition
        japanese_chars = sum(1 for c in text if TextStatistics._is_japanese_char(c))
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())

        # Reading time estimation (words per minute)
        reading_time_minutes = len(words) / 200  # Average reading speed

        stats = {
            'characters': char_count,
            'characters_no_spaces': char_count_no_spaces,
            'words': len(words),
            'lines': len(lines),
            'paragraphs': len([line for line in lines if line.strip()]),
            'sentences': len(re.findall(r'[.!?]+', text)),
            'japanese_characters': japanese_chars,
            'english_characters': english_chars,
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / max(1, len(re.findall(r'[.!?]+', text))),
            'reading_time_minutes': reading_time_minutes,
            'language_ratio': {
                'japanese': japanese_chars / char_count if char_count > 0 else 0,
                'english': english_chars / char_count if char_count > 0 else 0
            }
        }

        return stats

    @staticmethod
    def _is_japanese_char(char: str) -> bool:
        """Check if character is Japanese."""
        code = ord(char)
        japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # Kanji
        ]
        return any(start <= code <= end for start, end in japanese_ranges)


class JapaneseUtils:
    """Utilities and data for Japanese text processing."""

    # Basic martial arts terminology dictionary
    martialArtsTerms = {
        # Basic martial arts
        '武道': {'romaji': 'budō', 'translation': 'martial way', 'category': 'martial_arts'},
        '武術': {'romaji': 'bujutsu', 'translation': 'martial art', 'category': 'martial_arts'},
        '空手': {'romaji': 'karate', 'translation': 'karate', 'category': 'martial_arts'},
        '空手道': {'romaji': 'karate-dō', 'translation': 'way of karate', 'category': 'martial_arts'},
        '柔道': {'romaji': 'jūdō', 'translation': 'judo', 'category': 'martial_arts'},
        '剣道': {'romaji': 'kendō', 'translation': 'kendo', 'category': 'martial_arts'},
        '合気道': {'romaji': 'aikidō', 'translation': 'aikido', 'category': 'martial_arts'},
        '柔術': {'romaji': 'jūjutsu', 'translation': 'jujutsu', 'category': 'martial_arts'},
        '居合道': {'romaji': 'iaidō', 'translation': 'iaido', 'category': 'martial_arts'},

        # Training and practice
        '型': {'romaji': 'kata', 'translation': 'form', 'category': 'training'},
        '形': {'romaji': 'kata', 'translation': 'form', 'category': 'training'},
        '組手': {'romaji': 'kumite', 'translation': 'sparring', 'category': 'training'},
        '乱取り': {'romaji': 'randori', 'translation': 'free practice', 'category': 'training'},
        '稽古': {'romaji': 'keiko', 'translation': 'practice', 'category': 'training'},
        '練習': {'romaji': 'renshū', 'translation': 'practice', 'category': 'training'},

        # Places and people
        '道場': {'romaji': 'dōjō', 'translation': 'dojo', 'category': 'place'},
        '先生': {'romaji': 'sensei', 'translation': 'teacher', 'category': 'person'},
        '師範': {'romaji': 'shihan', 'translation': 'master instructor', 'category': 'person'},
        '弟子': {'romaji': 'deshi', 'translation': 'student', 'category': 'person'},
        '生徒': {'romaji': 'seito', 'translation': 'student', 'category': 'person'},

        # Ranks and grades
        '段': {'romaji': 'dan', 'translation': 'dan rank', 'category': 'rank'},
        '級': {'romaji': 'kyū', 'translation': 'kyu grade', 'category': 'rank'},
        '帯': {'romaji': 'obi', 'translation': 'belt', 'category': 'rank'},
        '黒帯': {'romaji': 'kuro-obi', 'translation': 'black belt', 'category': 'rank'},
        '免許': {'romaji': 'menkyo', 'translation': 'license', 'category': 'rank'},

        # Philosophy and concepts
        '武士道': {'romaji': 'bushidō', 'translation': 'way of the warrior', 'category': 'philosophy'},
        '心': {'romaji': 'kokoro', 'translation': 'heart/mind', 'category': 'philosophy'},
        '気': {'romaji': 'ki', 'translation': 'spirit/energy', 'category': 'philosophy'},
        '和': {'romaji': 'wa', 'translation': 'harmony', 'category': 'philosophy'},
        '礼': {'romaji': 'rei', 'translation': 'bow/respect', 'category': 'etiquette'},
        '礼儀': {'romaji': 'reigi', 'translation': 'etiquette', 'category': 'etiquette'},

        # Weapons and equipment
        '刀': {'romaji': 'katana', 'translation': 'sword', 'category': 'weapon'},
        '剣': {'romaji': 'ken', 'translation': 'sword', 'category': 'weapon'},
        '木刀': {'romaji': 'bokutō', 'translation': 'wooden sword', 'category': 'weapon'},
        '竹刀': {'romaji': 'shinai', 'translation': 'bamboo sword', 'category': 'weapon'},
        '杖': {'romaji': 'jō', 'translation': 'staff', 'category': 'weapon'},
        '棒': {'romaji': 'bō', 'translation': 'staff', 'category': 'weapon'},
        '薙刀': {'romaji': 'naginata', 'translation': 'naginata', 'category': 'weapon'},

        # Clothing
        '道着': {'romaji': 'dōgi', 'translation': 'practice uniform', 'category': 'clothing'},
        '道衣': {'romaji': 'dōi', 'translation': 'practice jacket', 'category': 'clothing'},
        '袴': {'romaji': 'hakama', 'translation': 'hakama', 'category': 'clothing'},
        '着物': {'romaji': 'kimono', 'translation': 'kimono', 'category': 'clothing'},

        # Historical terms
        '武士': {'romaji': 'bushi', 'translation': 'warrior', 'category': 'historical'},
        '侍': {'romaji': 'samurai', 'translation': 'samurai', 'category': 'historical'},
        '忍者': {'romaji': 'ninja', 'translation': 'ninja', 'category': 'historical'},
        '浪人': {'romaji': 'rōnin', 'translation': 'masterless samurai', 'category': 'historical'},
        '将軍': {'romaji': 'shōgun', 'translation': 'shogun', 'category': 'historical'},
        '大名': {'romaji': 'daimyō', 'translation': 'feudal lord', 'category': 'historical'},

        # Schools and styles
        '流': {'romaji': 'ryū', 'translation': 'school/style', 'category': 'school'},
        '派': {'romaji': 'ha', 'translation': 'faction/group', 'category': 'school'},
        '会': {'romaji': 'kai', 'translation': 'association', 'category': 'organization'},
        '館': {'romaji': 'kan', 'translation': 'hall/building', 'category': 'place'},

        # Techniques and movements
        '技': {'romaji': 'waza', 'translation': 'technique', 'category': 'technique'},
        '投げ': {'romaji': 'nage', 'translation': 'throw', 'category': 'technique'},
        '固め': {'romaji': 'katame', 'translation': 'hold/pin', 'category': 'technique'},
        '当て': {'romaji': 'ate', 'translation': 'strike', 'category': 'technique'},
        '蹴り': {'romaji': 'keri', 'translation': 'kick', 'category': 'technique'},
        '突き': {'romaji': 'tsuki', 'translation': 'thrust', 'category': 'technique'},

        # Body parts and positions
        '手': {'romaji': 'te', 'translation': 'hand', 'category': 'body'},
        '足': {'romaji': 'ashi', 'translation': 'foot/leg', 'category': 'body'},
        '腰': {'romaji': 'koshi', 'translation': 'hip/waist', 'category': 'body'},
        '肩': {'romaji': 'kata', 'translation': 'shoulder', 'category': 'body'},

        # Common terms in Draeger's research
        '研究': {'romaji': 'kenkyū', 'translation': 'research', 'category': 'academic'},
        '歴史': {'romaji': 'rekishi', 'translation': 'history', 'category': 'academic'},
        '文化': {'romaji': 'bunka', 'translation': 'culture', 'category': 'academic'},
        '伝統': {'romaji': 'dentō', 'translation': 'tradition', 'category': 'academic'},
        '古典': {'romaji': 'koten', 'translation': 'classical', 'category': 'academic'},
        '現代': {'romaji': 'gendai', 'translation': 'modern', 'category': 'academic'},
    }

    @classmethod
    def get_terms_by_category(cls, category: str) -> Dict[str, Dict[str, str]]:
        """Get all terms belonging to a specific category."""
        return {term: info for term, info in cls.martialArtsTerms.items()
                if info.get('category') == category}

    @classmethod
    def search_terms(cls, query: str) -> List[Dict[str, str]]:
        """Search for terms by Japanese text, romaji, or translation."""
        query = query.lower()
        results = []

        for term, info in cls.martialArtsTerms.items():
            if (query in term.lower() or
                query in info.get('romaji', '').lower() or
                query in info.get('translation', '').lower()):
                result = {'term': term}
                result.update(info)
                results.append(result)

        return results

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all available categories."""
        categories = set()
        for info in cls.martialArtsTerms.values():
            categories.add(info.get('category', 'unknown'))
        return sorted(list(categories))


# Utility functions
def extract_martial_arts_terms(text: str) -> List[str]:
    """Extract potential martial arts terminology from text."""
    # Common martial arts terms (expandable)
    martial_arts_patterns = [
        r'\b(?:kata|kumite|dojo|sensei|karate|judo|aikido|kendo|iaido)\b',
        r'\b(?:bushido|samurai|ninja|ronin|shogun|daimyo)\b',
        r'\b(?:gi|hakama|obi|bokken|shinai|katana|wakizashi)\b',
        r'\b(?:dan|kyu|kyuu|menkyo|shihan|hanshi)\b',
        # Japanese terms in romaji
        r'\b[a-z]+(?:ryu|kai|kan|do|jutsu|gata)\b',
    ]

    terms = set()
    for pattern in martial_arts_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update(matches)

    return sorted(list(terms))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling Japanese punctuation."""
    # Sentence delimiters for both English and Japanese
    sentence_pattern = r'[.!?。！？]+\s*'
    sentences = re.split(sentence_pattern, text)

    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 3:  # Filter out very short fragments
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def normalize_japanese_text(text: str) -> str:
    """Normalize Japanese text (full-width to half-width, etc.)."""
    try:
        import jaconv
        # Convert full-width alphanumeric to half-width
        normalized = jaconv.z2h(text, ascii=True, digit=True)
        return normalized
    except ImportError:
        logger.warning("jaconv not available - Japanese normalization disabled")
        return text


def confidence_score_text(text: str) -> float:
    """Calculate confidence score for OCR text quality."""
    if not text.strip():
        return 0.0

    score = 1.0

    # Penalize for too many special characters
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_char_ratio > 0.3:
        score *= (1 - special_char_ratio)

    # Penalize for isolated characters
    words = text.split()
    single_char_words = sum(1 for word in words if len(word) == 1 and word.isalpha())
    if single_char_words > len(words) * 0.2:
        score *= 0.5

    # Boost score for recognizable patterns
    if re.search(r'\b(?:the|and|of|to|in|is|are|was|were)\b', text, re.IGNORECASE):
        score *= 1.1

    # Boost for Japanese patterns
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text):
        score *= 1.05

    return min(1.0, score)