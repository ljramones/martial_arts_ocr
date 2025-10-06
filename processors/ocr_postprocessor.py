"""
OCR post-processing and text cleanup utilities.
"""
import re
import json
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)


class OCRPostProcessor:
    """Handles post-OCR text cleanup and corrections."""

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.symspell = self._initialize_symspell()

        # Load corrections in order of application
        self.general_corrections = self._load_general_corrections()
        self.typewriter_corrections = self._load_typewriter_corrections()
        self.domain_corrections = {}

        if domain != "general":
            self._load_domain_corrections(domain)

    def _initialize_symspell(self):
        """Initialize SymSpell for spelling correction."""
        try:
            from symspellpy import SymSpell, Verbosity
            sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Update path to processors/data/
            freq_path = Path(__file__).parent / "data" / "frequency_en_82k.txt"
            if freq_path.exists():
                sym.load_dictionary(str(freq_path), term_index=0, count_index=1, separator="\t")
                return sym
        except Exception:
            pass
        return None

    def _load_general_corrections(self) -> Dict:
        """Load general OCR corrections that apply to most documents."""
        return {
            'char_substitutions': {
                # Very common OCR errors
                'rn': 'm',
                'vv': 'w',
                'cl': 'd',
                'cI': 'd',
                ' ls ': ' is ',
                'ls ': 'is ',  # Add this for beginning of line
                ' ls': ' is',  # Add this for end of line
                'Wwe': 'We',
                'alva': 'always',
                # Add these specific fixes
                'tock': 'took',
                'nount aoe': 'mounted it',
                ' ron ': ' for ',
                'ryw': 'ryu',
                'S500': '500',
                'geneologists': 'genealogists',
                'aji®': 'aji"',
                'c¢)': 'c)',
            },
            'regex_patterns': [
                # Character confusions
                (r'(?<=[a-z])0(?=[a-z])', 'o'),
                (r'(?<=[a-z])1(?=[a-z])', 'l'),
                (r'(?<=[a-z])5(?=[a-z])', 's'),
                (r'\|(?=[A-Z])', 'I'),

                # Fix double capital I to ll in word context
                (r'(?<=[a-zA-Z])II(?=[a-z])', 'll'),
                (r'(?<=[a-z])II(?=[a-zA-Z])', 'll'),
                (r'\bII(?=[a-z])', 'll'),

                # Common word-level errors
                (r'\bls\b', 'is'),
                (r'\blS\b', 'is'),
                (r'\bl5\b', 'is'),
                (r'\blt\b', 'it'),
                (r'\blf\b', 'If'),
                (r'\bln\b', 'In'),
                (r'\bwlth\b', 'with'),
                (r'\bthls\b', 'this'),
                (r'\bthl5\b', 'this'),
                (r'\bcone\b', 'one'),
                (r'\bpecple\b', 'people'),
                (r'\bdetemines\b', 'determines'),

                # Handle contractions with II
                (r"I'II\b", "I'll"),
                (r"we'II\b", "we'll"),
                (r"you'II\b", "you'll"),
                (r"they'II\b", "they'll"),

                # Whitespace and punctuation
                (r'\s+', ' '),
                (r'\n{3,}', '\n\n'),
                (r'^\s+|\s+$', ''),
                (r'\s+([.,;:!?])', r'\1'),
                (r'([.,;:!?])\1+', r'\1'),
                (r'([.!?])\s*"\s*', r'\1"'),
                (r'"\s*([.!?])', r'"\1'),
            ]
        }

    def _load_typewriter_corrections(self) -> Dict:
        """Load typewriter-specific corrections."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "typewriter_corrections.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass

        # Default typewriter corrections
        return {
            'artifacts': [
                (r'^[A-Z]{1,2}\s+(?=[a-z])', ''),  # Remove stray capitals at line start
                (r'\{[¢c]\}?', ''),  # Remove bracket artifacts
                (r'\+[a-z]\+\??\s*\.?\s*', ''),  # Remove +e+ type artifacts
                (r'^\s*LJ\s+', ''),  # Remove LJ artifact
            ]
        }

    def _load_domain_corrections(self, domain: str):
        """Load domain-specific corrections from configuration file."""
        try:
            # The config is in processors/config/, not ../config/
            config_path = Path(__file__).parent / "config" / f"{domain}_corrections.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.domain_corrections = json.load(f)
                logger.info(f"Loaded domain corrections for {domain}")
        except Exception as e:
            logger.warning(f"Could not load domain corrections for {domain}: {e}")

    def clean_text(self, text: str, confidence: float = 1.0, boxes: Optional[List[Dict]] = None) -> str:
        """Apply general-purpose OCR cleanup."""
        if not text:
            return text

        # 1. Basic normalization
        text = unicodedata.normalize("NFKC", text)

        # 2. Apply typewriter corrections if they look applicable
        if self._looks_like_typewriter(text):
            text = self._apply_typewriter_corrections(text)

        # 3. Line-based processing
        text = self._process_lines(text)

        # 4. Apply general corrections
        text = self._apply_general_corrections(text)

        # 5. Apply domain-specific corrections
        text = self.apply_domain_corrections(text)

        # 6. Statistical spelling correction (only for low confidence)
        if self.symspell and confidence < 0.8:
            text = self._apply_spelling_correction(text)

        # 7. Final whitespace cleanup
        text = self._normalize_whitespace(text)

        # 8. Debug and fix persistent "ls" issue
        # Debug: Find and examine the "ls" instances
        ls_matches = list(re.finditer(r'.{5}ls.{5}', text))
        if ls_matches:
            logger.warning(f"Found {len(ls_matches)} 'ls' patterns:")
            for match in ls_matches:
                logger.warning(f"  Context: '{match.group()}' (chars: {[ord(c) for c in match.group()]})")

        # Try multiple approaches to fix "ls"
        text = re.sub(r'\bls\b', 'is', text)  # Word boundary approach
        text = re.sub(r'(\s)ls(\s)', r'\1is\2', text)  # With spaces
        text = re.sub(r'(this\s)ls(\sthe)', r'\1is\2', text, flags=re.IGNORECASE)  # Specific pattern

        # Final absolute fallback
        text = text.replace(" ls ", " is ")
        text = text.replace("This ls", "This is")
        text = text.replace("this ls", "this is")

        return text

    def apply_domain_corrections(self, text: str) -> str:
        """Apply domain-specific corrections if configured."""
        if not self.domain_corrections:
            logger.info(f"No domain corrections loaded for {self.domain}")
            return text

        logger.info(f"Applying {len(self.domain_corrections)} domain correction types")  # Changed to info

        # Apply term corrections
        if 'term_corrections' in self.domain_corrections:
            for wrong, correct in self.domain_corrections['term_corrections'].items():
                text = text.replace(wrong, correct)

            # Apply common OCR errors specific to domain
        if 'common_ocr_errors' in self.domain_corrections:
            logger.info(
                f"Applying {len(self.domain_corrections['common_ocr_errors'])} common OCR error corrections")  # Changed to info
            for wrong, correct in self.domain_corrections['common_ocr_errors'].items():
                if wrong in text:
                    logger.info(f"Replacing '{wrong}' with '{correct}'")  # Changed to info
                    text = text.replace(wrong, correct)

        # Apply contextual regex patterns
        if 'contextual_corrections' in self.domain_corrections:
            for correction in self.domain_corrections['contextual_corrections']:
                pattern = correction['pattern']
                replacement = correction['replacement']
                flags = re.IGNORECASE if correction.get('ignore_case', False) else 0
                text = re.sub(pattern, replacement, text, flags=flags)

        # Apply phrase corrections
        if 'phrase_corrections' in self.domain_corrections:
            for correction in self.domain_corrections['phrase_corrections']:
                text = text.replace(correction['wrong'], correction['correct'])

        # Clean up section markers
        if 'section_markers' in self.domain_corrections:
            for marker in self.domain_corrections['section_markers']:
                pattern = marker['pattern']
                replacement = marker['replacement']
                text = re.sub(pattern, replacement, text)

        # Apply quote fixes
        if 'quote_fixes' in self.domain_corrections:
            for fix in self.domain_corrections['quote_fixes']:
                pattern = fix['pattern']
                replacement = fix['replacement']
                text = re.sub(pattern, replacement, text)

        text = re.sub(r'\bls\b', 'is', text)

        return text

    def _looks_like_typewriter(self, text: str) -> bool:
        """Detect if text looks like it came from a typewriter."""
        # Check for common typewriter artifacts
        indicators = [
            r'^\s*[A-Z]{1,2}\s+[a-z]',  # Stray capitals at start
            r'\{[¢c]\}',  # Bracket artifacts
            r'\+[a-z]\+',  # Plus sign artifacts
            r'\bls\b',  # Common l/i confusion
            r'^\s*LJ\s+',  # LJ artifact
        ]

        # Check first 500 chars for efficiency
        sample = text[:500] if len(text) > 500 else text
        matches = sum(1 for pattern in indicators if re.search(pattern, sample, re.MULTILINE))
        return matches >= 2

    def _apply_typewriter_corrections(self, text: str) -> str:
        """Apply typewriter-specific corrections."""
        if 'artifacts' in self.typewriter_corrections:
            for pattern, replacement in self.typewriter_corrections['artifacts']:
                if isinstance(pattern, str):
                    text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        if 'typewriter_artifacts' in self.typewriter_corrections:
            for artifact in self.typewriter_corrections['typewriter_artifacts']:
                pattern = artifact['pattern']
                replacement = artifact['replacement']
                text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        if 'misaligned_characters' in self.typewriter_corrections:
            for char_fix in self.typewriter_corrections['misaligned_characters']:
                pattern = char_fix['pattern']
                replacement = char_fix['replacement']
                text = re.sub(pattern, replacement, text)

        return text

    def _process_lines(self, text: str) -> str:
        """Process text line by line for hyphenation and merging."""
        lines = text.split('\n')
        processed = []
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Handle hyphenated line breaks
            if line.endswith('-') and i + 1 < len(lines):
                next_line = lines[i + 1].lstrip()
                # Check if next line starts with lowercase (word continuation)
                if next_line and next_line[0].islower():
                    # Merge hyphenated word
                    processed.append(line[:-1] + next_line)
                    i += 2
                    continue

            # Check for soft wrap (line continues without punctuation)
            if processed and line and line[0].islower():
                last = processed[-1]
                # Don't merge if previous line ends with sentence terminator
                if not re.search(r'[.!?:;]$', last):
                    # Merge with previous line
                    processed[-1] = last + ' ' + line
                    i += 1
                    continue

            processed.append(line)
            i += 1

        # Remove duplicate consecutive lines
        deduped = []
        for line in processed:
            if not deduped or line != deduped[-1]:
                deduped.append(line)

        return '\n'.join(deduped)

    def _apply_general_corrections(self, text: str) -> str:

        """Apply general OCR corrections."""
        corrections = self.general_corrections

        # Apply character substitutions (context-free)
        for wrong, correct in corrections['char_substitutions'].items():
            text = text.replace(wrong, correct)

        # Apply regex patterns
        for pattern, replacement in corrections['regex_patterns']:
            text = re.sub(pattern, replacement, text)

        # Debug: Check before fixing ls
        if 'ls' in text:
            logger.debug(f"Found 'ls' in text before final fix")

        # Final aggressive fix for persistent "ls" problem
        text = re.sub(r'\bls\b', 'is', text)

        # Debug: Check after fixing ls
        if 'ls' in text:
            logger.warning(f"Still have 'ls' in text after regex fix!")

        # Also handle "Ls" at beginning of sentences
        text = re.sub(r'\bLs\b', 'Is', text)

        return text

    def _fix_ls_errors(self, text: str) -> str:
        """Specifically target 'ls' OCR errors which are very common in typewriter text."""
        # Use word boundaries to ensure we're catching standalone 'ls'
        text = re.sub(r'\bls\b', 'is', text)
        return text

    def _apply_spelling_correction(self, text: str) -> str:
        """Apply conservative spelling correction using SymSpell."""
        if not self.symspell:
            return text

        # Common words to never correct
        whitelist = {
            'say', 'day', 'took', 'take', 'make', 'made', 'own', 'way',
            'system', 'systems', 'this', 'that', 'these', 'those',
            'will', 'well', 'may', 'any', 'only', 'over', 'time',
            'human', 'style', 'styles', 'japanese', 'ryu', 'ha',
            'kata', 'kumite', 'dojo', 'sensei', 'karate', 'judo',
            'aikido', 'kendo', 'iaido', 'bushido', 'samurai', 'ninja',
        }

        lines = text.split('\n')
        corrected_lines = []

        for line in lines:
            # Find words to potentially correct
            words = re.findall(r'\b[A-Za-z]+\b', line)
            corrections_map = {}

            for word in words:
                # Skip short words, whitelist, and proper nouns
                if len(word) < 4 or word.lower() in whitelist or word[0].isupper():
                    continue

                suggestions = self.symspell.lookup(
                    word,
                    verbosity=1,  # Top suggestion only
                    max_edit_distance=1
                )

                if suggestions and suggestions[0].distance == 1:
                    # Only apply if frequency is significantly higher
                    orig_freq = self.symspell.words.get(word.lower(), 0)
                    sugg_freq = self.symspell.words.get(suggestions[0].term, 0)

                    if sugg_freq > orig_freq * 10:
                        # Preserve original capitalization
                        if word.isupper():
                            replacement = suggestions[0].term.upper()
                        elif word[0].isupper():
                            replacement = suggestions[0].term.capitalize()
                        else:
                            replacement = suggestions[0].term

                        corrections_map[word] = replacement

            # Apply all corrections to the line
            for wrong, correct in corrections_map.items():
                line = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, line)

            corrected_lines.append(line)

        return '\n'.join(corrected_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Final whitespace normalization."""
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize line endings
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)

        # Collapse excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)

        # Fix spacing after sentence endings
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        return text.strip()
