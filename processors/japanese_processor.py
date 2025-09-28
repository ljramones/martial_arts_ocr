"""
Japanese Text Processor for Martial Arts OCR
Handles Japanese text processing including romanization, translation, and analysis.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Japanese processing libraries
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    logging.warning("MeCab not available - Japanese morphological analysis disabled")

try:
    import pykakasi
    PYKAKASI_AVAILABLE = True
except ImportError:
    PYKAKASI_AVAILABLE = False
    logging.warning("pykakasi not available - romanization capabilities limited")

try:
    import argostranslate.translate
    import argostranslate.package
    ARGOS_AVAILABLE = True
except ImportError:
    ARGOS_AVAILABLE = False
    logging.warning("Argos Translate not available - offline translation disabled")

from config import get_config
from utils.text_utils import TextCleaner, LanguageDetector, JapaneseUtils

logger = logging.getLogger(__name__)


@dataclass
class JapaneseTextSegment:
    """Represents a segment of Japanese text with analysis."""
    original_text: str
    romaji: Optional[str] = None
    translation: Optional[str] = None
    morphology: Optional[List[Dict]] = None
    reading: Optional[str] = None
    pos_tags: Optional[List[str]] = None
    confidence: float = 0.0
    text_type: str = "unknown"  # "hiragana", "katakana", "kanji", "mixed"
    start_pos: int = 0
    end_pos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'romaji': self.romaji,
            'translation': self.translation,
            'morphology': self.morphology,
            'reading': self.reading,
            'pos_tags': self.pos_tags,
            'confidence': self.confidence,
            'text_type': self.text_type,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'length': len(self.original_text)
        }


@dataclass
class JapaneseProcessingResult:
    """Results from Japanese text processing."""
    original_text: str
    segments: List[JapaneseTextSegment]
    overall_romaji: Optional[str]
    overall_translation: Optional[str]
    language_analysis: Dict[str, Any]
    martial_arts_terms: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]
    confidence_score: float
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'segments': [segment.to_dict() for segment in self.segments],
            'overall_romaji': self.overall_romaji,
            'overall_translation': self.overall_translation,
            'language_analysis': self.language_analysis,
            'martial_arts_terms': self.martial_arts_terms,
            'processing_metadata': self.processing_metadata,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'summary': {
                'total_segments': len(self.segments),
                'has_romaji': self.overall_romaji is not None,
                'has_translation': self.overall_translation is not None,
                'martial_arts_terms_found': len(self.martial_arts_terms),
                'character_count': len(self.original_text)
            }
        }


class JapaneseProcessor:
    """Main Japanese text processing class."""

    def __init__(self):
        self.config = get_config().JAPANESE_PROCESSING
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()

        # Initialize MeCab if available
        self.mecab_tagger = None
        if MECAB_AVAILABLE:
            self._initialize_mecab()

        # Initialize pykakasi if available
        self.kakasi_converter = None
        if PYKAKASI_AVAILABLE:
            self._initialize_kakasi()

        # Initialize Argos Translate if available
        self._initialize_argos()

        # Load martial arts terminology
        self.martial_arts_dict = self._load_martial_arts_dictionary()

        # Japanese character patterns
        self.hiragana_pattern = re.compile(r'[\u3040-\u309F]+')
        self.katakana_pattern = re.compile(r'[\u30A0-\u30FF]+')
        self.kanji_pattern = re.compile(r'[\u4E00-\u9FFF]+')
        self.japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')

    def _initialize_mecab(self):
        """Initialize MeCab morphological analyzer."""
        try:
            # Try different dictionary options
            dict_options = [
                '',  # Default dictionary
                '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd',
                '-d /usr/local/lib/mecab/dic/ipadic',
                f'-d {self.config.get("mecab_dict", "")}'
            ]

            for option in dict_options:
                try:
                    if option:
                        self.mecab_tagger = MeCab.Tagger(option)
                    else:
                        self.mecab_tagger = MeCab.Tagger()

                    # Test the tagger - FIXED: Use proper Japanese characters
                    test_result = self.mecab_tagger.parse("テスト")  # "test" in katakana
                    if test_result:
                        logger.info(f"MeCab initialized successfully with option: {option or 'default'}")
                        break
                except Exception as e:
                    logger.debug(f"MeCab option '{option}' failed: {e}")
                    continue

            if not self.mecab_tagger:
                logger.warning("Failed to initialize MeCab with any dictionary option")

        except Exception as e:
            logger.error(f"MeCab initialization failed: {e}")

    def _initialize_kakasi(self):
        """Initialize pykakasi for romanization."""
        try:
            kakasi = pykakasi.kakasi()

            # Configure romanization settings
            romanization_system = self.config.get('romanization_system', 'hepburn')

            if romanization_system == 'hepburn':
                kakasi.setMode("H", "a")  # Hiragana to ASCII (Hepburn)
                kakasi.setMode("K", "a")  # Katakana to ASCII
                kakasi.setMode("J", "a")  # Japanese to ASCII
            elif romanization_system == 'kunrei':
                kakasi.setMode("H", "a")
                kakasi.setMode("K", "a")
                kakasi.setMode("J", "a")
                # Note: pykakasi uses Hepburn by default, kunrei would need custom mapping

            self.kakasi_converter = kakasi.getConverter()
            logger.info(f"pykakasi initialized with {romanization_system} romanization")

        except Exception as e:
            logger.error(f"pykakasi initialization failed: {e}")
            self.kakasi_converter = None

    def _initialize_argos(self):
        """Initialize Argos Translate for offline translation."""
        try:
            if not ARGOS_AVAILABLE:
                return

            # Check if Japanese-English package is installed
            installed_packages = argostranslate.package.get_installed_packages()
            ja_en_package = None

            for package in installed_packages:
                if package.from_code == 'ja' and package.to_code == 'en':
                    ja_en_package = package
                    break

            if ja_en_package:
                logger.info("Argos Translate Japanese-English package found")
            else:
                logger.warning("Argos Translate Japanese-English package not installed")

        except Exception as e:
            logger.error(f"Argos Translate initialization failed: {e}")

    def _load_martial_arts_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Load martial arts terminology dictionary."""
        martial_arts_dict = {}

        # Start with built-in terms from utils with validation
        try:
            if hasattr(JapaneseUtils, 'martialArtsTerms'):
                base_terms = JapaneseUtils.martialArtsTerms
                if isinstance(base_terms, dict):
                    martial_arts_dict.update(base_terms)
                    logger.debug(f"Loaded {len(base_terms)} terms from JapaneseUtils")
                else:
                    logger.warning("JapaneseUtils.martialArtsTerms is not a dictionary")
            else:
                logger.warning("JapaneseUtils.martialArtsTerms not found")
        except Exception as e:
            logger.warning(f"Could not load built-in martial arts terms: {e}")

        # Try to load additional terms from external file with validation
        try:
            dict_file = Path(__file__).parent / "data" / "martial_arts_terms.json"
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    additional_terms = json.load(f)

                # Validate structure
                if isinstance(additional_terms, dict):
                    # Validate each term entry
                    valid_terms = {}
                    for term, info in additional_terms.items():
                        if isinstance(term, str) and isinstance(info, dict):
                            # Ensure required fields exist
                            if not info.get('romaji'):
                                info['romaji'] = ''
                            if not info.get('translation'):
                                info['translation'] = ''
                            if not info.get('category'):
                                info['category'] = 'martial_arts'
                            valid_terms[term] = info
                        else:
                            logger.warning(f"Invalid term entry: {term} -> {info}")

                    martial_arts_dict.update(valid_terms)
                    logger.info(f"Loaded {len(valid_terms)} additional martial arts terms")
                else:
                    logger.warning("martial_arts_terms.json does not contain a dictionary")
            else:
                logger.debug(f"Martial arts dictionary file not found: {dict_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in martial arts dictionary: {e}")
        except Exception as e:
            logger.debug(f"Could not load additional martial arts dictionary: {e}")

        # Add fallback terms if dictionary is empty
        if not martial_arts_dict:
            logger.warning("No martial arts terms loaded, using fallback dictionary")
            martial_arts_dict = {
                '武道': {'romaji': 'budo', 'translation': 'martial way', 'category': 'philosophy'},
                '武術': {'romaji': 'bujutsu', 'translation': 'martial art', 'category': 'general'},
                '空手': {'romaji': 'karate', 'translation': 'empty hand', 'category': 'art'},
                '柔道': {'romaji': 'judo', 'translation': 'gentle way', 'category': 'art'},
                '剣道': {'romaji': 'kendo', 'translation': 'way of sword', 'category': 'art'},
                '合気道': {'romaji': 'aikido', 'translation': 'way of harmony', 'category': 'art'},
                '道場': {'romaji': 'dojo', 'translation': 'training hall', 'category': 'place'},
                '先生': {'romaji': 'sensei', 'translation': 'teacher', 'category': 'person'},
                '型': {'romaji': 'kata', 'translation': 'form', 'category': 'technique'},
                '気': {'romaji': 'ki', 'translation': 'spirit/energy', 'category': 'concept'}
            }

        logger.info(f"Martial arts dictionary loaded with {len(martial_arts_dict)} terms")
        return martial_arts_dict


    def process_text(self, text: str) -> JapaneseProcessingResult:
        """
        Process Japanese text with full analysis.

        Args:
            text: Input text containing Japanese

        Returns:
            JapaneseProcessingResult with comprehensive analysis
        """
        start_time = datetime.now()

        try:
            logger.info("Starting Japanese text processing")

            # INPUT VALIDATION - NEW
            if not isinstance(text, str):
                raise TypeError(f"Input must be string, got {type(text)}")

            if not text or not text.strip():
                logger.warning("Empty or whitespace-only input text")
                return self._create_empty_result(text, start_time)

            # Validate UTF-8 encoding
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError as e:
                logger.error(f"Invalid UTF-8 encoding in input text: {e}")
                raise ValueError(f"Input text contains invalid UTF-8 characters: {e}")

            # Check for extremely long input (potential DoS)
            if len(text) > 100000:  # 100KB limit
                logger.warning(f"Very long input text: {len(text)} characters")
                # Truncate with warning rather than failing
                text = text[:100000]
                logger.warning("Input text truncated to 100,000 characters")

            # Clean the input text
            cleaned_text, cleaning_stats = self.text_cleaner.clean_text(text)

            # Detect Japanese segments
            japanese_segments = self._extract_japanese_segments(cleaned_text)

            if not japanese_segments:
                logger.info("No Japanese text segments found")
                return self._create_empty_result(text, start_time)

            # Process each segment
            processed_segments = []
            for segment_text, start_pos, end_pos in japanese_segments:
                segment = self._process_japanese_segment(segment_text, start_pos, end_pos)
                processed_segments.append(segment)

            # Generate overall romanization and translation
            overall_romaji = self._generate_overall_romaji(processed_segments)
            overall_translation = self._generate_overall_translation(processed_segments)

            # Analyze language composition
            language_analysis = self._analyze_language_composition(cleaned_text, processed_segments)

            # Extract martial arts terms
            martial_arts_terms = self._extract_martial_arts_terms(cleaned_text)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(processed_segments)

            # Create processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_metadata = {
                'processing_date': datetime.now().isoformat(),
                'mecab_available': MECAB_AVAILABLE,
                'pykakasi_available': PYKAKASI_AVAILABLE,
                'argos_available': ARGOS_AVAILABLE,
                'romanization_system': self.config.get('romanization_system', 'hepburn'),
                'translation_enabled': self.config.get('enable_translation', True),
                'cleaning_stats': cleaning_stats.to_dict(),
                'segments_processed': len(processed_segments),
                'martial_arts_terms_found': len(martial_arts_terms)
            }

            result = JapaneseProcessingResult(
                original_text=text,
                segments=processed_segments,
                overall_romaji=overall_romaji,
                overall_translation=overall_translation,
                language_analysis=language_analysis,
                martial_arts_terms=martial_arts_terms,
                processing_metadata=processing_metadata,
                confidence_score=confidence_score,
                processing_time=processing_time
            )

            logger.info(f"Japanese text processing completed in {processing_time:.2f}s")
            logger.info(
                f"Processed {len(processed_segments)} segments with {len(martial_arts_terms)} martial arts terms")

            return result

        except Exception as e:
            logger.error(f"Japanese text processing failed: {e}")
            raise

    def _extract_japanese_segments(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract Japanese text segments from mixed text.

        Args:
            text: Input text

        Returns:
            List of (segment_text, start_pos, end_pos) tuples
        """
        segments = []

        for match in self.japanese_pattern.finditer(text):
            segment_text = match.group()
            start_pos = match.start()
            end_pos = match.end()

            # Skip very short segments (likely noise)
            if len(segment_text) >= 1:
                segments.append((segment_text, start_pos, end_pos))

        logger.debug(f"Extracted {len(segments)} Japanese segments")
        return segments

    def _process_japanese_segment(self, segment_text: str, start_pos: int, end_pos: int) -> JapaneseTextSegment:
        """
        Process a single Japanese text segment.

        Args:
            segment_text: Japanese text segment
            start_pos: Start position in original text
            end_pos: End position in original text

        Returns:
            Processed JapaneseTextSegment
        """
        # INPUT VALIDATION - NEW
        if not segment_text or not isinstance(segment_text, str):
            logger.warning(f"Invalid segment text: {segment_text}")
            return JapaneseTextSegment(
                original_text=segment_text or "",
                confidence=0.0,
                text_type="invalid",
                start_pos=start_pos,
                end_pos=end_pos
            )

        try:
            # Determine text type
            text_type = self._classify_japanese_text_type(segment_text)

            # Generate romanization with error handling
            romaji = None
            try:
                romaji = self._romanize_text(segment_text)
            except Exception as e:
                logger.debug(f"Romanization failed for '{segment_text}': {e}")

            # Perform morphological analysis with error handling
            morphology = None
            try:
                morphology = self._analyze_morphology(segment_text)
            except Exception as e:
                logger.debug(f"Morphological analysis failed for '{segment_text}': {e}")

            # Extract readings and POS tags
            reading = None
            pos_tags = []
            if morphology:
                try:
                    reading = self._extract_reading_from_morphology(morphology)
                    pos_tags = self._extract_pos_tags_from_morphology(morphology)
                except Exception as e:
                    logger.debug(f"Reading extraction failed for '{segment_text}': {e}")

            # Generate translation with error handling
            translation = None
            try:
                translation = self._translate_text(segment_text)
            except Exception as e:
                logger.debug(f"Translation failed for '{segment_text}': {e}")

            # Calculate confidence
            confidence = self._calculate_segment_confidence(segment_text, romaji, translation, morphology)

            return JapaneseTextSegment(
                original_text=segment_text,
                romaji=romaji,
                translation=translation,
                morphology=morphology,
                reading=reading,
                pos_tags=pos_tags,
                confidence=confidence,
                text_type=text_type,
                start_pos=start_pos,
                end_pos=end_pos
            )

        except Exception as e:
            # STANDARDIZED ERROR HANDLING - Log and return minimal segment
            logger.warning(f"Failed to process segment '{segment_text}': {e}")
            return JapaneseTextSegment(
                original_text=segment_text,
                confidence=0.0,
                text_type="error",
                start_pos=start_pos,
                end_pos=end_pos
            )


    def _classify_japanese_text_type(self, text: str) -> str:
        """
        Classify the type of Japanese text.

        Args:
            text: Japanese text

        Returns:
            Text type: "hiragana", "katakana", "kanji", "mixed"
        """
        hiragana_count = len(self.hiragana_pattern.findall(text))
        katakana_count = len(self.katakana_pattern.findall(text))
        kanji_count = len(self.kanji_pattern.findall(text))

        total_japanese = hiragana_count + katakana_count + kanji_count

        if total_japanese == 0:
            return "unknown"

        # Determine predominant type
        if hiragana_count > katakana_count and hiragana_count > kanji_count:
            return "hiragana" if katakana_count == 0 and kanji_count == 0 else "mixed"
        elif katakana_count > hiragana_count and katakana_count > kanji_count:
            return "katakana" if hiragana_count == 0 and kanji_count == 0 else "mixed"
        elif kanji_count > hiragana_count and kanji_count > katakana_count:
            return "kanji" if hiragana_count == 0 and katakana_count == 0 else "mixed"
        else:
            return "mixed"

    def _romanize_text(self, text: str) -> Optional[str]:
        """
        Convert Japanese text to romaji.

        Args:
            text: Japanese text

        Returns:
            Romanized text or None if failed
        """
        try:
            # Try pykakasi first
            if self.kakasi_converter:
                result = self.kakasi_converter.do(text)
                if result and result.strip():
                    return result.strip()

            # Fallback to simple character mapping
            return self._simple_romanization(text)

        except Exception as e:
            logger.debug(f"Romanization failed for '{text}': {e}")
            return None

    def _simple_romanization(self, text: str) -> str:
        """
        Simple romanization fallback using character mapping.

        Args:
            text: Japanese text

        Returns:
            Romanized text
        """
        # Basic hiragana to romaji mapping - CORRECTED UTF-8 ENCODING
        hiragana_map = {
            # Basic vowels
            'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',

            # K sounds
            'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
            'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',

            # S sounds
            'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
            'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',

            # T sounds
            'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
            'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',

            # N sounds
            'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',

            # H sounds
            'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
            'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
            'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',

            # M sounds
            'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',

            # Y sounds
            'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',

            # R sounds
            'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',

            # W sounds and N
            'わ': 'wa', 'ゐ': 'wi', 'ゑ': 'we', 'を': 'wo', 'ん': 'n',

            # Special characters
            'ー': '-',  # Long vowel mark
            'っ': '',  # Small tsu (gemination) - handled specially

            # Small ya, yu, yo
            'ゃ': 'ya', 'ゅ': 'yu', 'ょ': 'yo',

            # Additional combinations
            'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
            'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
            'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
            'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
            'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
            'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
            'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
            'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
            'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
            'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
            'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo'
        }

        # Katakana to romaji mapping - CORRECTED UTF-8 ENCODING
        katakana_map = {
            # Basic vowels
            'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',

            # K sounds
            'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
            'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',

            # S sounds
            'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
            'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',

            # T sounds
            'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
            'ダ': 'da', 'ヂ': 'ji', 'ヅ': 'zu', 'デ': 'de', 'ド': 'do',

            # N sounds
            'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',

            # H sounds
            'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
            'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
            'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',

            # M sounds
            'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',

            # Y sounds
            'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',

            # R sounds
            'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',

            # W sounds and N
            'ワ': 'wa', 'ヰ': 'wi', 'ヱ': 'we', 'ヲ': 'wo', 'ン': 'n',

            # Special characters
            'ー': '-',  # Long vowel mark
            'ッ': '',  # Small tsu (gemination)

            # Small ya, yu, yo
            'ャ': 'ya', 'ュ': 'yu', 'ョ': 'yo',

            # Extended katakana for foreign words
            'ヴ': 'vu',
            'ファ': 'fa', 'フィ': 'fi', 'フェ': 'fe', 'フォ': 'fo',
            'ティ': 'ti', 'ディ': 'di',
            'トゥ': 'tu', 'ドゥ': 'du',
            'ウィ': 'wi', 'ウェ': 'we', 'ウォ': 'wo',

            # Combinations
            'キャ': 'kya', 'キュ': 'kyu', 'キョ': 'kyo',
            'シャ': 'sha', 'シュ': 'shu', 'ショ': 'sho',
            'チャ': 'cha', 'チュ': 'chu', 'チョ': 'cho',
            'ニャ': 'nya', 'ニュ': 'nyu', 'ニョ': 'nyo',
            'ヒャ': 'hya', 'ヒュ': 'hyu', 'ヒョ': 'hyo',
            'ミャ': 'mya', 'ミュ': 'myu', 'ミョ': 'myo',
            'リャ': 'rya', 'リュ': 'ryu', 'リョ': 'ryo',
            'ギャ': 'gya', 'ギュ': 'gyu', 'ギョ': 'gyo',
            'ジャ': 'ja', 'ジュ': 'ju', 'ジョ': 'jo',
            'ビャ': 'bya', 'ビュ': 'byu', 'ビョ': 'byo',
            'ピャ': 'pya', 'ピュ': 'pyu', 'ピョ': 'pyo'
        }

        # Combine mappings
        char_map = {**hiragana_map, **katakana_map}

        # Add martial arts specific kanji with CORRECTED UTF-8 ENCODING
        char_map.update({
            '武': 'bu',  # martial, military
            '道': 'dō',  # way, path
            '術': 'jutsu',  # art, technique
            '空': 'kara',  # empty, sky
            '手': 'te',  # hand
            '柔': 'jū',  # gentle, soft
            '剣': 'ken',  # sword
            '合': 'ai',  # together, harmony
            '気': 'ki',  # spirit, energy
            '型': 'kata',  # form, pattern
            '組': 'kumi',  # group, set
            '先': 'sen',  # before, previous
            '生': 'sei',  # life, student
            '師': 'shi',  # teacher, master
            '範': 'han',  # example, model
            '段': 'dan',  # step, level
            '級': 'kyū',  # class, grade
            '帯': 'obi',  # belt, sash
            '礼': 'rei',  # courtesy, bow
            '心': 'kokoro',  # heart, mind
            '和': 'wa',  # harmony, peace
            '流': 'ryū',  # style, school
            '会': 'kai',  # meeting, association
            '館': 'kan',  # building, hall
            '場': 'ba',  # place, field
            '真': 'shin',  # true, real
            '正': 'sei',  # correct, proper
            '古': 'ko',  # old, ancient
            '新': 'shin',  # new
            '大': 'dai',  # big, great
            '小': 'shō',  # small, little
            '中': 'chū',  # middle, center
            '上': 'jō',  # upper, above
            '下': 'ge',  # lower, below
            '前': 'mae',  # front, before
            '後': 'ato',  # after, behind
            '左': 'hidari',  # left
            '右': 'migi',  # right
            '内': 'uchi',  # inside, inner
            '外': 'soto',  # outside, outer
            '自': 'ji',  # self
            '他': 'ta',  # other
            '一': 'ichi',  # one
            '二': 'ni',  # two
            '三': 'san',  # three
            '四': 'shi',  # four
            '五': 'go',  # five
            '六': 'roku',  # six
            '七': 'shichi',  # seven
            '八': 'hachi',  # eight
            '九': 'kyū',  # nine
            '十': 'jū'  # ten
        })

        # Process text character by character
        result = ''
        i = 0
        while i < len(text):
            # Check for two-character combinations first
            if i < len(text) - 1:
                two_char = text[i:i + 2]
                if two_char in char_map:
                    result += char_map[two_char]
                    i += 2
                    continue

            # Check single character
            char = text[i]
            if char in char_map:
                # Handle small tsu (っ/ッ) - doubles the next consonant
                if char in ['っ', 'ッ'] and i < len(text) - 1:
                    next_char = text[i + 1]
                    next_romaji = char_map.get(next_char, next_char)
                    if next_romaji and next_romaji[0].isalpha():
                        result += next_romaji[0]  # Double the consonant
                else:
                    result += char_map[char]
            elif char.isascii():
                result += char
            else:
                result += char  # Keep unknown characters as-is

            i += 1

        return result

    def _analyze_morphology(self, text: str) -> Optional[List[Dict]]:
        """
        Perform morphological analysis using MeCab.

        Args:
            text: Japanese text

        Returns:
            List of morphological analysis results or None
        """
        if not self.mecab_tagger:
            return None

        try:
            result = self.mecab_tagger.parse(text)
            if not result:
                return None

            morphology = []
            lines = result.strip().split('\n')

            for line in lines:
                if line == 'EOS' or not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')

                    morph_info = {
                        'surface': surface,
                        'pos': features[0] if len(features) > 0 else '',
                        'pos_detail1': features[1] if len(features) > 1 else '',
                        'pos_detail2': features[2] if len(features) > 2 else '',
                        'pos_detail3': features[3] if len(features) > 3 else '',
                        'inflection': features[4] if len(features) > 4 else '',
                        'conjugation': features[5] if len(features) > 5 else '',
                        'base_form': features[6] if len(features) > 6 else surface,
                        'reading': features[7] if len(features) > 7 else '',
                        'pronunciation': features[8] if len(features) > 8 else ''
                    }

                    morphology.append(morph_info)

            return morphology if morphology else None

        except Exception as e:
            logger.debug(f"Morphological analysis failed for '{text}': {e}")
            return None

    def _extract_reading_from_morphology(self, morphology: List[Dict]) -> Optional[str]:
        """Extract reading information from morphological analysis."""
        if not morphology:
            return None

        readings = []
        for morph in morphology:
            reading = morph.get('reading', '') or morph.get('pronunciation', '')
            if reading and reading != '*':
                readings.append(reading)
            else:
                readings.append(morph.get('surface', ''))

        return ''.join(readings) if readings else None

    def _extract_pos_tags_from_morphology(self, morphology: List[Dict]) -> List[str]:
        """Extract part-of-speech tags from morphological analysis."""
        if not morphology:
            return []

        pos_tags = []
        for morph in morphology:
            pos = morph.get('pos', '')
            if pos:
                pos_tags.append(pos)

        return pos_tags

    def _translate_text(self, text: str) -> Optional[str]:
        """
        Translate Japanese text to target language.

        Args:
            text: Japanese text

        Returns:
            Translated text or None if failed
        """
        if not self.config.get('enable_translation', True):
            return None

        try:
            # Check if it's a known martial arts term first
            if text in self.martial_arts_dict:
                return self.martial_arts_dict[text].get('translation', None)

            # Try Argos Translate
            if ARGOS_AVAILABLE:
                translation = argostranslate.translate.translate(text, "ja", "en")
                if translation and translation.strip() and translation != text:
                    return translation.strip()

            # Fallback: simple word lookup for common terms
            return self._simple_translation_lookup(text)

        except Exception as e:
            logger.debug(f"Translation failed for '{text}': {e}")
            return None

    def _simple_translation_lookup(self, text: str) -> Optional[str]:
        """Simple translation lookup for common terms."""
        # EXPANDED: Basic translation dictionary for martial arts terms
        translations = {
            # Core martial arts terms
            '武道': 'martial way',
            '武術': 'martial art',
            '武芸': 'martial arts',
            '武者': 'warrior',
            '武士': 'samurai',

            # Specific arts
            '空手': 'karate',
            '空手道': 'karate-do',
            '柔道': 'judo',
            '剣道': 'kendo',
            '合気道': 'aikido',
            '少林寺拳法': 'shorinji kempo',
            '弓道': 'kyudo',
            '相撲': 'sumo',
            '忍術': 'ninjutsu',
            '古武道': 'koryu budo',
            '古武術': 'koryu bujutsu',

            # Training terms
            '型': 'form',
            '形': 'kata',
            '組手': 'sparring',
            '乱取り': 'randori',
            '試合': 'match',
            '稽古': 'practice',
            '修行': 'training',
            '鍛錬': 'discipline',

            # Places and people
            '道場': 'dojo',
            '武道館': 'budokan',
            '先生': 'teacher',
            '師範': 'master instructor',
            '師匠': 'master',
            '弟子': 'student',
            '門下生': 'disciple',
            '先輩': 'senior',
            '後輩': 'junior',

            # Ranks and equipment
            '段': 'dan',
            '級': 'kyu',
            '帯': 'belt',
            '黒帯': 'black belt',
            '白帯': 'white belt',
            '道着': 'dogi',
            '道衣': 'training uniform',

            # Philosophy and concepts
            '礼': 'bow',
            '礼儀': 'etiquette',
            '心': 'mind/heart',
            '精神': 'spirit',
            '気': 'spirit/energy',
            '気合': 'fighting spirit',
            '和': 'harmony',
            '道': 'way/path',
            '真': 'truth',
            '正': 'correct',
            '美': 'beauty',
            '技': 'technique',
            '力': 'power/strength',
            '速': 'speed',

            # Weapons
            '刀': 'sword',
            '剣': 'sword',
            '太刀': 'tachi',
            '脇差': 'wakizashi',
            '短刀': 'tanto',
            '槍': 'spear',
            '薙刀': 'naginata',
            '弓': 'bow',
            '矢': 'arrow',
            '杖': 'staff',
            '棒': 'bo staff',

            # Techniques
            '突き': 'thrust',
            '打ち': 'strike',
            '蹴り': 'kick',
            '投げ': 'throw',
            '極め': 'joint lock',
            '絞め': 'choke',
            '受け': 'block',
            '避け': 'evasion',

            # Directions and positions
            '前': 'front',
            '後': 'back',
            '左': 'left',
            '右': 'right',
            '上': 'up/upper',
            '下': 'down/lower',
            '中': 'middle',
            '内': 'inside',
            '外': 'outside',

            # Common words
            '一': 'one',
            '二': 'two',
            '三': 'three',
            '四': 'four',
            '五': 'five',
            '始め': 'begin',
            '終わり': 'end',
            '止め': 'stop',
            '間': 'space/timing',
            '時': 'time'
        }

        return translations.get(text, None)

    def _calculate_segment_confidence(self, text: str, romaji: Optional[str],
                                      translation: Optional[str], morphology: Optional[List[Dict]]) -> float:
        """Calculate confidence score for a text segment."""
        score = 0.0

        # Base score for having text
        if text:
            score += 0.2

        # Romaji available
        if romaji:
            score += 0.3

        # Translation available
        if translation:
            score += 0.3

        # Morphological analysis available
        if morphology:
            score += 0.2

        # Length bonus (longer text generally more reliable)
        if len(text) > 1:
            score += min(len(text) * 0.05, 0.2)

        # Known martial arts term bonus
        if text in self.martial_arts_dict:
            score += 0.1

        return min(1.0, score)

    def _generate_overall_romaji(self, segments: List[JapaneseTextSegment]) -> Optional[str]:
        """Generate overall romanization from all segments."""
        romaji_parts = []

        for segment in segments:
            if segment.romaji:
                romaji_parts.append(segment.romaji)
            else:
                # Fallback to simple romanization
                simple_romaji = self._simple_romanization(segment.original_text)
                if simple_romaji:
                    romaji_parts.append(simple_romaji)

        return ' '.join(romaji_parts) if romaji_parts else None

    def _generate_overall_translation(self, segments: List[JapaneseTextSegment]) -> Optional[str]:
        """Generate overall translation from all segments."""
        translation_parts = []

        for segment in segments:
            if segment.translation:
                translation_parts.append(segment.translation)
            else:
                # Keep original text if no translation available
                translation_parts.append(segment.original_text)

        return ' '.join(translation_parts) if translation_parts else None

    def _analyze_language_composition(self, text: str, segments: List[JapaneseTextSegment]) -> Dict[str, Any]:
        """Analyze the language composition of the text."""
        total_chars = len(text)
        japanese_chars = sum(len(segment.original_text) for segment in segments)

        # Count character types
        hiragana_count = len(self.hiragana_pattern.findall(text))
        katakana_count = len(self.katakana_pattern.findall(text))
        kanji_count = len(self.kanji_pattern.findall(text))

        # Calculate ratios
        japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
        english_ratio = 1 - japanese_ratio

        return {
            'total_characters': total_chars,
            'japanese_characters': japanese_chars,
            'english_characters': total_chars - japanese_chars,
            'japanese_ratio': japanese_ratio,
            'english_ratio': english_ratio,
            'character_types': {
                'hiragana': hiragana_count,
                'katakana': katakana_count,
                'kanji': kanji_count
            },
            'segments_analyzed': len(segments),
            'dominant_language': 'japanese' if japanese_ratio > 0.5 else 'english',
            'is_mixed_language': 0.1 < japanese_ratio < 0.9
        }

    def _extract_martial_arts_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract martial arts terminology from text."""
        found_terms = []

        # Check against dictionary
        for term, info in self.martial_arts_dict.items():
            if term in text:
                # Find all occurrences
                start = 0
                while True:
                    pos = text.find(term, start)
                    if pos == -1:
                        break

                    term_info = {
                        'term': term,
                        'position': pos,
                        'length': len(term),
                        'romaji': info.get('romaji', ''),
                        'translation': info.get('translation', ''),
                        'category': info.get('category', 'martial_arts'),
                        'context': text[max(0, pos - 10):pos + len(term) + 10].strip()
                    }
                    found_terms.append(term_info)
                    start = pos + 1

        # Sort by position
        found_terms.sort(key=lambda x: x['position'])

        logger.debug(f"Found {len(found_terms)} martial arts terms")
        return found_terms

    def _calculate_confidence_score(self, segments: List[JapaneseTextSegment]) -> float:
        """Calculate overall confidence score for processing."""
        if not segments:
            return 0.0

        # Average segment confidence
        segment_confidences = [seg.confidence for seg in segments]
        avg_confidence = sum(segment_confidences) / len(segment_confidences)

        # Bonus for having multiple successful segments
        segment_bonus = min(len(segments) * 0.05, 0.2)

        # Bonus for having romanization
        romaji_bonus = 0.1 if any(seg.romaji for seg in segments) else 0

        # Bonus for having translations
        translation_bonus = 0.1 if any(seg.translation for seg in segments) else 0

        # Bonus for morphological analysis
        morphology_bonus = 0.1 if any(seg.morphology for seg in segments) else 0

        total_confidence = avg_confidence + segment_bonus + romaji_bonus + translation_bonus + morphology_bonus

        return min(1.0, total_confidence)

    def _create_empty_result(self, text: str, start_time: datetime) -> JapaneseProcessingResult:
        """Create an empty result when no Japanese text is found."""
        processing_time = (datetime.now() - start_time).total_seconds()

        return JapaneseProcessingResult(
            original_text=text,
            segments=[],
            overall_romaji=None,
            overall_translation=None,
            language_analysis={
                'total_characters': len(text),
                'japanese_characters': 0,
                'english_characters': len(text),
                'japanese_ratio': 0.0,
                'english_ratio': 1.0,
                'character_types': {'hiragana': 0, 'katakana': 0, 'kanji': 0},
                'segments_analyzed': 0,
                'dominant_language': 'english',
                'is_mixed_language': False
            },
            martial_arts_terms=[],
            processing_metadata={
                'processing_date': datetime.now().isoformat(),
                'mecab_available': MECAB_AVAILABLE,
                'pykakasi_available': PYKAKASI_AVAILABLE,
                'argos_available': ARGOS_AVAILABLE,
                'romanization_system': self.config.get('romanization_system', 'hepburn'),
                'translation_enabled': self.config.get('enable_translation', True),
                'segments_processed': 0,
                'martial_arts_terms_found': 0
            },
            confidence_score=0.0,
            processing_time=processing_time
        )

    def get_japanese_markup(self, text: str, result: JapaneseProcessingResult) -> str:
        """
        Generate HTML markup for Japanese text with hover tooltips.

        Args:
            text: Original text
            result: Processing result

        Returns:
            HTML markup with Japanese text highlighting
        """
        if not result.segments:
            return text

        # Sort segments by position to process in order
        sorted_segments = sorted(result.segments, key=lambda s: s.start_pos)

        marked_text = text
        offset = 0  # Track offset due to inserted HTML tags

        for segment in sorted_segments:
            # Adjust positions for previously inserted HTML
            start_pos = segment.start_pos + offset
            end_pos = segment.end_pos + offset

            # Create tooltip content
            tooltip_parts = []
            if segment.romaji:
                tooltip_parts.append(f"Romaji: {segment.romaji}")
            if segment.translation:
                tooltip_parts.append(f"Translation: {segment.translation}")
            if segment.reading and segment.reading != segment.romaji:
                tooltip_parts.append(f"Reading: {segment.reading}")

            tooltip_text = " | ".join(tooltip_parts) if tooltip_parts else "Japanese text"

            # Create HTML markup
            css_class = f"japanese-text japanese-{segment.text_type}"
            html_markup = f'<span class="{css_class}" title="{tooltip_text}" data-confidence="{segment.confidence:.2f}">{segment.original_text}</span>'

            # Replace in text
            marked_text = marked_text[:start_pos] + html_markup + marked_text[end_pos:]

            # Update offset
            offset += len(html_markup) - len(segment.original_text)

        return marked_text

    def romanize_text_simple(self, text: str) -> str:
        """
        Simple romanization for quick processing.

        Args:
            text: Japanese text

        Returns:
            Romanized text
        """
        try:
            if self.kakasi_converter:
                return self.kakasi_converter.do(text)
            else:
                return self._simple_romanization(text)
        except Exception as e:
            logger.debug(f"Simple romanization failed: {e}")
            return text

    def is_japanese_text(self, text: str) -> bool:
        """
        Check if text contains Japanese characters.

        Args:
            text: Text to check

        Returns:
            True if text contains Japanese characters
        """
        return bool(self.japanese_pattern.search(text))

    def extract_japanese_only(self, text: str) -> str:
        """
        Extract only Japanese characters from text.

        Args:
            text: Mixed text

        Returns:
            Japanese characters only
        """
        japanese_chars = self.japanese_pattern.findall(text)
        return ''.join(japanese_chars)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about Japanese processing capabilities."""
        return {
            'mecab_available': MECAB_AVAILABLE,
            'pykakasi_available': PYKAKASI_AVAILABLE,
            'argos_translate_available': ARGOS_AVAILABLE,
            'romanization_system': self.config.get('romanization_system', 'hepburn'),
            'translation_enabled': self.config.get('enable_translation', True),
            'martial_arts_terms_count': len(self.martial_arts_dict),
            'supported_features': {
                'romanization': PYKAKASI_AVAILABLE or True,  # Fallback available
                'morphological_analysis': MECAB_AVAILABLE,
                'translation': ARGOS_AVAILABLE,
                'martial_arts_terminology': True,
                'text_classification': True,
                'html_markup': True
            }
        }


# Utility functions for easy access
def process_japanese_text(text: str) -> JapaneseProcessingResult:
    """
    Convenient function to process Japanese text.

    Args:
        text: Input text

    Returns:
        JapaneseProcessingResult
    """
    processor = JapaneseProcessor()
    return processor.process_text(text)


def romanize_japanese(text: str) -> str:
    """
    Convenient function to romanize Japanese text.

    Args:
        text: Japanese text

    Returns:
        Romanized text
    """
    processor = JapaneseProcessor()
    return processor.romanize_text_simple(text)


def is_japanese(text: str) -> bool:
    """
    Convenient function to check if text contains Japanese.

    Args:
        text: Text to check

    Returns:
        True if contains Japanese characters
    """
    processor = JapaneseProcessor()
    return processor.is_japanese_text(text)


if __name__ == "__main__":
    # Test the Japanese processor
    test_text = "武道の研究 - The study of martial arts. This text contains 空手道 and other terms."

    processor = JapaneseProcessor()
    result = processor.process_text(test_text)

    print("Japanese Processing Test Results:")
    print(f"Original text: {result.original_text}")
    print(f"Segments found: {len(result.segments)}")

    for i, segment in enumerate(result.segments):
        print(f"\nSegment {i + 1}:")
        print(f"  Text: {segment.original_text}")
        print(f"  Romaji: {segment.romaji}")
        print(f"  Translation: {segment.translation}")
        print(f"  Type: {segment.text_type}")
        print(f"  Confidence: {segment.confidence:.2f}")

    print(f"\nOverall romaji: {result.overall_romaji}")
    print(f"Overall translation: {result.overall_translation}")
    print(f"Martial arts terms found: {len(result.martial_arts_terms)}")
    print(f"Processing confidence: {result.confidence_score:.2f}")
    print(f"Processing time: {result.processing_time:.3f}s")

    # Test HTML markup
    html_markup = processor.get_japanese_markup(test_text, result)
    print(f"\nHTML markup:\n{html_markup}")