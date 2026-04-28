from utils import LanguageDetector, TextCleaner


MIXED_TEXT = """Donn Draeger studied classical bujutsu.
武道とは何か。
Daitō-ryū Aiki-jūjutsu
カタカナー
「柔術」・budō — koryū
"""


def test_text_cleaner_preserves_japanese_macrons_and_martial_arts_punctuation():
    cleaned, stats = TextCleaner().clean_text(MIXED_TEXT)

    for token in ["武道", "柔術", "koryū", "budō", "Daitō-ryū", "ō", "ū", "—", "・", "「", "」", "カタカナー"]:
        assert token in cleaned
    assert "カタカナ一" not in cleaned
    assert stats.cleaned_length > 0


def test_text_cleaner_preserves_useful_line_breaks():
    text = "Line one\nLine two\n\n\nLine three"

    cleaned, _stats = TextCleaner().clean_text(text)

    assert "Line one\nLine two" in cleaned
    assert "\n\n" in cleaned
    assert "\n\n\n" not in cleaned


def test_language_detector_segments_mixed_english_and_japanese():
    segments = LanguageDetector().segment_by_language("Draeger 武道 koryū 柔術")

    assert any(segment.language == "ja" and segment.text == "武道" for segment in segments)
    assert any(segment.language == "ja" and segment.text == "柔術" for segment in segments)
    assert any("Draeger" in segment.text for segment in segments)
