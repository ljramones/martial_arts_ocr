from pathlib import Path

from martial_arts_ocr.ocr.postprocessor import OCRPostProcessor
from martial_arts_ocr.pipeline.adapters import document_result_from_ocr_output
from utils import TextCleaner

from fixtures.ocr_text_samples import MIXED_MARTIAL_ARTS_WORDS


PRESERVATION_TOKENS = [
    "武道",
    "柔術",
    "koryū",
    "budō",
    "Daitō-ryū",
    "ō",
    "ū",
    "—",
    "・",
    "「",
    "」",
]


def _run_cleanup_chain(text: str) -> str:
    postprocessed = OCRPostProcessor(domain="general").clean_text(
        text,
        confidence=0.95,
        boxes=[],
    )
    cleaned, _stats = TextCleaner().clean_text(postprocessed)
    return cleaned


def test_cleanup_chain_preserves_japanese_macrons_and_punctuation():
    text = "武道 と 柔術\nDaitō-ryū Aiki-jūjutsu\n「柔術」・budō — koryū"

    cleaned = _run_cleanup_chain(text)

    for token in PRESERVATION_TOKENS:
        assert token in cleaned


def test_cleanup_chain_does_not_rewrite_japanese_long_vowel_mark_to_kanji_one():
    text = "カラーテ と スーパー柔術"

    cleaned = _run_cleanup_chain(text)

    assert "ー" in cleaned
    assert "一" not in cleaned


def test_cleanup_chain_preserves_useful_line_breaks_and_normalizes_excess_whitespace():
    text = "  武道   と   柔術  \n\n\nDaitō-ryū   Aiki-jūjutsu\n「柔術」・budō   —   koryū  "

    cleaned = _run_cleanup_chain(text)

    assert cleaned.splitlines() == [
        "武道 と 柔術",
        "",
        "Daitō-ryū Aiki-jūjutsu",
        "「柔術」・budō — koryū",
    ]


def test_cleanup_chain_feeds_readable_text_hierarchy_and_serialization():
    cleaned = _run_cleanup_chain(
        "武道 とは何か。\nDaitō-ryū Aiki-jūjutsu\n「柔術」・budō — koryū"
    )

    result = document_result_from_ocr_output(
        {
            "text": cleaned,
            "confidence": 0.92,
            "words": MIXED_MARTIAL_ARTS_WORDS,
        },
        document_id=42,
        source_path=Path("synthetic_scan.jpg"),
    )
    page = result.pages[0]
    serialized_page = result.to_dict()["pages"][0]

    word_regions = [
        region for region in page.text_regions
        if region.metadata.get("ocr_level") == "word"
    ]
    line_regions = [
        region for region in page.text_regions
        if region.metadata.get("ocr_level") == "line"
    ]

    assert word_regions
    assert line_regions
    assert all(region.metadata["source"] == "ocr_engine" for region in word_regions)
    assert all(region.metadata["source"] == "ocr_normalization" for region in line_regions)
    assert page.metadata["readable_text"] == (
        "武道 とは何か。\n"
        "Daitō-ryū Aiki-jūjutsu\n"
        "「柔術」・budō — koryū"
    )
    assert page.metadata["ocr_word_count"] == len(MIXED_MARTIAL_ARTS_WORDS)
    assert page.metadata["ocr_line_count"] == 3

    assert serialized_page["raw_text"] == cleaned
    assert serialized_page["metadata"]["readable_text"] == page.metadata["readable_text"]
    assert serialized_page["metadata"]["ocr_word_count"] == len(MIXED_MARTIAL_ARTS_WORDS)
    assert serialized_page["metadata"]["ocr_line_count"] == 3
    assert serialized_page["metadata"]["ocr_text_boxes"]
    assert serialized_page["text_regions"]


def test_cleanup_chain_does_not_ascii_normalize_macrons_or_strip_japanese_punctuation():
    cleaned = _run_cleanup_chain("koryū budō Daitō-ryū 「柔術」・武道")

    assert "koryū" in cleaned
    assert "budō" in cleaned
    assert "Daitō-ryū" in cleaned
    assert "koryu" not in cleaned
    assert "budo" not in cleaned
    assert "Daito-ryu" not in cleaned
    assert "「柔術」・武道" in cleaned
