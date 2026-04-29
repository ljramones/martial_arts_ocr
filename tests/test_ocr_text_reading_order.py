
from martial_arts_ocr.pipeline.document_models import BoundingBox, PageResult, TextRegion
from martial_arts_ocr.pipeline.text_normalization import (
    add_readable_lines_to_page,
    group_word_regions_into_lines,
)


def word(region_id, text, x, y, width=30, height=12):
    return TextRegion(
        region_id=region_id,
        text=text,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
        confidence=0.9,
        metadata={"source": "ocr_engine", "engine": "fake_test", "ocr_level": "word"},
    )


def test_normal_paragraph_lines_group_correctly():
    lines = group_word_regions_into_lines([
        word("w1", "Donn", 10, 10),
        word("w2", "Draeger", 48, 11, width=50),
        word("w3", "studied", 10, 30, width=46),
        word("w4", "budō", 62, 30, width=36),
    ])

    assert [line.text for line in lines] == ["Donn Draeger", "studied budō"]


def test_slight_y_jitter_on_same_visual_line_groups_together():
    lines = group_word_regions_into_lines([
        word("w1", "Daitō-ryū", 10, 10, width=70, height=14),
        word("w2", "Aiki", 86, 13, width=32, height=12),
        word("w3", "jūjutsu", 124, 9, width=60, height=15),
    ])

    assert len(lines) == 1
    assert lines[0].text == "Daitō-ryū Aiki jūjutsu"


def test_close_centers_do_not_merge_when_y_overlap_is_weak():
    lines = group_word_regions_into_lines([
        word("w1", "top", 10, 10, width=30, height=8),
        word("w2", "bottom", 10, 18, width=50, height=8),
    ], y_tolerance=2)

    assert [line.text for line in lines] == ["top", "bottom"]


def test_large_x_gap_is_preserved_and_marked_uncertain():
    lines = group_word_regions_into_lines([
        word("w1", "left", 10, 10, width=30),
        word("w2", "right", 220, 10, width=40),
    ])

    assert lines[0].text == "left  right"
    assert lines[0].metadata["reading_order_uncertain"] is True


def test_caption_line_below_figure_like_gap_remains_separate():
    lines = group_word_regions_into_lines([
        word("w1", "Fig.", 10, 10, width=24),
        word("w2", "5", 40, 10, width=8),
        word("w3", "Examples", 10, 52, width=64),
        word("w4", "of", 82, 52, width=16),
        word("w5", "horimono", 106, 52, width=70),
    ])

    assert [line.text for line in lines] == ["Fig. 5", "Examples of horimono"]


def test_list_like_text_preserves_line_breaks():
    page = add_readable_lines_to_page(PageResult(page_number=1, text_regions=[
        word("w1", "1.", 10, 10, width=12),
        word("w2", "BUJUTSU", 30, 10, width=70),
        word("w3", "2.", 10, 32, width=12),
        word("w4", "BUDO", 30, 32, width=45),
    ]))

    assert page.metadata["readable_text"] == "1. BUJUTSU\n2. BUDO"
    assert page.metadata["ocr_line_count"] == 2


def test_mixed_english_japanese_macrons_and_punctuation_remain_unchanged():
    lines = group_word_regions_into_lines([
        word("w1", "武道", 10, 10, width=32),
        word("w2", "柔術", 48, 10, width=32),
        word("w3", "koryū", 10, 30, width=45),
        word("w4", "budō", 62, 30, width=40),
        word("w5", "Daitō-ryū", 110, 30, width=80),
        word("w6", "ō", 10, 52, width=12),
        word("w7", "ū", 28, 52, width=12),
        word("w8", "—", 46, 52, width=12),
        word("w9", "・", 64, 52, width=12),
        word("w10", "「」", 82, 52, width=18),
    ])
    readable = "\n".join(line.text for line in lines)

    for token in ["武道", "柔術", "koryū", "budō", "Daitō-ryū", "ō", "ū", "—", "・", "「」"]:
        assert token in readable


def test_line_metadata_records_grouping_method():
    line = group_word_regions_into_lines([word("w1", "hello", 10, 10)])[0]

    assert line.metadata["line_grouping_method"] == "adaptive_center_overlap_v1"
    assert line.metadata["reading_order_uncertain"] is False
