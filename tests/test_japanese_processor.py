def test_japanese_processor_module_imports():
    from martial_arts_ocr.japanese.processor import JapaneseProcessingResult, JapaneseTextSegment

    assert JapaneseTextSegment
    assert JapaneseProcessingResult

