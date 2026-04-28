def test_legacy_root_imports_remain_available():
    import config
    import database
    import models
    from processors.content_extractor import ContentExtractor
    from processors.japanese_processor import JapaneseProcessor
    from processors.ocr_processor import OCRProcessor
    from processors.page_reconstructor import PageReconstructor

    assert config.Config
    assert database.init_db
    assert models.Document
    assert OCRProcessor
    assert JapaneseProcessor
    assert ContentExtractor
    assert PageReconstructor


def test_package_native_imports_are_source_of_truth():
    from martial_arts_ocr.config import Config
    from martial_arts_ocr.db.database import init_db
    from martial_arts_ocr.db.models import Document
    from martial_arts_ocr.imaging.content_extractor import ContentExtractor
    from martial_arts_ocr.japanese.processor import JapaneseProcessor
    from martial_arts_ocr.ocr.processor import OCRProcessor
    from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor

    assert Config
    assert init_db
    assert Document
    assert OCRProcessor
    assert JapaneseProcessor
    assert ContentExtractor
    assert PageReconstructor


def test_legacy_imports_reexport_package_objects():
    from processors.ocr_processor import OCRProcessor as LegacyOCRProcessor
    from martial_arts_ocr.ocr.processor import OCRProcessor as PackageOCRProcessor

    assert LegacyOCRProcessor is PackageOCRProcessor
    assert LegacyOCRProcessor.__module__ == "martial_arts_ocr.ocr.processor"

