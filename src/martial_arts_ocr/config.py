"""Compatibility access to the legacy root configuration module."""

from config import (  # noqa: F401
    Config,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    allowed_file,
    get_config,
    get_extracted_content_path,
    get_japanese_config,
    get_layout_config,
    get_ocr_config,
    get_processed_path,
    get_upload_path,
)

