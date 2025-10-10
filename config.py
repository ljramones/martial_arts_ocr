"""
Configuration settings for Martial Arts OCR application.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
EXTRACTED_CONTENT_DIR = STATIC_DIR / "extracted_content"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, EXTRACTED_CONTENT_DIR]:
    directory.mkdir(exist_ok=True)


class Config:
    """Base configuration class."""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', 5000))

    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'}

    # Add this line:
    EXTRACTED_CONTENT_DIR = EXTRACTED_CONTENT_DIR  # Add as class attribute

    ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

    # Database settings
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'martial_arts_ocr.db'}"

    # OCR Engine Configuration
    OCR_ENGINES = {
        'tesseract': {
            'enabled': True,
            'languages': ['eng'],
            'config': '--psm 11 --oem 3',  # Use PSM 11 for sparse text
        },
        'easyocr': {
            'enabled': False,  # or True if you have it installed
            'languages': ['en'],
            'gpu': False,
            'confidence_threshold': 0.5
        }
    }

    # Image preprocessing settings
    IMAGE_PROCESSING = {
        'enhance_contrast': True,
        'denoise': True,
        'deskew': True,
        'resize_factor': 1.5,
        'min_image_size': (100, 100),
        'max_image_size': (2000, 2000),
        "illumination_correct": True,
        "morph_close": False,

        # For ocr_osd.py: Languages for the 0-vs-180 degree sniff
        'OCR_SNIFF_LANGUAGES': 'eng+jpn',

        # For denoise.py: Threshold for applying pre-boost sharpening
        'BLUR_PREBOOST_THRESHOLD': 180.0,

        # For geometry.py: Parameters for the small-angle deskew
        'DESKEW_HOUGH_THRESHOLD': 100,
        'DESKEW_HOUGH_MIN_LINE_LENGTH': 100,
        'DESKEW_HOUGH_MAX_LINE_GAP': 10,

        # For orientation.py: Weights for the coarse orientation score
        'ORIENTATION_PROJ_WEIGHT': 1.0,
        'ORIENTATION_HORIZ_WEIGHT': 0.8,

        # --- BINARIZE PARAMETERS ---

        # For binarize.py: Sauvola thresholding parameters
        'SAUVOLA_WINDOW': 25,
        'SAUVOLA_K': 0.2,

        # For binarize.py: Unsharp masking parameters
        'UNSHARP_STRENGTH': 1.5,
        'UNSHARP_SIGMA': 1.0,

        # --- DENOISE PARAMETERS ---
        # For denoise.py: Unsharp masking parameters for the pre-boost step
        'PREBOOST_UNSHARP_STRENGTH': 1.8,
        'PREBOOST_UNSHARP_BLUR_WEIGHT': -0.8,

        'SCRIPT_DETECTION_ENABLED': True,  # Auto-detect CJK vs Latin
        'MIXED_CONTENT_MODE': 'auto',  # 'text_only', 'mixed', 'auto'
        'CJK_SAUVOLA_K': 0.15,  # Gentler for complex strokes
        'IMAGE_REGION_PRESERVATION': True,  # Mask non-text areas

        # --- DEBUG PARAMETERS ---
        'DEBUG_DIR': 'debug_output',  # Assuming you have this or similar
        'DEBUG_FILE_PREFIX': '',  # Optional prefix for debug filenames
        'DEBUG_FILE_LIMIT': 100,  # Max number of debug files to save

        'ORIENT_CKPT_CONVNEXT': 'orientation_model/checkpoints/orient_convnext_tiny.pth',
        'ORIENT_CKPT_EFFNET': 'orientation_model/checkpoints/orient_effnetv2s.pth',  # or None
        'ORIENT_ENS_MARGIN': 0.55,

        # honor this in facade; you already set True in your file
        'DISABLE_HEURISTIC_FALLBACK': True,
    }

    # Layout detection settings
    LAYOUT_DETECTION = {
        'text_block_min_area': 1000,  # Minimum area for text blocks
        'image_block_min_area': 2500,  # Minimum area for image blocks
        'margin_threshold': 20,  # Pixels for margin detection
        'line_spacing_threshold': 15,  # Pixels for line spacing
    }

    # Japanese text processing
    JAPANESE_PROCESSING = {
        'mecab_dict': 'unidic',  # or 'ipadic'
        'romanization_system': 'hepburn',  # 'hepburn', 'kunrei', 'nihon'
        'enable_translation': True,
        'translation_engine': 'argos',  # 'argos' for offline
        'furigana_mode': 'auto',  # 'auto', 'always', 'never'
        'kanji_threshold': 2,  # Minimum kanji level for furigana
    }

    # Output format settings
    OUTPUT_FORMATS = {
        'html': {
            'enabled': True,
            'preserve_layout': True,
            'include_css': True,
            'responsive': True,
        },
        'markdown': {
            'enabled': True,
            'include_images': True,
            'footnotes_for_japanese': True,
        },
        'json': {
            'enabled': True,
            'include_metadata': True,
            'include_coordinates': True,
        }
    }

    # Web interface settings
    WEB_INTERFACE = {
        'items_per_page': 20,
        'thumbnail_size': (200, 300),
        'enable_zoom': True,
        'enable_fullscreen': True,
        'show_confidence_scores': True,
    }

    # Logging configuration
    LOGGING = {
        'level': 'INFO',
        'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
        'rotation': '1 week',
        'retention': '1 month',
        'file': str(BASE_DIR / 'logs' / 'martial_arts_ocr.log'),
    }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOGGING = {
        **Config.LOGGING,
        'level': 'DEBUG',
    }


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'please-set-a-real-secret-key'

    # More restrictive settings for production
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB max file size

    # Disable development features
    WEB_INTERFACE = {
        **Config.WEB_INTERFACE,
        'show_confidence_scores': False,
    }


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'  # In-memory database for tests

    # Faster processing for tests
    OCR_ENGINES = {
        'tesseract': {
            **Config.OCR_ENGINES['tesseract'],
            'timeout': 10,
        },
        'easyocr': {
            **Config.OCR_ENGINES['easyocr'],
            'enabled': False,  # Disable for faster tests
        }
    }


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}


def get_config(config_name: str = None) -> Config:
    """Get configuration class by name."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    return config_map.get(config_name, DevelopmentConfig)


# Utility functions for accessing config values
def get_ocr_config() -> Dict[str, Any]:
    """Get OCR engine configuration."""
    config = get_config()
    return config.OCR_ENGINES


def get_japanese_config() -> Dict[str, Any]:
    """Get Japanese processing configuration."""
    config = get_config()
    return config.JAPANESE_PROCESSING


def get_layout_config() -> Dict[str, Any]:
    """Get layout detection configuration."""
    config = get_config()
    return config.LAYOUT_DETECTION


# File type validation
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    config = get_config()
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS)


# Path utilities
def get_upload_path(filename: str) -> Path:
    """Get full path for uploaded file."""
    return UPLOAD_DIR / filename


def get_processed_path(filename: str) -> Path:
    """Get full path for processed file."""
    return PROCESSED_DIR / filename


def get_extracted_content_path(filename: str) -> Path:
    """Get full path for extracted content."""
    return EXTRACTED_CONTENT_DIR / filename
