"""
Configuration settings for Martial Arts OCR application.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MARTIAL_ARTS_OCR_DATA_DIR", BASE_DIR / "data"))
RUNTIME_DIR = Path(os.environ.get("MARTIAL_ARTS_OCR_RUNTIME_DIR", DATA_DIR / "runtime"))
UPLOAD_DIR = Path(os.environ.get("MARTIAL_ARTS_OCR_UPLOAD_DIR", RUNTIME_DIR / "uploads"))
PROCESSED_DIR = Path(os.environ.get("MARTIAL_ARTS_OCR_PROCESSED_DIR", RUNTIME_DIR / "processed"))
DB_DIR = Path(os.environ.get("MARTIAL_ARTS_OCR_DB_DIR", RUNTIME_DIR / "db"))
STATIC_DIR = BASE_DIR / "static"
EXTRACTED_CONTENT_DIR = STATIC_DIR / "extracted_content"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, DB_DIR, EXTRACTED_CONTENT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


def configure_runtime_paths(
    data_dir: str | Path | None = None,
    upload_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> None:
    """Update runtime paths without requiring module reloads."""
    global DATA_DIR, RUNTIME_DIR, UPLOAD_DIR, PROCESSED_DIR, DB_DIR

    if data_dir is not None:
        DATA_DIR = Path(data_dir)
        RUNTIME_DIR = DATA_DIR / "runtime"
    if upload_dir is not None:
        UPLOAD_DIR = Path(upload_dir)
    elif data_dir is not None:
        UPLOAD_DIR = RUNTIME_DIR / "uploads"
    if processed_dir is not None:
        PROCESSED_DIR = Path(processed_dir)
    elif data_dir is not None:
        PROCESSED_DIR = RUNTIME_DIR / "processed"
    if data_dir is not None:
        DB_DIR = RUNTIME_DIR / "db"

    for directory in [UPLOAD_DIR, PROCESSED_DIR, DB_DIR, EXTRACTED_CONTENT_DIR]:
        directory.mkdir(exist_ok=True, parents=True)

    Config.DATA_DIR = DATA_DIR
    Config.RUNTIME_DIR = RUNTIME_DIR
    Config.UPLOAD_FOLDER = str(UPLOAD_DIR)
    Config.DATABASE_URL = f"sqlite:///{DB_DIR / 'martial_arts_ocr.db'}"


class Config:
    """Base configuration class."""

    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    RUNTIME_DIR = RUNTIME_DIR
    STATIC_DIR = STATIC_DIR

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
    DATABASE_URL = f"sqlite:///{DB_DIR / 'martial_arts_ocr.db'}"

    # Review-mode extraction settings. Disabled by default because real-page
    # review found useful but broad/label-heavy crops on some pages.
    ENABLE_IMAGE_REGION_EXTRACTION = os.environ.get(
        "ENABLE_IMAGE_REGION_EXTRACTION",
        "false",
    ).lower() == "true"
    IMAGE_REGION_EXTRACTION_SAVE_CROPS = os.environ.get(
        "IMAGE_REGION_EXTRACTION_SAVE_CROPS",
        "true",
    ).lower() == "true"
    IMAGE_REGION_EXTRACTION_FAIL_ON_ERROR = os.environ.get(
        "IMAGE_REGION_EXTRACTION_FAIL_ON_ERROR",
        "false",
    ).lower() == "true"
    ENABLE_PADDLE_LAYOUT_FUSION = os.environ.get(
        "ENABLE_PADDLE_LAYOUT_FUSION",
        "false",
    ).lower() == "true"
    PADDLE_LAYOUT_MODEL_DIR = os.environ.get("PADDLE_LAYOUT_MODEL_DIR") or None

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

        # Text mask (Phase-2) parameters
        'TEXTMASK_BLACKHAT_K': 31,  # odd ≥ 15; bigger = smoother text extraction
        'TEXTMASK_MSER_DELTA': 5,
        'TEXTMASK_MSER_MIN_AREA': 30,
        'TEXTMASK_MSER_MAX_AREA_RATIO': 0.2,  # fraction of page
        'TEXTMASK_JOIN_W': 15,  # horizontal joining of glyphs
        'TEXTMASK_JOIN_H': 3,  # vertical joining (line thickness)

        # --- DENOISE PARAMETERS ---
        # For denoise.py: Unsharp masking parameters for the pre-boost step
        'PREBOOST_UNSHARP_STRENGTH': 1.8,
        'PREBOOST_UNSHARP_BLUR_WEIGHT': -0.8,

        'SCRIPT_DETECTION_ENABLED': True,  # Auto-detect CJK vs Latin
        'MIXED_CONTENT_MODE': 'auto',  # 'text_only', 'mixed', 'auto'
        'CJK_SAUVOLA_K': 0.15,  # Gentler for complex strokes
        'IMAGE_REGION_PRESERVATION': True,  # Mask non-text areas

        # --- DEBUG PARAMETERS ---
        'DEBUG_DIR': 'data/notebook_outputs/debug_output',
        'DEBUG_FILE_PREFIX': '',  # Optional prefix for debug filenames
        'DEBUG_FILE_LIMIT': 100,  # Max number of debug files to save

        'ORIENT_CKPT_CONVNEXT': str(BASE_DIR / 'experiments/orientation_model/checkpoints/orient_convnext_tiny.pth'),
        'ORIENT_CKPT_EFFNET': str(BASE_DIR / 'experiments/orientation_model/checkpoints/orient_effnetv2s.pth'),  # or None
        'ORIENT_ENS_MARGIN': 0.55,

        # honor this in facade; you already set True in your file
        'DISABLE_HEURISTIC_FALLBACK': True,
    }

    # Layout detection settings
    LAYOUT_DETECTION = {

        # Which detectors to run
        'enabled_detectors': ['figure', 'contours'],
        # Optional extras
        'contours_always': False,
        "figure_margin": 20,

        # ---------------------------
        # YOLO figure detector config
        # ---------------------------
        # Turn this on to swap FigureDetector -> YOLOFigureDetector
        'use_yolo_figure': os.environ.get('USE_YOLO_FIGURE', 'false').lower() == 'true',

        # Absolute path recommended; point this at the best.pt you trained
        'yolo_model_path': os.environ.get(
            'YOLO_MODEL_PATH',
            str(RUNTIME_DIR / 'runs/detect/train6/weights/best.pt')
        ),

        # Inference thresholds tuned for documents (adjust in overrides as needed)
        'yolo_conf': 0.22,  # 0.20–0.25 typical
        'yolo_iou': 0.60,  # preserves neighboring panels
        'yolo_imgsz': 1536,  # good balance for thin lines
        'yolo_tta': False,  # set True for high-recall QA runs only

        # Classical heuristic "figure" candidates are filtered because they can
        # be text. YOLO remains opt-in and should use its own high-confidence
        # path rather than broad corpus-specific exemptions.
        'filter_text_exempt_types': [],
        'figure_isolation_white': 0.55,

        # OCR-free text-like rejection for image/diagram candidates
        'region_reject_text_like': True,
        'region_reject_rotated_text_like': True,
        'region_text_like_min_components': 24,
        'region_text_like_min_density': 0.14,
        'region_text_like_max_density': 0.35,
        'region_text_like_min_median_component_area': 60.0,
        'region_text_like_max_median_component_area': 260.0,
        'region_text_like_max_small_component_fraction': 0.45,
        'region_title_text_max_components': 40,
        'region_title_text_max_row_occupancy': 0.55,
        'region_title_text_min_col_occupancy': 0.62,
        'region_text_line_max_height': 90,
        'region_text_line_min_aspect_ratio': 2.0,
        'region_text_line_min_density': 0.20,
        'region_text_line_max_density': 0.50,
        'region_text_line_min_col_occupancy': 0.75,
        'region_sparse_text_band_max_height': 180,
        'region_sparse_text_band_min_aspect_ratio': 3.0,
        'region_sparse_text_band_min_density': 0.12,
        'region_sparse_text_band_max_density': 0.35,
        'region_sparse_text_band_max_median_component_area': 70.0,
        'region_sparse_text_band_min_small_component_fraction': 0.55,
        'region_vertical_text_max_aspect_ratio': 0.45,
        'region_rotated_text_min_row_occupancy': 0.80,
        'region_rotated_text_min_col_occupancy': 0.75,
        'region_preserve_labeled_diagrams': True,
        'region_labeled_diagram_min_component_area_ratio': 2.4,
        'region_labeled_diagram_min_small_component_fraction': 0.30,
        'region_labeled_diagram_max_density': 0.35,
        'region_merge_overlapping_regions': True,
        'region_merge_adjacent_regions': True,
        'region_overlap_merge_iou_threshold': 0.35,
        'region_contained_region_suppression_threshold': 0.85,
        'region_contained_parent_max_area_ratio': 5.0,
        'region_adjacent_merge_gap_px': 24,
        'region_adjacent_merge_max_area_growth_ratio': 1.75,
        'region_adjacent_merge_min_axis_overlap_ratio': 0.25,
        'region_text_score_reject_threshold': 0.72,
        'region_visual_score_override_threshold': 0.58,
        'region_broad_crop_area_ratio': 0.25,
        'region_broad_crop_visual_override_threshold': 0.78,
        'region_photo_like_min_std': 55.0,
        'region_photo_like_min_dark_fraction': 0.10,
        'region_photo_like_min_edge_density': 0.16,
        'region_visual_min_dimension_for_photo': 120,
        'region_enable_ocr_text_suppression': True,
        'region_ocr_high_overlap_threshold': 0.60,
        'region_ocr_moderate_overlap_threshold': 0.25,
        'region_ocr_low_overlap_threshold': 0.10,
        'region_ocr_rescue_figure_score_threshold': 0.70,
        'region_ocr_rescue_photo_score_threshold': 0.70,
        'region_ocr_rescue_sparse_symbol_score_threshold': 0.65,
        'region_ocr_text_mask_dilation_px': 4,
        'region_enable_mixed_region_refinement': False,
        'region_mixed_region_min_ocr_overlap': 0.25,
        'region_enable_paddle_layout_fusion': False,
        'region_paddle_layout_model_dir': None,

        # ---------------------------
        # Merging & NMS postprocess
        # ---------------------------
        # How aggressively to merge *diagram* boxes (contours); YOLO boxes are 'figure'
        'diagram_merge_iou': 0.10,
        'diagram_merge_gap': 12,

        # Final NMS across all region types
        'final_iou_nms': 0.30,

        # ---------------------------
        # Global sanity (guards)
        # ---------------------------
        # Optional: drop page-scale boxes & page-border boxes early
        'image_max_area_ratio': 0.45,  # ignore boxes >45% of page
        'image_border_margin': 10,  # require a small inset from page edges

        'text_block_min_area': 1000,  # Minimum area for text blocks
        'image_block_min_area': 2500,  # Minimum area for image blocks
        'margin_threshold': 20,  # Pixels for margin detection
        'line_spacing_threshold': 15,  # Pixels for line spacing
        'halo_ring': 4,  # int pixels
        'halo_min_white': 0.85,  # float 0..1

        # Variance / photo
        'variance_window_min': 128,
        'variance_window_rel': 0.125,  # min(h,w)/8
        'variance_stride_rel': 1.0,  # stride = window * 1.0
        'variance_min': 100.0,
        'variance_max': 5000.0,
        'variance_grad_smooth_thresh': 50.0,
        'variance_expand_intensity_thresh': 30.0,
        'variance_min_expanded_area': 20000,

        # Uniform
        'uniform_close_kernel': 15,
        'uniform_min_area_ratio': 0.03,
        'uniform_aspect_min': 0.3,
        'uniform_aspect_max': 3.0,
        'uniform_std_min': 10.0,
        'uniform_std_max': 100.0,

        'contour_min_area': 15000,
        'contour_max_area_ratio': 0.5,
        'contour_aspect_min': 0.3,
        'contour_aspect_max': 3.5,
        'contour_left_bias_xmax': 0.6,

        'contour_canny_lo': 30,
        'contour_canny_hi': 90,
        'contour_dilate_kernel': 5,
        'contour_dilate_iters': 2,

        'contour_hough_min_line_length': 30,
        'contour_hough_max_gap': 10,
        'contour_hough_threshold': 50,
        'contour_edge_density_min': 0.02,
        'contour_edge_density_max': 0.15,
        'contour_white_ratio_min': 0.6,
        'contour_white_ratio_left_bias': 0.7,

        # Final TextRegionFilter owns semantic text rejection by default.
        # Keep broad early CC rejection opt-in because it hid sparse drawings,
        # but reject page-edge text bands early so they do not hide nested art.
        'contour_reject_text_like_early': False,
        'contour_reject_page_edge_text_like': True,
        'contour_cc_small_thresh': 200,
        'contour_cc_small_count': 30,
        'contour_cc_median_area_max': 150,

        'contour_topk': 6,

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
