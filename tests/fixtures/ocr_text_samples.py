from __future__ import annotations


ENGLISH_PARAGRAPH_WORDS = [
    {"text": "Donn", "x": 10, "y": 12, "width": 30, "height": 12, "confidence": 0.95, "engine": "fake_test"},
    {"text": "Draeger", "x": 45, "y": 12, "width": 50, "height": 12, "confidence": 0.94, "engine": "fake_test"},
    {"text": "studied", "x": 100, "y": 12, "width": 48, "height": 12, "confidence": 0.93, "engine": "fake_test"},
    {"text": "classical", "x": 10, "y": 32, "width": 58, "height": 12, "confidence": 0.92, "engine": "fake_test"},
    {"text": "bujutsu.", "x": 74, "y": 32, "width": 56, "height": 12, "confidence": 0.91, "engine": "fake_test"},
]


MIXED_MARTIAL_ARTS_WORDS = [
    {"text": "武道", "x": 10, "y": 10, "width": 32, "height": 14, "confidence": 0.9, "engine": "fake_test", "language": "ja"},
    {"text": "とは何か。", "x": 46, "y": 10, "width": 70, "height": 14, "confidence": 0.88, "engine": "fake_test", "language": "ja"},
    {"text": "Daitō-ryū", "x": 10, "y": 34, "width": 72, "height": 12, "confidence": 0.92, "engine": "fake_test"},
    {"text": "Aiki-jūjutsu", "x": 88, "y": 34, "width": 92, "height": 12, "confidence": 0.9, "engine": "fake_test"},
    {"text": "「柔術」・budō", "x": 10, "y": 58, "width": 110, "height": 14, "confidence": 0.89, "engine": "fake_test"},
    {"text": "—", "x": 126, "y": 58, "width": 12, "height": 14, "confidence": 0.87, "engine": "fake_test"},
    {"text": "koryū", "x": 144, "y": 58, "width": 44, "height": 14, "confidence": 0.88, "engine": "fake_test"},
]


NOISY_OCR_TEXT = "  Donn   Draeger\\n\\n\\n studied   budō  \\nDaitō-ryū   "


TESSERACT_LIKE_ROWS = {
    "text": ["Donn", "Draeger", "武道", "koryū"],
    "left": [10, 45, 10, 55],
    "top": [12, 12, 34, 34],
    "width": [30, 50, 38, 44],
    "height": [12, 12, 14, 12],
    "conf": [95, 94, 88, 91],
    "level": [5, 5, 5, 5],
    "engine": "tesseract",
}


EASYOCR_LIKE_POLYGON_ROWS = [
    ([(10, 10), (82, 10), (82, 24), (10, 24)], "Daitō-ryū", 0.9),
    ([(10, 34), (92, 35), (91, 49), (10, 48)], "「柔術」・budō", 0.86),
]
