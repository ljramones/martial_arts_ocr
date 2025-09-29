# Martial Arts OCR - Draeger Lectures Digitization

A Python-based system for digitizing scanned martial arts documents with OCR, Japanese text processing, and embedded image preservation. Specifically designed for processing Donn Draeger's lecture materials.

## Features

- **Advanced OCR**: Extract text from scanned documents using Tesseract and EasyOCR
- **Mixed Content Support**: Detect and preserve embedded images (diagrams, illustrations)
- **Japanese Text Processing**: 
  - Automatic romanization (romaji conversion)
  - Translation capabilities
  - Furigana annotation support
- **Layout Preservation**: Maintain original document structure and image positioning
- **Web Interface**: Local web application for viewing processed documents
- **Offline Operation**: No internet required, all processing done locally

## Technology Stack

- **OCR Engines**: Tesseract, EasyOCR
- **Japanese Processing**: MeCab, UniDic, pykakasi, Argos Translate
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Web Framework**: Flask
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, vanilla JavaScript

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine
- System dependencies for Japanese text processing

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert
sudo apt-get install libmecab-dev mecab mecab-ipadic-utf8
```

#### macOS
```bash
brew install tesseract tesseract-lang
brew install mecab mecab-ipadic
```

#### Windows
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Install with Japanese language packs
- Download MeCab from: https://taku910.github.io/mecab/

### Python Environment

1. Clone the repository:
```bash
git clone https://github.com/ljramones/martial_arts_ocr
cd martial_arts_ocr
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Japanese dictionaries:
```bash
python setup_japanese.py
```

## Quick Start

1. **Start the application**:
```bash
python app.py
```

2. **Access the web interface**:
   - Open http://localhost:5000 in your browser

3. **Upload scanned documents**:
   - Drag and drop JPEG files or use the upload button
   - Supported formats: JPG, JPEG, PNG, TIFF

4. **Process documents**:
   - Click "Process" to start OCR and analysis
   - View results with embedded images preserved

## Project Structure

```
martial_arts_ocr/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── setup_japanese.py           # Japanese dictionary setup
├── processors/
│   ├── __init__.py
│   ├── layout_detector.py      # Detect text vs image regions
│   ├── content_extractor.py    # Extract text and images
│   ├── ocr_processor.py        # OCR processing
│   ├── japanese_processor.py   # Japanese text analysis
│   └── page_reconstructor.py   # HTML page generation
├── static/
│   ├── css/
│   │   ├── main.css           # Main stylesheet
│   │   └── viewer.css         # Document viewer styles
│   ├── js/
│   │   ├── upload.js          # File upload handling
│   │   ├── viewer.js          # Document viewer
│   │   └── japanese.js        # Japanese text interactions
│   └── extracted_content/      # Processed images and data
├── templates/
│   ├── base.html              # Base template
│   ├── index.html             # Upload interface
│   ├── page_view.html         # Single page viewer
│   └── gallery.html           # Document gallery
├── uploads/                    # Original uploaded files
├── processed/                  # Processed document data
├── database.py                 # Database operations
├── models.py                   # Data models
└── utils/
    ├── __init__.py
    ├── image_utils.py         # Image processing utilities
    └── text_utils.py          # Text processing utilities
```

## Configuration

Edit `config.py` to customize:

- OCR engine preferences
- Japanese processing options
- Output formats
- File storage locations

## Usage Examples

### Basic Document Processing
```python
from processors.ocr_processor import OCRProcessor

processor = OCRProcessor()
result = processor.process_image('path/to/scan.jpg')
print(result.text)  # Extracted text
print(result.images)  # Embedded images with positions
```

### Japanese Text Analysis
```python
from processors.japanese_processor import JapaneseProcessor

jp_processor = JapaneseProcessor()
analysis = jp_processor.analyze_text("武道の研究")
print(analysis.romaji)      # "budō no kenkyū"
print(analysis.translation) # "martial arts research"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Donn Draeger for his foundational work in martial arts research
- The Tesseract OCR community
- Japanese NLP tool developers (MeCab, UniDic)
- Open source computer vision libraries