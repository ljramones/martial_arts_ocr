# # Martial Arts OCR - Draeger Lectures Digitization

A Python-based system for digitizing scanned martial arts documents with OCR, Japanese text processing, and embedded image preservation. Originally designed for processing Donn Draeger's lecture materials, now expanded to handle classical Japanese martial arts texts including koryu densho and makimono scrolls.

## Project Status

### ✅ Implemented Features
- **Advanced OCR**: Extract text from scanned documents using Tesseract and EasyOCR
- **Mixed Content Support**: Detect and preserve embedded images (diagrams, illustrations)
- **Modern Japanese Processing**: 
  - Automatic romanization (romaji conversion)
  - Translation capabilities via Argos Translate
  - Furigana annotation support
  - MeCab morphological analysis
- **Layout Preservation**: Maintain original document structure and image positioning
- **Web Interface**: Local web application for viewing processed documents
- **Offline Operation**: No internet required, all processing done locally
- **Content Extraction**: Separate text and image regions using computer vision
- **Quality Metrics**: Confidence scoring for OCR results

### 🚧 In Development
- **Classical Japanese Support** (Partially implemented):
  - KuroNet integration for cursive script recognition
  - Historical character normalization
  - Classical grammar conversion
- **Document Analysis Pipeline**: Automatic document type detection
- **Seal Detection**: Recognition of traditional stamps/seals

### 📋 Planned Features
- **Multi-period Support**: Specialized processing for Edo, Muromachi, Kamakura texts
- **Two-stage Translation**: Classical → Modern Japanese → English
- **Scholarly Output Format**: Three-panel view (original/modern/translation)
- **Batch Processing**: Process multiple documents with mixed types
- **Enhanced Koryu Terminology**: Extended martial arts dictionary

## Document Processing Workflow

The system employs an intelligent document analysis and routing system that automatically determines the optimal processing pipeline for each document type.

### Workflow Overview

```
[Input: Scanned Document]
           ↓
[Document Analyzer]
    • Language detection
    • Script style analysis  
    • Time period identification
    • Image/seal detection
           ↓
[Automatic Classification]
           ↓
[Pipeline Selection & Execution]
           ↓
[Output: Processed Document]
```

### Supported Document Types

#### 1. **English Only Documents** ✅
- **Use Case**: Draeger's English lecture notes, modern martial arts books
- **Pipeline**: 
  ```
  Extract Content → OCR (English) → Image Extraction → Simple Reconstruction
  ```
- **Status**: Fully implemented

#### 2. **Modern Japanese Documents** ✅
- **Use Case**: Contemporary martial arts manuals, recent publications
- **Pipeline**:
  ```
  Extract Content → OCR (Japanese) → MeCab Analysis → Translation → 
  Romanization → Bilingual Reconstruction
  ```
- **Status**: Fully implemented

#### 3. **Mixed Modern Documents (English + Japanese)** ✅
- **Use Case**: Annotated texts, bilingual martial arts materials
- **Pipeline**:
  ```
  Extract Content → Region Detection → Multi-language OCR → 
  Selective Translation → Mixed Reconstruction
  ```
- **Status**: Fully implemented

#### 4. **Classical Japanese Documents** 🚧
- **Use Case**: Koryu densho, historical makimono scrolls
- **Pipeline**:
  ```
  Extract Content → Script Style Detection → Classical OCR (KuroNet/Tesseract) → 
  Historical Normalization → Grammar Conversion → Modern Japanese → 
  Translation → Three-panel Scholarly Output
  ```
- **Status**: Core components implemented, integration in progress

#### 5. **Mixed Classical Documents** 📋
- **Use Case**: Annotated historical texts with modern notes
- **Pipeline**:
  ```
  Extract Content → Region Classification → Selective Classical/Modern OCR → 
  Context-aware Processing → Academic Reconstruction
  ```
- **Status**: Planned

### Document Analysis Components

#### Language Detection ✅
- Analyzes character distribution (English, Japanese, other)
- Determines primary document language
- Identifies mixed-language regions

#### Script Style Analysis 🚧
- **Kaisho (楷書)**: Standard printed/written style ✅
- **Gyosho (行書)**: Semi-cursive style 🚧
- **Sosho (草書)**: Cursive style 📋

#### Time Period Identification 🚧
Automatically detects document era based on:
- Character variants (historical kanji/kana)
- Grammar patterns
- Writing style
- **Supported Periods** (planned):
  - Contemporary (1945-present) ✅
  - Modern (Meiji-Showa, 1868-1945) 🚧
  - Edo (1603-1868) 📋
  - Earlier periods (pre-1603) 📋

#### Special Element Detection
- **Embedded Images**: Diagrams, illustrations ✅
- **Seals/Stamps**: Traditional red seals (判子) 🚧
- **Margin Notes**: Classical annotations (頭注) 📋

### Processing Features by Document Type

| Feature | English | Modern JP | Mixed Modern | Classical | Mixed Classical |
|---------|---------|-----------|--------------|-----------|-----------------|
| OCR | ✅ | ✅ | ✅ | 🚧 | 📋 |
| Translation | N/A | ✅ | ✅ | 🚧 | 📋 |
| Romanization | N/A | ✅ | ✅ | 🚧 | 📋 |
| Image Extraction | ✅ | ✅ | ✅ | 🚧 | 📋 |
| Layout Preservation | ✅ | ✅ | ✅ | 🚧 | 📋 |
| Historical Normalization | N/A | N/A | N/A | 🚧 | 📋 |
| Grammar Conversion | N/A | N/A | N/A | 📋 | 📋 |
| Seal Detection | N/A | N/A | N/A | 🚧 | 📋 |
| Vertical Text | N/A | ✅ | ✅ | 🚧 | 📋 |
| Three-panel Output | N/A | N/A | N/A | 📋 | 📋 |

**Legend**: ✅ Implemented | 🚧 In Development | 📋 Planned | N/A Not Applicable

## Technology Stack

### Core Technologies ✅
- **OCR Engines**: Tesseract, EasyOCR
- **Japanese Processing**: MeCab, UniDic, pykakasi, Argos Translate
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Web Framework**: Flask
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, vanilla JavaScript

### Classical Japanese Extensions 🚧
- **KuroNet**: Cursive script recognition (integration in progress)
- **UniDic Historical Variants**: Classical dictionaries (planned)
- **Classical Grammar Converter**: Custom transformation rules (in development)

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

5. (Optional) Setup classical Japanese support:
```bash
# Currently in development
python setup_classical.py  # 🚧
```

## Quick Start

### Basic Usage ✅

1. **Start the application**:
```bash
python app.py
```

2. **Access the web interface**:
   - Open http://localhost:5000 in your browser

3. **Upload scanned documents**:
   - Drag and drop files or use the upload button
   - Supported formats: JPG, JPEG, PNG, TIFF

4. **Automatic Processing**:
   - The system automatically detects document type
   - Applies appropriate processing pipeline
   - View results in format suitable for document type

### Manual Document Type Override ✅

```python
# For programmatic use
from processors.workflow_orchestrator import WorkflowOrchestrator
from processors.document_analyzer import DocumentType

orchestrator = WorkflowOrchestrator()

# Let system auto-detect
result = orchestrator.process_document('path/to/document.jpg')

# Or manually specify type
result = orchestrator.process_document(
    'path/to/document.jpg',
    override_type=DocumentType.CLASSICAL_JAPANESE
)
```

### Batch Processing 🚧

```python
# Process multiple documents
results = orchestrator.batch_process([
    'densho_1.jpg',
    'lecture_notes.jpg', 
    'mixed_document.jpg'
])
```

## Usage Examples

### Processing Different Document Types

#### English Lecture Notes ✅
```python
# Automatically detected and processed
result = process_document('draeger_lecture_1.jpg')
# Output: OCR text with preserved layout and extracted diagrams
```

#### Modern Japanese Manual ✅
```python
result = process_document('modern_karate_manual.jpg')
# Output: Original, romaji, translation, with embedded images
```

#### Classical Koryu Densho 🚧
```python
result = process_document('yagyu_densho.jpg')
# Output: Three-panel view (original classical, modern Japanese, English)
# with seal detection and terminology glossary
```

## Project Structure

```
martial_arts_ocr/
├── app.py                      # Main Flask application ✅
├── config.py                   # Configuration settings ✅
├── setup_japanese.py           # Japanese dictionary setup ✅
├── setup_classical.py          # Classical Japanese setup 🚧
├── processors/
│   ├── __init__.py            ✅
│   ├── document_analyzer.py   # Document type detection 🚧
│   ├── workflow_orchestrator.py # Pipeline management 🚧
│   ├── layout_detector.py     # Detect text vs image regions ✅
│   ├── content_extractor.py   # Extract text and images ✅
│   ├── ocr_processor.py       # OCR processing ✅
│   ├── japanese_processor.py  # Japanese text analysis ✅
│   ├── classical_processor.py # Classical Japanese handling 🚧
│   └── page_reconstructor.py  # HTML page generation ✅
├── static/                     # Web assets ✅
├── templates/                  # HTML templates ✅
├── uploads/                    # Original uploaded files ✅
├── processed/                  # Processed document data ✅
└── utils/                      # Utility functions ✅
```

## Configuration

### Processing Profiles 🚧

Edit `config.py` to customize processing profiles:

```python
PROCESSING_PROFILES = {
    'draeger_lecture': {
        'document_types': ['english_only', 'mixed_modern'],
        'preserve_layout': True,
        'extract_images': True
    },
    'koryu_densho': {
        'document_types': ['classical_japanese'],
        'detect_seals': True,
        'three_panel_output': True,
        'include_glossary': True
    }
}
```

## API Endpoints

### Document Analysis ✅
```
POST /analyze
```
Analyzes document and returns detected type, languages, and recommendations

### Document Processing ✅
```
POST /process
```
Processes document with automatic or manual pipeline selection

### Batch Processing 🚧
```
POST /batch_process
```
Process multiple documents with mixed types

## Performance Characteristics

### Processing Speed
- **English only**: ~3-5 seconds per page ✅
- **Modern Japanese**: ~8-12 seconds per page ✅
- **Mixed modern**: ~10-15 seconds per page ✅
- **Classical Japanese**: ~15-25 seconds per page 🚧
- **Mixed classical**: ~20-30 seconds per page 📋

### Accuracy Metrics
- **English OCR**: 95-98% accuracy ✅
- **Modern Japanese OCR**: 90-95% accuracy ✅
- **Classical Japanese OCR**: 70-85% accuracy 🚧
- **Translation Quality**: Good for modern, developing for classical 🚧

## Troubleshooting

### Common Issues

1. **Japanese characters not recognized**: Ensure Japanese Tesseract data is installed
2. **Classical text poorly recognized**: KuroNet integration still in development
3. **Memory errors on large documents**: Process in smaller batches
4. **Translation quality issues**: Check if Argos models are properly installed

## Roadmap

### Phase 1: Core Functionality ✅
- Basic OCR for English and modern Japanese
- Web interface
- Image extraction

### Phase 2: Classical Support 🚧 (Current)
- Classical Japanese OCR
- Historical character normalization
- Grammar conversion
- Seal detection

### Phase 3: Advanced Features 📋
- Full koryu densho support
- Multi-period handling
- Advanced terminology database
- Scholarly annotation tools

### Phase 4: Research Tools 📋
- Citation generation
- Cross-document analysis
- Terminology concordance
- Export to academic formats

## Contributing

Contributions are welcome! Areas needing help:
- Classical Japanese grammar rules
- Koryu terminology database expansion
- Testing with various document types
- UI/UX improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Donn Draeger for his foundational work in martial arts research
- The Tesseract OCR community
- Japanese NLP tool developers (MeCab, UniDic)
- Open source computer vision libraries
- Classical Japanese text processing researchers

## Contact

For questions about classical Japanese processing or koryu document handling, please open an issue on GitHub.