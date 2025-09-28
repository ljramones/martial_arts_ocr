# Processors Overview - Martial Arts OCR System

## Architecture Summary

The processors form the core processing pipeline of the martial arts OCR system, handling everything from initial content extraction through final document reconstruction. The system follows a modular architecture where each processor has a specific responsibility and can operate independently or as part of the complete pipeline.

## Processing Flow

```
Input Image → Content Extractor → OCR Processor → Japanese Processor → Page Reconstructor → Output
```

## Core Processors

### 1. Content Extractor (`content_extractor.py`)

**Purpose**: Separates text and image content from scanned documents using layout analysis and region classification.

**Key Features**:
- **Layout Analysis**: Detects and classifies regions as text or image using computer vision techniques
- **Feature-Based Classification**: Uses edge density, aspect ratios, connected components, and stroke width analysis
- **Quality Assessment**: Calculates confidence scores for extracted regions
- **Image Preservation**: Extracts and saves embedded diagrams separately with thumbnails

**Key Classes**:
- `ContentExtractor`: Main extraction coordinator
- `ExtractedText`: Text region with metadata
- `ExtractedImage`: Image region with embedded data
- `ContentExtractionResult`: Complete extraction results with metrics

**Technical Approach**:
- Computer vision algorithms for region detection
- Morphological analysis for character recognition
- Heuristic-based classification with confidence scoring
- Automatic region merging to handle overlapping content

---

### 2. Japanese Processor (`japanese_processor.py`)

**Purpose**: Comprehensive Japanese text analysis including romanization, translation, and morphological breakdown.

**Key Features**:
- **Multi-Engine Support**: MeCab morphological analysis, pykakasi romanization, Argos offline translation
- **Character Classification**: Distinguishes hiragana, katakana, kanji, and mixed text
- **Martial Arts Focus**: 60+ specialized terminology database with academic translations
- **HTML Markup**: Interactive tooltips showing romaji and translations

**Key Classes**:
- `JapaneseProcessor`: Main processing coordinator
- `JapaneseTextSegment`: Individual text segment with analysis
- `JapaneseProcessingResult`: Complete analysis results

**Technical Approach**:
- Unicode pattern matching for character type detection
- MeCab integration for linguistic analysis
- Fallback romanization systems for offline operation
- Academic-grade terminology matching

---

### 3. OCR Processor (`ocr_processor.py`)

**Purpose**: Coordinates multiple OCR engines and integrates with Japanese processing for comprehensive text extraction.

**Key Features**:
- **Dual OCR Engine**: Tesseract and EasyOCR with intelligent result selection
- **Quality Scoring**: Multi-factor confidence calculation and engine comparison
- **Language Integration**: Seamless Japanese text processing pipeline
- **Output Generation**: HTML and Markdown format creation

**Key Classes**:
- `OCRProcessor`: Main processing coordinator
- `TesseractEngine`: Tesseract OCR wrapper
- `EasyOCREngine`: EasyOCR wrapper
- `ProcessingResult`: Complete document processing results

**Technical Approach**:
- Engine-specific optimization and configuration
- Confidence-based result selection algorithm
- Integrated language processing pipeline
- Quality metrics and validation

---

### 4. Page Reconstructor (`page_reconstructor.py`)

**Purpose**: Reconstructs documents with preserved layout, embedded images, and academic-quality presentation.

**Key Features**:
- **Layout Preservation**: Maintains original document structure using absolute positioning
- **Interactive Elements**: Japanese text with hover tooltips and confidence indicators
- **Academic Styling**: Professional CSS with proper typography for research use
- **Responsive Design**: Print-friendly with confidence overlays for quality assessment

**Key Classes**:
- `PageReconstructor`: Main reconstruction coordinator
- `PageElement`: Individual page elements with positioning
- `ReconstructedPage`: Complete page with HTML/CSS output

**Technical Approach**:
- CSS absolute positioning for layout preservation
- Interactive JavaScript elements for enhanced usability
- Academic typography standards
- Metadata integration for research documentation

## Integration Architecture

### Data Flow Between Processors

1. **Content Extractor** → **OCR Processor**
   - Regions and extracted images
   - Layout analysis results
   - Quality metrics

2. **OCR Processor** → **Japanese Processor**
   - Raw text content
   - Language detection hints
   - Region-specific text

3. **Japanese Processor** → **Page Reconstructor**
   - Processed text with markup
   - Translation and romanization
   - Martial arts terminology

4. **All Processors** → **Database/Output**
   - Processing metadata
   - Quality scores
   - Academic documentation

### Quality Control Pipeline

Each processor implements comprehensive quality control:

- **Confidence Scoring**: 0-1 scale for all operations
- **Fallback Strategies**: Graceful degradation when libraries unavailable
- **Error Handling**: Robust exception handling with logging
- **Validation**: Content validation and quality metrics

## Academic Research Features

### Donn Draeger Specialization

- **Terminology Database**: Comprehensive martial arts vocabulary with academic translations
- **Historical Context**: Preservation of original layout and embedded diagrams
- **Research Quality**: Professional documentation suitable for academic publication
- **Metadata Tracking**: Complete processing provenance for scholarly citation

### Quality Assurance

- **Multiple OCR Engines**: Cross-validation for accuracy
- **Confidence Indicators**: Visual quality assessment in output
- **Processing Statistics**: Comprehensive metrics for research validation
- **Error Documentation**: Transparent quality reporting

## Technical Dependencies

### Core Libraries
- **OpenCV**: Computer vision and image processing
- **Tesseract/EasyOCR**: OCR engines
- **MeCab**: Japanese morphological analysis
- **pykakasi**: Japanese romanization
- **Argos Translate**: Offline translation

### Fallback Strategies
- **Simple romanization**: When pykakasi unavailable
- **Basic translation**: Martial arts dictionary lookup
- **Character mapping**: Manual hiragana/katakana conversion
- **Layout estimation**: When region detection fails

## Performance Characteristics

### Processing Speed
- **Content Extraction**: ~2-5 seconds per page
- **OCR Processing**: ~5-15 seconds depending on engines
- **Japanese Analysis**: ~1-3 seconds per segment
- **Page Reconstruction**: ~1-2 seconds

### Memory Usage
- **Image Processing**: ~50-200MB per document
- **OCR Engines**: ~100-500MB initialization
- **Japanese Libraries**: ~50-100MB
- **Total System**: ~200MB-1GB depending on document size

## Error Handling and Robustness

### Graceful Degradation
- Missing libraries disable specific features without system failure
- Alternative processing paths for different configurations
- Quality indicators show when fallback methods used

### Logging and Debugging
- Comprehensive logging at DEBUG, INFO, WARNING, ERROR levels
- Processing statistics for performance monitoring
- Quality metrics for academic validation

## Future Extension Points

### Processor Modularity
- New OCR engines can be added as plugins
- Additional language processors follow same interface
- Custom reconstruction templates for different document types

### Academic Features
- Additional terminology databases
- Custom markup for specific research fields
- Integration with citation management systems

## Usage Patterns

### Development/Testing
```python
from processors import process_document
result = process_document("draeger_lecture.jpg")
print(f"Quality: {result.quality_score:.1%}")
```

### Production Pipeline
```python
from processors import ContentExtractor, OCRProcessor, PageReconstructor

extractor = ContentExtractor()
ocr = OCRProcessor()
reconstructor = PageReconstructor()

# Full pipeline with error handling
extraction = extractor.extract_content(image_path)
processing = ocr.process_document(image_path)
page = reconstructor.reconstruct_page(processing)
```

### Batch Processing
```python
from processors import batch_extract_content
results = batch_extract_content(image_list, output_dir)
```

This processor architecture provides a robust, academically-focused system for digitizing Donn Draeger's martial arts materials with preservation of both content and scholarly context.