# Utils Directory Overview

This directory contains utility modules for image and text processing in the Martial Arts OCR system. These utilities handle the complex preprocessing, analysis, and post-processing needed to extract high-quality text from scanned martial arts documents.

## Module Structure

```
utils/
├── __init__.py          # Package initialization
├── image_utils.py       # Image processing and layout analysis
├── text_utils.py        # Text cleaning and language processing
└── OVERVIEW.md          # This documentation file
```

## Core Philosophy

The utilities are designed around the specific challenges of digitizing historical martial arts documents:

- **Mixed content**: Documents contain both text and technical diagrams
- **Multiple languages**: English text with Japanese terminology
- **Varied quality**: Scanned documents with different lighting and skew
- **Preservation**: Maintain original layout and positioning of elements

---

## Image Utils (`image_utils.py`)

### Purpose
Handles all image-related operations including preprocessing for OCR, layout analysis, and region extraction.

### Key Classes

#### `ImageProcessor`
Main class for preparing images for OCR processing.

**Features:**
- Automatic deskewing (corrects tilted scans)
- Noise reduction and contrast enhancement
- Optimal resizing for OCR accuracy
- Binary conversion for text recognition

**Example Usage:**
```python
from utils.image_utils import ImageProcessor

processor = ImageProcessor()

# Load and preprocess an image
image = processor.load_image("draeger_lecture_page1.jpg")
processed = processor.preprocess_for_ocr(image)

# Get image information
info = processor.get_image_info("draeger_lecture_page1.jpg")
print(f"Image size: {info.width}x{info.height}")
print(f"File format: {info.format}")
```

#### `LayoutAnalyzer`
Detects and separates text regions from diagrams/images.

**Features:**
- Text region detection using MSER (Maximally Stable Extremal Regions)
- Image/diagram region detection using contour analysis
- Overlap detection and region merging
- Confidence scoring for detected regions

**Example Usage:**
```python
from utils.image_utils import LayoutAnalyzer

analyzer = LayoutAnalyzer()

# Detect different types of content
text_regions = analyzer.detect_text_regions(image)
image_regions = analyzer.detect_image_regions(image)

# Merge overlapping regions
all_regions = text_regions + image_regions
merged_regions = analyzer.merge_overlapping_regions(all_regions)

# Process each region
for region in merged_regions:
    print(f"Found {region.region_type} at ({region.x}, {region.y})")
    print(f"Size: {region.width}x{region.height}, Confidence: {region.confidence}")
```

#### `ImageRegion`
Data class representing a rectangular area in an image.

**Properties:**
```python
region = ImageRegion(x=100, y=50, width=200, height=150, region_type="text")

print(region.area)          # 30000
print(region.bbox)          # (100, 50, 300, 200)
print(region.to_dict())     # Complete region data
```

### Utility Functions

```python
from utils.image_utils import (
    save_image, extract_image_region, 
    create_thumbnail, validate_image_file
)

# Save processed image
save_image(processed_image, "output/processed.jpg", quality=95)

# Extract specific region
diagram = extract_image_region(image, diagram_region)

# Create thumbnail for web interface
thumb = create_thumbnail(image, size=(200, 300))

# Validate uploaded file
is_valid = validate_image_file("user_upload.jpg")
```

---

## Text Utils (`text_utils.py`)

### Purpose
Handles text cleaning, language detection, and formatting for optimal OCR results and user presentation.

### Key Classes

#### `TextCleaner`
Cleans and normalizes OCR output to remove artifacts and errors.

**Features:**
- Common OCR error correction (rn→m, vv→w, 0→o in words)
- Artifact removal (stray punctuation, isolated characters)
- Whitespace normalization
- Aggressive cleaning mode for heavily corrupted text

**Example Usage:**
```python
from utils.text_utils import TextCleaner

cleaner = TextCleaner()

# Raw OCR output (with typical errors)
raw_text = """
THE  DRAEGER  LECTURES
AT
THE  UMVERSITY  OF  HAWAll

1.  BUJUTSU  AND  BUDO.
2.  RANKlNG  SYSTEMS  IN  THE  JAPANESE  MARTlAL  ARTS:  MODERN  VS.  CLASSlCAL.
"""

# Clean the text
cleaned_text, stats = cleaner.clean_text(raw_text)

print("Cleaned text:")
print(cleaned_text)
print(f"\nCleaning stats:")
print(f"Characters removed: {stats.characters_removed}")
print(f"Compression ratio: {stats.compression_ratio:.2f}")
```

#### `LanguageDetector`
Identifies and segments text by language (English vs Japanese).

**Features:**
- Automatic language detection using langdetect
- Japanese character range detection
- Mixed-language text segmentation
- Confidence scoring for language identification

**Example Usage:**
```python
from utils.text_utils import LanguageDetector

detector = LanguageDetector()

# Mixed English/Japanese text
mixed_text = "The term 武道 (budō) refers to martial arts. Kendo is one example."

# Segment by language
segments = detector.segment_by_language(mixed_text)

for segment in segments:
    print(f"Language: {segment.language}")
    print(f"Text: '{segment.text}'")
    print(f"Confidence: {segment.confidence}")
    print("---")

# Quick language check
is_japanese = detector.is_japanese_text("武道の研究")  # True
```

#### `TextFormatter`
Converts text to different output formats.

**Example Usage:**
```python
from utils.text_utils import TextFormatter

formatter = TextFormatter()

text = "The Draeger Lectures\n\nBujutsu and Budo study."

# Convert to different formats
html = formatter.to_html(text, preserve_formatting=True)
markdown = formatter.to_markdown(text, title="Lecture Notes")
plain = formatter.to_plain_text(text, max_line_length=60)
```

#### `TextStatistics`
Calculates comprehensive statistics about text content.

**Example Usage:**
```python
from utils.text_utils import TextStatistics

stats = TextStatistics.get_stats(text)

print(f"Words: {stats['words']}")
print(f"Characters: {stats['characters']}")
print(f"Japanese chars: {stats['japanese_characters']}")
print(f"Reading time: {stats['reading_time_minutes']:.1f} minutes")
print(f"Language ratio - Japanese: {stats['language_ratio']['japanese']:.2f}")
```

### Utility Functions

```python
from utils.text_utils import (
    extract_martial_arts_terms, split_into_sentences,
    normalize_japanese_text, confidence_score_text
)

# Extract martial arts terminology
terms = extract_martial_arts_terms(text)
# Returns: ['kata', 'dojo', 'bushido', 'karate-do', etc.]

# Split into sentences (handles Japanese punctuation)
sentences = split_into_sentences(text)

# Normalize Japanese text
normalized = normalize_japanese_text("ａｂｃ１２３")  # "abc123"

# Calculate text confidence score
confidence = confidence_score_text(ocr_output)  # 0.0 to 1.0
```

---

## Complete Workflow Examples

### Processing a Draeger Lecture Page

```python
from utils.image_utils import ImageProcessor, LayoutAnalyzer
from utils.text_utils import TextCleaner, LanguageDetector

# 1. Load and preprocess image
processor = ImageProcessor()
image = processor.load_image("lecture_page.jpg")
processed_image = processor.preprocess_for_ocr(image)

# 2. Analyze layout
analyzer = LayoutAnalyzer()
text_regions = analyzer.detect_text_regions(processed_image)
image_regions = analyzer.detect_image_regions(processed_image)

print(f"Found {len(text_regions)} text regions")
print(f"Found {len(image_regions)} image regions")

# 3. Process OCR results (assuming OCR was run)
cleaner = TextCleaner()
detector = LanguageDetector()

raw_ocr_text = "Raw OCR output here..."
cleaned_text, cleaning_stats = cleaner.clean_text(raw_ocr_text)

# 4. Segment by language
segments = detector.segment_by_language(cleaned_text)

# 5. Generate output
for segment in segments:
    if segment.language == 'ja':
        print(f"Japanese text found: {segment.text}")
    else:
        print(f"English text: {segment.text}")
```

### Batch Processing Multiple Images

```python
from pathlib import Path
from utils.image_utils import ImageProcessor, validate_image_file
from utils.text_utils import TextStatistics

processor = ImageProcessor()
scan_directory = Path("scanned_lectures")

total_stats = {
    'processed': 0,
    'failed': 0,
    'total_text_length': 0
}

for image_file in scan_directory.glob("*.jpg"):
    try:
        # Validate file
        if not validate_image_file(str(image_file)):
            print(f"Invalid image file: {image_file}")
            total_stats['failed'] += 1
            continue
        
        # Get image info
        info = processor.get_image_info(str(image_file))
        print(f"Processing: {image_file.name} ({info.width}x{info.height})")
        
        # Preprocess
        image = processor.load_image(str(image_file))
        processed = processor.preprocess_for_ocr(image)
        
        # Save processed version
        output_path = f"processed/{image_file.stem}_processed.jpg"
        save_image(processed, output_path)
        
        total_stats['processed'] += 1
        
    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
        total_stats['failed'] += 1

print(f"\nBatch processing complete:")
print(f"Processed: {total_stats['processed']}")
print(f"Failed: {total_stats['failed']}")
```

### Quality Assessment Pipeline

```python
from utils.text_utils import confidence_score_text, TextStatistics
from utils.image_utils import ImageProcessor

def assess_ocr_quality(image_path: str, ocr_text: str) -> dict:
    """Comprehensive quality assessment of OCR results."""
    
    # Image quality metrics
    processor = ImageProcessor()
    image_info = processor.get_image_info(image_path)
    
    # Text quality metrics
    text_confidence = confidence_score_text(ocr_text)
    text_stats = TextStatistics.get_stats(ocr_text)
    
    # Quality assessment
    quality_score = text_confidence
    
    # Penalize very short text (likely failed OCR)
    if text_stats['words'] < 10:
        quality_score *= 0.5
    
    # Boost score for reasonable text length
    if 50 <= text_stats['words'] <= 1000:
        quality_score *= 1.1
    
    # Assessment report
    assessment = {
        'overall_quality': 'excellent' if quality_score > 0.8 else
                          'good' if quality_score > 0.6 else
                          'fair' if quality_score > 0.4 else 'poor',
        'confidence_score': quality_score,
        'text_stats': text_stats,
        'image_info': image_info.to_dict(),
        'recommendations': []
    }
    
    # Recommendations
    if quality_score < 0.6:
        assessment['recommendations'].append("Consider re-scanning with higher resolution")
    if text_stats['japanese_characters'] > 0:
        assessment['recommendations'].append("Japanese text detected - verify romanization")
    if image_info.width < 1500:
        assessment['recommendations'].append("Low resolution image - OCR accuracy may be limited")
    
    return assessment

# Usage
quality = assess_ocr_quality("lecture_page.jpg", ocr_output)
print(f"Quality: {quality['overall_quality']}")
print(f"Score: {quality['confidence_score']:.2f}")
for rec in quality['recommendations']:
    print(f"• {rec}")
```

---

## Configuration Integration

The utilities respect settings from `config.py`:

```python
# In config.py
IMAGE_PROCESSING = {
    'enhance_contrast': True,
    'denoise': True,
    'deskew': True,
    'resize_factor': 1.5,
    'min_image_size': (100, 100),
    'max_image_size': (2000, 2000),
}

LAYOUT_DETECTION = {
    'text_block_min_area': 1000,
    'image_block_min_area': 2500,
    'margin_threshold': 20,
    'line_spacing_threshold': 15,
}
```

The utilities automatically use these settings for consistent processing across the application.

---

## Error Handling

All utilities include comprehensive error handling:

```python
try:
    processor = ImageProcessor()
    image = processor.load_image("problematic_image.jpg")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid image format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

Errors are logged using Python's logging module, so they can be tracked and debugged effectively.

---

## Performance Considerations

- **Memory usage**: Large images are processed in chunks when possible
- **Processing time**: Preprocessing steps can be disabled for faster processing
- **Caching**: Processed images can be cached to avoid reprocessing
- **Batch processing**: Utilities support batch operations for multiple files

## Extension Points

The utility modules are designed to be easily extended:

- Add new OCR error correction patterns to `TextCleaner`
- Implement additional image preprocessing algorithms in `ImageProcessor`
- Add specialized martial arts terminology detection
- Integrate additional language detection libraries
- Implement custom layout analysis algorithms for specific document types

This modular design allows the system to evolve and improve while maintaining backward compatibility.