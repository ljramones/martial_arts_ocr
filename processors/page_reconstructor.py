"""
Page Reconstructor for Martial Arts OCR
Reconstructs document pages with preserved layout, embedded images, and formatted text.
Generates HTML pages that maintain the original document structure.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config import get_config
from utils.image_utils import ImageRegion, save_image, extract_image_region
from utils.text_utils import TextFormatter, TextStatistics
from processors.ocr_processor import ProcessingResult
from processors.japanese_processor import JapaneseProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class PageElement:
    """Represents an element on a reconstructed page."""
    element_type: str  # "text", "image", "heading", "paragraph"
    content: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_type': self.element_type,
            'content': self.content,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }


@dataclass
class ReconstructedPage:
    """A complete reconstructed page with layout and styling."""
    page_id: str
    title: str
    elements: List[PageElement]
    html_content: str
    css_styles: str
    page_width: int
    page_height: int
    reconstruction_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'page_id': self.page_id,
            'title': self.title,
            'elements': [element.to_dict() for element in self.elements],
            'html_content': self.html_content,
            'css_styles': self.css_styles,
            'page_width': self.page_width,
            'page_height': self.page_height,
            'reconstruction_metadata': self.reconstruction_metadata
        }


class PageReconstructor:
    """Main page reconstruction class."""

    def __init__(self):
        self.config = get_config()
        self.text_formatter = TextFormatter()

        # Layout configuration
        self.layout_config = {
            'margin_top': 40,
            'margin_bottom': 40,
            'margin_left': 40,
            'margin_right': 40,
            'line_height': 1.6,
            'paragraph_spacing': 20,
            'heading_spacing': 30,
            'image_padding': 15,
            'text_image_spacing': 25
        }

        # Typography settings
        self.typography = {
            'base_font_size': 14,
            'heading_font_size': 18,
            'line_height': 1.6,
            'font_family': "'Georgia', 'Times New Roman', serif",
            'japanese_font_family': "'Hiragino Sans', 'Yu Gothic', 'Meiryo', sans-serif"
        }

    def reconstruct_page(self, processing_result: ProcessingResult,
                         original_image_path: str = None) -> ReconstructedPage:
        """
        Reconstruct a page from OCR processing results.

        Args:
            processing_result: Results from OCR processing
            original_image_path: Path to original image for reference

        Returns:
            ReconstructedPage with layout and content
        """
        try:
            logger.info("Starting page reconstruction")

            # Generate page ID
            page_id = f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Extract page dimensions from original image if available
            page_width, page_height = self._get_page_dimensions(
                original_image_path, processing_result
            )

            # Create page elements from processing results
            elements = self._create_page_elements(processing_result, page_width, page_height)

            # Sort elements by position (top-to-bottom, left-to-right)
            sorted_elements = self._sort_elements_by_layout(elements)

            # Generate CSS styles
            css_styles = self._generate_css_styles(page_width, page_height)

            # Generate HTML content
            html_content = self._generate_html_content(sorted_elements, processing_result)

            # Create title from content
            title = self._generate_page_title(processing_result)

            # Create reconstruction metadata
            reconstruction_metadata = {
                'reconstruction_date': datetime.now().isoformat(),
                'source_image': original_image_path,
                'total_elements': len(elements),
                'text_elements': len([e for e in elements if e.element_type == 'text']),
                'image_elements': len([e for e in elements if e.element_type == 'image']),
                'japanese_detected': processing_result.japanese_result is not None,
                'overall_confidence': processing_result.overall_confidence,
                'processing_time': processing_result.processing_time
            }

            reconstructed_page = ReconstructedPage(
                page_id=page_id,
                title=title,
                elements=sorted_elements,
                html_content=html_content,
                css_styles=css_styles,
                page_width=page_width,
                page_height=page_height,
                reconstruction_metadata=reconstruction_metadata
            )

            logger.info(f"Page reconstruction completed: {len(elements)} elements")
            return reconstructed_page

        except Exception as e:
            logger.error(f"Page reconstruction failed: {e}")
            raise

    def _get_page_dimensions(self, image_path: str,
                             processing_result: ProcessingResult) -> Tuple[int, int]:
        """Get page dimensions from original image or estimate from regions."""
        try:
            if image_path and Path(image_path).exists():
                # Load original image to get dimensions
                image = cv2.imread(image_path)
                if image is not None:
                    height, width = image.shape[:2]
                    return width, height
        except Exception as e:
            logger.debug(f"Could not load original image: {e}")

        # Fallback: estimate from text and image regions
        max_x = max_y = 0

        for region in processing_result.text_regions + processing_result.image_regions:
            max_x = max(max_x, region.x + region.width)
            max_y = max(max_y, region.y + region.height)

        # Add some padding if dimensions were estimated
        if max_x > 0 and max_y > 0:
            return max_x + 100, max_y + 100

        # Default page size (A4 at 150 DPI)
        return 1240, 1754

    def _create_page_elements(self, processing_result: ProcessingResult,
                              page_width: int, page_height: int) -> List[PageElement]:
        """Create page elements from processing results."""
        elements = []

        # Add text elements
        text_elements = self._create_text_elements(processing_result)
        elements.extend(text_elements)

        # Add image elements
        image_elements = self._create_image_elements(processing_result)
        elements.extend(image_elements)

        return elements

    def _create_text_elements(self, processing_result: ProcessingResult) -> List[PageElement]:
        """Create text elements from OCR results."""
        elements = []

        # If we have detailed bounding boxes from OCR, use them
        if (processing_result.best_ocr_result.bounding_boxes and
                len(processing_result.best_ocr_result.bounding_boxes) > 0):

            elements.extend(self._create_elements_from_bounding_boxes(processing_result))

        # Otherwise, use text regions
        elif processing_result.text_regions:
            elements.extend(self._create_elements_from_text_regions(processing_result))

        # Fallback: create single text block
        else:
            elements.append(self._create_single_text_element(processing_result))

        return elements

    def _create_elements_from_bounding_boxes(self,
                                             processing_result: ProcessingResult) -> List[PageElement]:
        """Create elements from detailed OCR bounding boxes."""
        elements = []

        for bbox in processing_result.best_ocr_result.bounding_boxes:
            if bbox.get('text', '').strip():
                # Determine element type based on text characteristics
                element_type = self._classify_text_element(bbox['text'])

                # Apply Japanese markup if needed
                content = bbox['text']
                if processing_result.japanese_result:
                    content = self._apply_japanese_markup(content, processing_result.japanese_result)

                element = PageElement(
                    element_type=element_type,
                    content=content,
                    x=bbox['x'],
                    y=bbox['y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    confidence=bbox.get('confidence', 0) / 100.0,
                    metadata={'original_text': bbox['text']}
                )
                elements.append(element)

        return elements

    def _create_elements_from_text_regions(self,
                                           processing_result: ProcessingResult) -> List[PageElement]:
        """Create elements from text regions."""
        elements = []

        # Split text into lines/paragraphs and distribute across regions
        text_lines = processing_result.cleaned_text.split('\n')
        text_lines = [line.strip() for line in text_lines if line.strip()]

        if not text_lines:
            return elements

        # Distribute text across regions
        lines_per_region = max(1, len(text_lines) // len(processing_result.text_regions))

        for i, region in enumerate(processing_result.text_regions):
            start_idx = i * lines_per_region
            end_idx = min(start_idx + lines_per_region, len(text_lines))

            if start_idx < len(text_lines):
                region_text = '\n'.join(text_lines[start_idx:end_idx])

                # Apply Japanese markup if needed
                if processing_result.japanese_result:
                    region_text = self._apply_japanese_markup(region_text, processing_result.japanese_result)

                element = PageElement(
                    element_type='paragraph',
                    content=region_text,
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    confidence=region.confidence,
                    metadata={'region_type': region.region_type}
                )
                elements.append(element)

        return elements

    def _create_single_text_element(self, processing_result: ProcessingResult) -> PageElement:
        """Create a single text element when no regions are available."""
        content = processing_result.cleaned_text

        # Apply Japanese markup if needed
        if processing_result.japanese_result:
            content = self._apply_japanese_markup(content, processing_result.japanese_result)

        # Estimate dimensions based on content
        estimated_width = min(800, len(content) * 8)  # Rough estimate
        estimated_height = (content.count('\n') + 1) * 20

        return PageElement(
            element_type='paragraph',
            content=content,
            x=40,  # Left margin
            y=40,  # Top margin
            width=estimated_width,
            height=estimated_height,
            confidence=processing_result.overall_confidence,
            metadata={'estimated_dimensions': True}
        )

    def _create_image_elements(self, processing_result: ProcessingResult) -> List[PageElement]:
        """Create image elements from extracted images."""
        elements = []

        for img_info in processing_result.extracted_images:
            # Get region information
            region_data = img_info.get('region', {})

            # Create relative path for web display
            image_path = Path(img_info['path'])
            relative_path = f"extracted_content/{image_path.name}"

            element = PageElement(
                element_type='image',
                content=relative_path,
                x=region_data.get('x', 0),
                y=region_data.get('y', 0),
                width=region_data.get('width', img_info.get('width', 200)),
                height=region_data.get('height', img_info.get('height', 150)),
                confidence=region_data.get('confidence', 0.8),
                metadata={
                    'filename': img_info['filename'],
                    'description': img_info.get('description', ''),
                    'area': img_info.get('area', 0)
                }
            )
            elements.append(element)

        return elements

    def _classify_text_element(self, text: str) -> str:
        """Classify text element type based on content characteristics."""
        text = text.strip()

        # Check for heading patterns
        if (len(text) < 100 and
                (text.isupper() or
                 any(text.startswith(prefix) for prefix in ['CHAPTER', 'SECTION', 'PART']) or
                 text.endswith(':'))):
            return 'heading'

        # Check for short text (likely labels or captions)
        if len(text) < 50:
            return 'text'

        # Default to paragraph for longer text
        return 'paragraph'

    def _apply_japanese_markup(self, text: str, japanese_result: JapaneseProcessingResult) -> str:
        """Apply Japanese text markup for enhanced display."""
        # This would typically call the Japanese processor's markup method
        # For now, we'll do basic Japanese text highlighting
        import re

        # Find Japanese characters and wrap them
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')

        def replace_japanese(match):
            japanese_text = match.group()
            # Find corresponding segment in Japanese result
            for segment in japanese_result.segments:
                if segment.original_text in japanese_text:
                    tooltip_parts = []
                    if segment.romaji:
                        tooltip_parts.append(f"Romaji: {segment.romaji}")
                    if segment.translation:
                        tooltip_parts.append(f"Translation: {segment.translation}")

                    tooltip = " | ".join(tooltip_parts) if tooltip_parts else "Japanese text"

                    return f'<span class="japanese-text" title="{tooltip}" data-confidence="{segment.confidence:.2f}">{japanese_text}</span>'

            return f'<span class="japanese-text">{japanese_text}</span>'

        return japanese_pattern.sub(replace_japanese, text)

    def _sort_elements_by_layout(self, elements: List[PageElement]) -> List[PageElement]:
        """Sort elements by their position on the page (reading order)."""
        # Sort by Y position first (top to bottom), then by X position (left to right)
        return sorted(elements, key=lambda e: (e.y, e.x))

    def _generate_css_styles(self, page_width: int, page_height: int) -> str:
        """Generate CSS styles for the reconstructed page."""
        css = f"""
        .reconstructed-page {{
            width: {page_width}px;
            height: {page_height}px;
            position: relative;
            background: white;
            margin: 0 auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            font-family: {self.typography['font_family']};
            font-size: {self.typography['base_font_size']}px;
            line-height: {self.typography['line_height']};
        }}

        .page-element {{
            position: absolute;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }}

        .page-element:hover {{
            border-color: #007acc;
            background-color: rgba(0, 122, 204, 0.05);
        }}

        .element-heading {{
            font-size: {self.typography['heading_font_size']}px;
            font-weight: bold;
            margin-bottom: {self.layout_config['heading_spacing']}px;
            color: #2c3e50;
        }}

        .element-paragraph {{
            margin-bottom: {self.layout_config['paragraph_spacing']}px;
            text-align: justify;
            color: #34495e;
        }}

        .element-text {{
            color: #2c3e50;
        }}

        .element-image {{
            padding: {self.layout_config['image_padding']}px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }}

        .element-image img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}

        .japanese-text {{
            color: #8e44ad;
            font-weight: 600;
            font-family: {self.typography['japanese_font_family']};
            cursor: help;
            border-bottom: 1px dotted #8e44ad;
        }}

        .japanese-text:hover {{
            background-color: rgba(142, 68, 173, 0.1);
        }}

        .confidence-indicator {{
            position: absolute;
            top: -20px;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        .page-element:hover .confidence-indicator {{
            opacity: 1;
        }}

        .confidence-high {{ background-color: #27ae60 !important; }}
        .confidence-medium {{ background-color: #f39c12 !important; }}
        .confidence-low {{ background-color: #e74c3c !important; }}

        @media print {{
            .reconstructed-page {{
                box-shadow: none;
                margin: 0;
            }}

            .page-element:hover {{
                border-color: transparent;
                background-color: transparent;
            }}

            .confidence-indicator {{
                display: none;
            }}
        }}
        """

        return css

    def _generate_html_content(self, elements: List[PageElement],
                               processing_result: ProcessingResult) -> str:
        """Generate HTML content for the reconstructed page."""
        html_elements = []

        for element in elements:
            element_html = self._create_element_html(element)
            html_elements.append(element_html)

        # Wrap in page container
        elements_html = '\n'.join(html_elements)

        # Add metadata panel
        metadata_panel = self._generate_metadata_panel(processing_result)

        html = f"""
        <div class="reconstructed-page">
            {elements_html}
        </div>
        {metadata_panel}
        """

        return html

    def _create_element_html(self, element: PageElement) -> str:
        """Create HTML for a single page element."""
        # Determine confidence class
        confidence_class = self._get_confidence_class(element.confidence)

        # Create confidence indicator
        confidence_indicator = f"""
        <div class="confidence-indicator {confidence_class}">
            {element.confidence:.0%}
        </div>
        """

        if element.element_type == 'image':
            return f"""
            <div class="page-element element-{element.element_type}" 
                 style="left: {element.x}px; top: {element.y}px; width: {element.width}px; height: {element.height}px;">
                <img src="{element.content}" alt="{element.metadata.get('description', 'Extracted image')}" 
                     title="{element.metadata.get('description', 'Extracted image')}">
                {confidence_indicator}
            </div>
            """
        else:
            return f"""
            <div class="page-element element-{element.element_type}" 
                 style="left: {element.x}px; top: {element.y}px; width: {element.width}px; min-height: {element.height}px;">
                {element.content}
                {confidence_indicator}
            </div>
            """

    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class based on confidence level."""
        if confidence >= 0.8:
            return 'confidence-high'
        elif confidence >= 0.5:
            return 'confidence-medium'
        else:
            return 'confidence-low'

    def _generate_metadata_panel(self, processing_result: ProcessingResult) -> str:
        """Generate metadata panel for the page."""
        stats = processing_result.text_statistics

        japanese_info = ""
        if processing_result.japanese_result:
            japanese_info = f"""
            <div class="metadata-section">
                <h4>Japanese Text Analysis</h4>
                <p>Segments: {len(processing_result.japanese_result.segments)}</p>
                <p>Martial Arts Terms: {len(processing_result.japanese_result.martial_arts_terms)}</p>
                <p>Confidence: {processing_result.japanese_result.confidence_score:.1%}</p>
            </div>
            """

        return f"""
        <div class="metadata-panel" style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3>Document Processing Information</h3>

            <div class="metadata-section">
                <h4>Text Statistics</h4>
                <p>Characters: {stats.get('characters', 0)}</p>
                <p>Words: {stats.get('words', 0)}</p>
                <p>Reading Time: {stats.get('reading_time_minutes', 0):.1f} minutes</p>
            </div>

            {japanese_info}

            <div class="metadata-section">
                <h4>Processing Quality</h4>
                <p>Overall Confidence: {processing_result.overall_confidence:.1%}</p>
                <p>Quality Score: {processing_result.quality_score:.1%}</p>
                <p>Processing Time: {processing_result.processing_time:.2f}s</p>
            </div>

            <div class="metadata-section">
                <h4>Layout Elements</h4>
                <p>Text Regions: {len(processing_result.text_regions)}</p>
                <p>Images Extracted: {len(processing_result.extracted_images)}</p>
                <p>OCR Engine: {processing_result.best_ocr_result.engine}</p>
            </div>
        </div>
        """

    def _generate_page_title(self, processing_result: ProcessingResult) -> str:
        """Generate a title for the page based on content."""
        text = processing_result.cleaned_text

        # Try to find a title in the first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if (line and len(line) < 100 and
                    (line.isupper() or ':' in line or 'CHAPTER' in line.upper())):
                return line

        # Fallback: use first significant line
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line[:50] + ('...' if len(line) > 50 else '')

        # Final fallback
        return f"Document processed {datetime.now().strftime('%Y-%m-%d')}"

    def save_reconstructed_page(self, page: ReconstructedPage, output_dir: str) -> str:
        """Save reconstructed page as HTML file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate complete HTML document
            html_document = self._generate_complete_html_document(page)

            # Save HTML file
            html_file = output_path / f"{page.page_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_document)

            # Save metadata as JSON
            metadata_file = output_path / f"{page.page_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(page.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Reconstructed page saved: {html_file}")
            return str(html_file)

        except Exception as e:
            logger.error(f"Failed to save reconstructed page: {e}")
            raise

    def _generate_complete_html_document(self, page: ReconstructedPage) -> str:
        """Generate complete HTML document with embedded CSS."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{page.title}</title>
            <style>
                {page.css_styles}

                body {{
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}

                .page-container {{
                    max-width: {page.page_width + 100}px;
                    margin: 0 auto;
                }}

                .metadata-panel {{
                    margin-top: 30px;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}

                .metadata-section {{
                    margin-bottom: 20px;
                }}

                .metadata-section h4 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                    font-size: 14px;
                }}

                .metadata-section p {{
                    margin: 5px 0;
                    font-size: 12px;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="page-container">
                <h1>{page.title}</h1>
                {page.html_content}
            </div>
        </body>
        </html>
        """


# Utility functions
def reconstruct_page(processing_result: ProcessingResult,
                     original_image_path: str = None) -> ReconstructedPage:
    """
    Convenient function to reconstruct a page from processing results.

    Args:
        processing_result: Results from OCR processing
        original_image_path: Path to original image

    Returns:
        ReconstructedPage with layout and content
    """
    reconstructor = PageReconstructor()
    return reconstructor.reconstruct_page(processing_result, original_image_path)


def save_page_as_html(page: ReconstructedPage, output_dir: str) -> str:
    """
    Save reconstructed page as HTML file.

    Args:
        page: Reconstructed page
        output_dir: Output directory

    Returns:
        Path to saved HTML file
    """
    reconstructor = PageReconstructor()
    return reconstructor.save_reconstructed_page(page, output_dir)


if __name__ == "__main__":
    # Test the page reconstructor
    print("Page Reconstructor Test")
    print("=" * 50)

    # This would typically be used with actual processing results
    print("Page reconstructor module loaded successfully")
    print("Use reconstruct_page() function with OCR processing results")