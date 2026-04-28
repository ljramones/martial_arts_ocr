"""
Data models for OCR processing results.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class OCRResult:
    """Results from OCR processing."""
    text: str
    confidence: float
    processing_time: float
    engine: str
    bounding_boxes: List[Dict] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'engine': self.engine,
            'bounding_boxes': self.bounding_boxes or [],
            'metadata': self.metadata or {}
        }


@dataclass
class ProcessingResult:
    """Complete processing result for a document."""
    document_id: Optional[int]
    page_id: Optional[int]

    # OCR results
    ocr_results: List[OCRResult]
    best_ocr_result: OCRResult
    raw_text: str
    cleaned_text: str

    # Layout analysis
    text_regions: List['ImageRegion']
    image_regions: List['ImageRegion']
    extracted_images: List[Dict[str, Any]]

    # Language processing
    japanese_result: Optional['JapaneseProcessingResult']
    language_segments: List[Dict]
    text_statistics: Dict[str, Any]

    # Quality metrics
    overall_confidence: float
    quality_score: float
    processing_time: float

    # Output formats
    html_content: str
    markdown_content: str

    # Metadata
    processing_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'page_id': self.page_id,
            'ocr_results': [result.to_dict() for result in self.ocr_results],
            'best_ocr_result': self.best_ocr_result.to_dict(),
            'raw_text': self.raw_text,
            'cleaned_text': self.cleaned_text,
            'text_regions': [region.to_dict() for region in self.text_regions],
            'image_regions': [region.to_dict() for region in self.image_regions],
            'extracted_images': self.extracted_images,
            'japanese_result': self.japanese_result.to_dict() if self.japanese_result else None,
            'language_segments': self.language_segments,
            'text_statistics': self.text_statistics,
            'overall_confidence': self.overall_confidence,
            'quality_score': self.quality_score,
            'processing_time': self.processing_time,
            'html_content': self.html_content,
            'markdown_content': self.markdown_content,
            'processing_metadata': self.processing_metadata
        }