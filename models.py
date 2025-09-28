"""
Database models for Martial Arts OCR application.
Defines the data structures for documents, pages, and processing results.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

from database import Base


class Document(Base):
    """Main document record representing an uploaded file."""
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False, unique=True)  # Stored filename
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime, default=func.now(), nullable=False)
    processing_date = Column(DateTime, nullable=True)

    # Processing status: 'uploaded', 'processing', 'completed', 'failed'
    status = Column(String(20), default='uploaded', nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # File metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    image_format = Column(String(10), nullable=True)

    # Processing configuration used
    ocr_engine = Column(String(20), nullable=True)  # 'tesseract', 'easyocr', 'both'
    processing_config = Column(JSON, nullable=True)

    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    processing_results = relationship("ProcessingResult", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.original_filename}', status='{self.status}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'original_filename': self.original_filename,
            'filename': self.filename,
            'file_size': self.file_size,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'processing_date': self.processing_date.isoformat() if self.processing_date else None,
            'status': self.status,
            'error_message': self.error_message,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_format': self.image_format,
            'ocr_engine': self.ocr_engine,
            'processing_config': self.processing_config,
            'page_count': len(self.pages) if self.pages else 0
        }

    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        return self.status == 'completed'

    @property
    def has_error(self) -> bool:
        """Check if document processing failed."""
        return self.status == 'failed'

    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.status == 'processing'


class Page(Base):
    """Individual page within a document (for multi-page documents)."""
    __tablename__ = 'pages'

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)  # 1-indexed

    # Page image information
    image_path = Column(String(500), nullable=True)  # Path to extracted page image
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)

    # Processing metadata
    processing_time = Column(Float, nullable=True)  # Seconds
    ocr_confidence = Column(Float, nullable=True)  # 0-1

    # Detected regions
    text_regions = Column(JSON, nullable=True)  # List of text region coordinates
    image_regions = Column(JSON, nullable=True)  # List of image region coordinates

    # Relationships
    document = relationship("Document", back_populates="pages")
    processing_result = relationship("ProcessingResult", back_populates="page", uselist=False)

    def __repr__(self):
        return f"<Page(id={self.id}, document_id={self.document_id}, page_number={self.page_number})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'page_number': self.page_number,
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'processing_time': self.processing_time,
            'ocr_confidence': self.ocr_confidence,
            'text_regions': self.text_regions,
            'image_regions': self.image_regions
        }


class ProcessingResult(Base):
    """Results from OCR and Japanese text processing."""
    __tablename__ = 'processing_results'

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)
    page_id = Column(Integer, ForeignKey('pages.id'), nullable=True, index=True)

    # Processing metadata
    processing_date = Column(DateTime, default=func.now(), nullable=False)
    ocr_engine_used = Column(String(20), nullable=False)  # 'tesseract', 'easyocr'
    processing_time = Column(Float, nullable=False)  # Total processing time in seconds

    # OCR results
    raw_ocr_text = Column(Text, nullable=True)  # Raw OCR output
    cleaned_text = Column(Text, nullable=True)  # Cleaned OCR text
    ocr_confidence = Column(Float, nullable=True)  # Overall OCR confidence (0-1)
    ocr_metadata = Column(JSON, nullable=True)  # Engine-specific metadata

    # Text processing results
    has_japanese = Column(Boolean, default=False, nullable=False, index=True)
    japanese_segments = Column(JSON, nullable=True)  # Japanese text segments
    language_analysis = Column(JSON, nullable=True)  # Language composition analysis
    martial_arts_terms = Column(JSON, nullable=True)  # Detected martial arts terminology

    # Japanese processing results
    overall_romaji = Column(Text, nullable=True)  # Overall romanization
    overall_translation = Column(Text, nullable=True)  # Overall translation
    japanese_confidence = Column(Float, nullable=True)  # Japanese processing confidence
    japanese_metadata = Column(JSON, nullable=True)  # Japanese processing metadata

    # Output formats
    html_content = Column(Text, nullable=True)  # HTML with markup
    markdown_content = Column(Text, nullable=True)  # Markdown format

    # Extracted content
    extracted_images = Column(JSON, nullable=True)  # List of extracted image info
    text_statistics = Column(JSON, nullable=True)  # Text analysis statistics

    # Quality metrics
    quality_score = Column(Float, nullable=True)  # Overall quality assessment (0-1)
    confidence_breakdown = Column(JSON, nullable=True)  # Detailed confidence metrics

    # Relationships
    document = relationship("Document", back_populates="processing_results")
    page = relationship("Page", back_populates="processing_result")

    def __repr__(self):
        return f"<ProcessingResult(id={self.id}, document_id={self.document_id}, engine='{self.ocr_engine_used}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'page_id': self.page_id,
            'processing_date': self.processing_date.isoformat() if self.processing_date else None,
            'ocr_engine_used': self.ocr_engine_used,
            'processing_time': self.processing_time,
            'raw_ocr_text': self.raw_ocr_text,
            'cleaned_text': self.cleaned_text,
            'ocr_confidence': self.ocr_confidence,
            'ocr_metadata': self.ocr_metadata,
            'has_japanese': self.has_japanese,
            'japanese_segments': self.japanese_segments,
            'language_analysis': self.language_analysis,
            'martial_arts_terms': self.martial_arts_terms,
            'overall_romaji': self.overall_romaji,
            'overall_translation': self.overall_translation,
            'japanese_confidence': self.japanese_confidence,
            'japanese_metadata': self.japanese_metadata,
            'html_content': self.html_content,
            'markdown_content': self.markdown_content,
            'extracted_images': self.extracted_images,
            'text_statistics': self.text_statistics,
            'quality_score': self.quality_score,
            'confidence_breakdown': self.confidence_breakdown
        }

    @property
    def text_content(self) -> str:
        """Get the best available text content."""
        return self.cleaned_text or self.raw_ocr_text or ""

    @property
    def has_high_confidence(self) -> bool:
        """Check if the result has high confidence scores."""
        return (self.ocr_confidence or 0) > 0.8 and (self.quality_score or 0) > 0.8

    def get_japanese_segments_list(self) -> List[Dict]:
        """Get Japanese segments as a list of dictionaries."""
        if not self.japanese_segments:
            return []
        if isinstance(self.japanese_segments, str):
            try:
                return json.loads(self.japanese_segments)
            except json.JSONDecodeError:
                return []
        return self.japanese_segments or []

    def get_martial_arts_terms_list(self) -> List[Dict]:
        """Get martial arts terms as a list of dictionaries."""
        if not self.martial_arts_terms:
            return []
        if isinstance(self.martial_arts_terms, str):
            try:
                return json.loads(self.martial_arts_terms)
            except json.JSONDecodeError:
                return []
        return self.martial_arts_terms or []

    def get_extracted_images_list(self) -> List[Dict]:
        """Get extracted images as a list of dictionaries."""
        if not self.extracted_images:
            return []
        if isinstance(self.extracted_images, str):
            try:
                return json.loads(self.extracted_images)
            except json.JSONDecodeError:
                return []
        return self.extracted_images or []


# Utility functions for working with models
def create_document(filename: str, original_filename: str, file_size: int,
                    image_width: int = None, image_height: int = None,
                    image_format: str = None) -> Document:
    """Create a new document record."""
    return Document(
        filename=filename,
        original_filename=original_filename,
        file_size=file_size,
        image_width=image_width,
        image_height=image_height,
        image_format=image_format
    )


def create_page(document_id: int, page_number: int, image_path: str = None,
                image_width: int = None, image_height: int = None) -> Page:
    """Create a new page record."""
    return Page(
        document_id=document_id,
        page_number=page_number,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height
    )


def create_processing_result(document_id: int, ocr_engine_used: str,
                             processing_time: float, page_id: int = None) -> ProcessingResult:
    """Create a new processing result record."""
    return ProcessingResult(
        document_id=document_id,
        page_id=page_id,
        ocr_engine_used=ocr_engine_used,
        processing_time=processing_time
    )


# Model mixins for common functionality
class TimestampMixin:
    """Mixin for models that need created/updated timestamps."""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class SoftDeleteMixin:
    """Mixin for models that support soft deletion."""
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)

    def soft_delete(self):
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.now()

    def restore(self):
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


# Database event listeners for automatic updates
from sqlalchemy import event


@event.listens_for(Document, 'before_update')
def document_before_update(mapper, connection, target):
    """Automatically update processing_date when status changes to completed."""
    if target.status == 'completed' and not target.processing_date:
        target.processing_date = datetime.now()


@event.listens_for(ProcessingResult, 'before_insert')
def processing_result_before_insert(mapper, connection, target):
    """Automatically update document status when processing result is created."""
    # This would need to be handled in the application layer to avoid circular imports
    pass