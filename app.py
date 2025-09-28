"""
Main Flask application for Martial Arts OCR.
Handles file uploads, OCR processing, and web interface.
"""
import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from datetime import datetime

# Local imports
from config import get_config, allowed_file, get_upload_path, get_processed_path
from database import init_db, get_db_session
from models import Document, Page, ProcessingResult
from processors.ocr_processor import OCRProcessor
from processors.layout_detector import LayoutDetector
from processors.japanese_processor import JapaneseProcessor

# Initialize Flask app
app = Flask(__name__)
config = get_config()
app.config.from_object(config)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
ocr_processor = OCRProcessor()
layout_detector = LayoutDetector()
japanese_processor = JapaneseProcessor()

# Initialize database
with app.app_context():
    init_db()


@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"

            filepath = get_upload_path(filename)
            file.save(filepath)

            # Create database record
            with get_db_session() as session:
                document = Document(
                    filename=filename,
                    original_filename=file.filename,
                    upload_date=datetime.now(),
                    status='uploaded'
                )
                session.add(document)
                session.commit()
                document_id = document.id

            flash(f'File "{file.filename}" uploaded successfully', 'success')
            return redirect(url_for('process_document', document_id=document_id))

        else:
            flash('Invalid file type. Please upload JPG, PNG, or TIFF files.', 'error')
            return redirect(request.url)

    except RequestEntityTooLarge:
        flash('File too large. Maximum size is 16MB.', 'error')
        return redirect(request.url)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        flash('Upload failed. Please try again.', 'error')
        return redirect(request.url)


@app.route('/process/<int:document_id>')
def process_document(document_id):
    """Process a document through OCR pipeline."""
    try:
        with get_db_session() as session:
            document = session.get(Document, document_id)
            if not document:
                flash('Document not found', 'error')
                return redirect(url_for('index'))

            # Update status
            document.status = 'processing'
            session.commit()

        # Get file path
        filepath = get_upload_path(document.filename)
        if not filepath.exists():
            flash('File not found', 'error')
            return redirect(url_for('index'))

        # Process the document
        result = process_image_file(str(filepath), document_id)

        if result['success']:
            with get_db_session() as session:
                doc = session.get(Document, document_id)
                doc.status = 'completed'
                doc.processing_date = datetime.now()
                session.commit()

            flash('Document processed successfully', 'success')
            return redirect(url_for('view_document', document_id=document_id))
        else:
            with get_db_session() as session:
                doc = session.get(Document, document_id)
                doc.status = 'failed'
                doc.error_message = result.get('error', 'Unknown error')
                session.commit()

            flash(f'Processing failed: {result.get("error", "Unknown error")}', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Processing error: {e}")
        flash('Processing failed. Please try again.', 'error')
        return redirect(url_for('index'))


def process_image_file(filepath: str, document_id: int) -> dict:
    """Process a single image file through the OCR pipeline."""
    try:
        logger.info(f"Processing file: {filepath}")

        # Step 1: Layout detection
        logger.info("Detecting layout...")
        layout_result = layout_detector.analyze_layout(filepath)

        # Step 2: OCR processing
        logger.info("Running OCR...")
        ocr_result = ocr_processor.process_image(filepath, layout_result)

        # Step 3: Japanese text processing (if any Japanese detected)
        japanese_result = None
        if ocr_result.has_japanese:
            logger.info("Processing Japanese text...")
            japanese_result = japanese_processor.process_text(ocr_result.japanese_text)

        # Step 4: Save results to database
        with get_db_session() as session:
            # Create page record
            page = Page(
                document_id=document_id,
                page_number=1,  # Single page for now
                image_path=filepath,
                layout_data=layout_result.to_dict(),
                processing_date=datetime.now()
            )
            session.add(page)
            session.flush()  # Get the page ID

            # Create processing result
            result = ProcessingResult(
                page_id=page.id,
                ocr_text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
                has_japanese=ocr_result.has_japanese,
                japanese_data=japanese_result.to_dict() if japanese_result else None,
                extracted_images=ocr_result.extracted_images,
                processing_time=ocr_result.processing_time
            )
            session.add(result)
            session.commit()

        # Step 5: Generate output files
        output_dir = get_processed_path(f"doc_{document_id}")
        output_dir.mkdir(exist_ok=True)

        # Save HTML version
        html_content = generate_html_output(ocr_result, japanese_result, layout_result)
        with open(output_dir / "page_1.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Save JSON version
        json_data = {
            'document_id': document_id,
            'ocr_result': ocr_result.to_dict(),
            'japanese_result': japanese_result.to_dict() if japanese_result else None,
            'layout_result': layout_result.to_dict(),
            'processing_date': datetime.now().isoformat()
        }

        import json
        with open(output_dir / "data.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        logger.info("Processing completed successfully")
        return {'success': True}

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {'success': False, 'error': str(e)}


def generate_html_output(ocr_result, japanese_result, layout_result) -> str:
    """Generate HTML output with preserved layout."""
    # This is a simplified version - will be enhanced later
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .japanese {{ color: #0066cc; }}
            .romaji {{ color: #666; font-style: italic; }}
            .translation {{ color: #008800; }}
            .image-placeholder {{ 
                border: 2px dashed #ccc; 
                padding: 20px; 
                margin: 10px 0; 
                text-align: center; 
            }}
        </style>
    </head>
    <body>
        <h1>OCR Processing Result</h1>
        <div class="content">
            <h2>Extracted Text</h2>
            <div class="text-content">
                {format_text_with_japanese(ocr_result.text, japanese_result)}
            </div>

            {generate_image_placeholders(ocr_result.extracted_images)}

            <div class="metadata">
                <h3>Processing Information</h3>
                <p>Confidence: {ocr_result.confidence:.2f}%</p>
                <p>Processing Time: {ocr_result.processing_time:.2f}s</p>
                <p>Japanese Text Detected: {'Yes' if ocr_result.has_japanese else 'No'}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html


def format_text_with_japanese(text: str, japanese_result) -> str:
    """Format text with Japanese annotations."""
    # Simplified formatting - will be enhanced
    if japanese_result:
        # Add romaji and translation as tooltips or footnotes
        formatted = text.replace('\n', '<br>')
        return f'<div class="japanese">{formatted}</div>'
    else:
        return text.replace('\n', '<br>')


def generate_image_placeholders(extracted_images: list) -> str:
    """Generate placeholders for extracted images."""
    if not extracted_images:
        return ""

    html = "<h3>Extracted Images</h3>"
    for i, img_info in enumerate(extracted_images):
        html += f"""
        <div class="image-placeholder">
            <p>Image {i + 1}: {img_info.get('type', 'Unknown')}</p>
            <p>Position: ({img_info.get('x', 0)}, {img_info.get('y', 0)})</p>
            <p>Size: {img_info.get('width', 0)} x {img_info.get('height', 0)}</p>
        </div>
        """
    return html


@app.route('/view/<int:document_id>')
def view_document(document_id):
    """View processed document."""
    try:
        with get_db_session() as session:
            document = session.get(Document, document_id)
            if not document:
                flash('Document not found', 'error')
                return redirect(url_for('index'))

            # Get the first page (single page for now)
            page = session.query(Page).filter_by(document_id=document_id).first()
            result = None
            if page:
                result = session.query(ProcessingResult).filter_by(page_id=page.id).first()

        return render_template('page_view.html',
                               document=document,
                               page=page,
                               result=result)

    except Exception as e:
        logger.error(f"View error: {e}")
        flash('Error loading document', 'error')
        return redirect(url_for('index'))


@app.route('/gallery')
def gallery():
    """View all processed documents."""
    try:
        with get_db_session() as session:
            documents = session.query(Document).order_by(Document.upload_date.desc()).all()

        return render_template('gallery.html', documents=documents)

    except Exception as e:
        logger.error(f"Gallery error: {e}")
        flash('Error loading gallery', 'error')
        return redirect(url_for('index'))


@app.route('/api/status/<int:document_id>')
def api_document_status(document_id):
    """API endpoint to check document processing status."""
    try:
        with get_db_session() as session:
            document = session.get(Document, document_id)
            if not document:
                return jsonify({'error': 'Document not found'}), 404

            return jsonify({
                'id': document.id,
                'status': document.status,
                'filename': document.original_filename,
                'upload_date': document.upload_date.isoformat() if document.upload_date else None,
                'processing_date': document.processing_date.isoformat() if document.processing_date else None,
                'error_message': document.error_message
            })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('500.html'), 500


def main():
    """Main entry point."""
    # Ensure necessary directories exist
    for directory in [config.UPLOAD_FOLDER,
                      get_processed_path(""),
                      app.static_folder + "/extracted_content"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Run the application
    logger.info(f"Starting Martial Arts OCR application")
    logger.info(f"Upload directory: {config.UPLOAD_FOLDER}")
    logger.info(f"Debug mode: {config.DEBUG}")

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )


if __name__ == '__main__':
    main()