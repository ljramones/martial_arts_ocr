"""
Main Flask application for Martial Arts OCR.
Handles file uploads, OCR processing, and web interface.
"""
import os
import json
from pathlib import Path
from threading import Thread

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from jinja2 import TemplateNotFound
import logging
from datetime import datetime
from PIL import Image

# Local imports
from config import get_config, allowed_file, get_upload_path, get_processed_path
from database import init_db, get_db_session
from models import Document, Page, ProcessingResult
from processors.ocr_processor import OCRProcessor
from processors.content_extractor import ContentExtractor
from processors.japanese_processor import JapaneseProcessor
from processors.page_reconstructor import PageReconstructor

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
try:
    ocr_processor = OCRProcessor()
    content_extractor = ContentExtractor()
    japanese_processor = JapaneseProcessor()
    page_reconstructor = PageReconstructor()
    logger.info("All processors initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize processors: {e}")
    raise

# Initialize database
with app.app_context():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# -------------------------
# Template context / filters
# -------------------------
@app.context_processor
def inject_globals():
    """Inject global variables into all templates."""
    return {
        'config': config,
        'current_year': datetime.now().year,
    }

@app.template_filter('filesizeformat')
def filesizeformat(value):
    """Format file size in human readable format."""
    if value is None:
        return "Unknown"
    try:
        bytes_val = float(value)
        if bytes_val < 1024:
            return f"{bytes_val:.0f} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_val / (1024 * 1024 * 1024):.1f} GB"
    except (ValueError, TypeError):
        return "Unknown"

# -------------------------
# Host enforcement (dev)
# -------------------------
@app.before_request
def enforce_allowed_hosts():
    allowed = getattr(app.config, "ALLOWED_HOSTS", None)
    if not allowed:
        return  # no restriction
    host = request.headers.get("Host", "")
    host_only = host.split(":", 1)[0].strip().lower()
    if host_only.startswith('[') and host_only.endswith(']'):
        host_only = host_only[1:-1]  # [::1] -> ::1
    if host_only not in allowed:
        app.logger.warning("Denied host %s (allowed: %s)", host_only, sorted(allowed))
        abort(403)

# -------------------------
# Health
# -------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}, 200

# -------------------------
# Background processing
# -------------------------
def _kickoff_processing_async(document_id: int):
    """Start processing in a background thread (no HTTP redirect)."""
    def _run():
        try:
            with app.app_context():
                # Read filename and set status to 'processing'
                with get_db_session() as session:
                    doc = session.get(Document, document_id)
                    if not doc:
                        logger.error("Document %s not found", document_id)
                        return
                    doc.status = "processing"
                    session.commit()
                    filename = doc.filename

                filepath = get_upload_path(filename)
                result = process_image_file(str(filepath), document_id)

                with get_db_session() as session:
                    doc = session.get(Document, document_id)
                    if not doc:
                        return
                    if result.get('success'):
                        doc.status = 'completed'
                        doc.processing_date = datetime.now()
                        doc.ocr_engine = result.get('ocr_engine', 'unknown')
                    else:
                        doc.status = 'failed'
                        doc.error_message = result.get('error', 'Unknown error')
                    session.commit()

        except Exception as e:
            logger.exception("Background processing error for doc %s", document_id)
            try:
                with get_db_session() as session:
                    doc = session.get(Document, document_id)
                    if doc:
                        doc.status = "failed"
                        doc.error_message = str(e)
                        session.commit()
            except Exception:
                logger.exception("Failed to record processing error in DB")
    Thread(target=_run, daemon=True).start()

# -------------------------
# Status API (single route)
# -------------------------
@app.get("/api/status/<int:document_id>")
def api_status(document_id):
    try:
        with get_db_session() as db:
            doc = db.get(Document, document_id)
            if not doc:
                return jsonify({"status": "unknown", "error_message": "Not found"}), 404
            return jsonify({
                "status": doc.status,
                "error_message": doc.error_message,
            }), 200
    except Exception as e:
        logger.error("Status API error: %s", e)
        return jsonify({"status": "unknown", "error_message": "Internal error"}), 500

# -------------------------
# Pages
# -------------------------
@app.route('/')
def index():
    """Main upload page with recent documents."""
    try:
        with get_db_session() as session:
            docs = (session.query(Document)
                    .order_by(Document.upload_date.desc())
                    .limit(5)
                    .all())
            # Materialize minimal fields to avoid detached refresh
            recent_documents = [{
                "id": d.id,
                "original_filename": d.original_filename,
                "upload_date": d.upload_date,
                "file_size": d.file_size,
                "status": d.status,
            } for d in docs]

        return render_template('index.html', recent_documents=recent_documents)

    except Exception as e:
        logger.error(f"Index page error: {e}")
        flash('Error loading page', 'error')
        return render_template('index.html', recent_documents=[])

# -------------------------
# Upload (JSON, starts background processing)
# -------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload returning JSON and starting background processing."""
    try:
        file = request.files.get('file')
        if not file or file.filename.strip() == '':
            return jsonify({"success": False, "error": "No file provided"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload JPG, PNG, TIFF, or BMP files."
            }), 400

        # Secure filename + timestamp to avoid collisions
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"

        # Save file
        filepath = get_upload_path(filename)
        file.save(filepath)

        # Image metadata (best-effort)
        width = height = None
        img_format = None
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                img_format = img.format
        except Exception as e:
            logger.warning("Could not read image metadata for %s: %s", filename, e)

        # Size
        file_size = filepath.stat().st_size

        # Create DB record
        with get_db_session() as session:
            document = Document(
                filename=filename,
                original_filename=file.filename,
                file_size=file_size,
                upload_date=datetime.now(),
                status='uploaded',
                image_width=width,
                image_height=height,
                image_format=img_format
            )
            session.add(document)
            session.commit()
            document_id = document.id

        logger.info('File uploaded successfully: %s -> %s', file.filename, filename)

        # Kick off processing in the background (no redirect)
        _kickoff_processing_async(document_id)

        # Frontend expects JSON: { success, documentId }
        return jsonify({"success": True, "documentId": document_id}), 200

    except RequestEntityTooLarge:
        return jsonify({"success": False, "error": "File too large. Maximum size is 16MB."}), 413

    except Exception as e:
        logger.exception("Upload error")
        return jsonify({"success": False, "error": "Upload failed. Please try again."}), 500

# -------------------------
# (Optional) Synchronous processing route (manual)
# -------------------------
@app.route('/process/<int:document_id>')
def process_document(document_id):
    """Process a document through OCR pipeline synchronously (UI-triggered)."""
    try:
        with get_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc:
                flash('Document not found', 'error')
                return redirect(url_for('index'))
            doc.status = 'processing'
            session.commit()
            filename = doc.filename  # capture before session closes

        filepath = get_upload_path(filename)
        if not filepath.exists():
            flash('File not found', 'error')
            return redirect(url_for('index'))

        result = process_image_file(str(filepath), document_id)

        with get_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc:
                flash('Document not found', 'error')
                return redirect(url_for('index'))
            if result['success']:
                doc.status = 'completed'
                doc.processing_date = datetime.now()
                doc.ocr_engine = result.get('ocr_engine', 'unknown')
                session.commit()
                flash('Document processed successfully', 'success')
                return redirect(url_for('view_document', document_id=document_id))
            else:
                doc.status = 'failed'
                doc.error_message = result.get('error', 'Unknown error')
                session.commit()
                flash(f'Processing failed: {result.get("error", "Unknown error")}', 'error')
                return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Processing error: {e}")
        flash('Processing failed. Please try again.', 'error')
        return redirect(url_for('index'))

# -------------------------
# Core processing function
# -------------------------
def process_image_file(filepath: str, document_id: int) -> dict:
    """Process image file through the unified OCR pipeline."""
    try:
        logger.info(f"Processing file: {filepath}")

        # Use the unified OCR processor
        processing_result = ocr_processor.process_document(filepath, document_id)

        # Save results to database
        with get_db_session() as session:
            page = Page(
                document_id=document_id,
                page_number=1,
                image_path=filepath,
                image_width=processing_result.processing_metadata.get('image_dimensions', {}).get('width'),
                image_height=processing_result.processing_metadata.get('image_dimensions', {}).get('height'),
                processing_time=processing_result.processing_time,
                ocr_confidence=processing_result.overall_confidence,
                text_regions=[region.to_dict() for region in processing_result.text_regions],
                image_regions=[region.to_dict() for region in processing_result.image_regions]
            )
            session.add(page)
            session.flush()

            db_result = ProcessingResult(
                document_id=document_id,
                page_id=page.id,
                ocr_engine_used=processing_result.best_ocr_result.engine,
                processing_time=processing_result.processing_time,
                raw_ocr_text=processing_result.raw_text,
                cleaned_text=processing_result.cleaned_text,
                ocr_confidence=processing_result.overall_confidence,
                ocr_metadata=processing_result.best_ocr_result.metadata,
                has_japanese=processing_result.japanese_result is not None,
                japanese_segments=[seg.to_dict() for seg in processing_result.japanese_result.segments] if processing_result.japanese_result else None,
                language_analysis=processing_result.japanese_result.language_analysis if processing_result.japanese_result else None,
                martial_arts_terms=processing_result.japanese_result.martial_arts_terms if processing_result.japanese_result else None,
                overall_romaji=processing_result.japanese_result.overall_romaji if processing_result.japanese_result else None,
                overall_translation=processing_result.japanese_result.overall_translation if processing_result.japanese_result else None,
                japanese_confidence=processing_result.japanese_result.confidence_score if processing_result.japanese_result else None,
                japanese_metadata=processing_result.japanese_result.processing_metadata if processing_result.japanese_result else None,
                html_content=processing_result.html_content,
                markdown_content=processing_result.markdown_content,
                extracted_images=processing_result.extracted_images,
                text_statistics=processing_result.text_statistics,
                quality_score=processing_result.quality_score,
                confidence_breakdown={
                    'ocr_confidence': processing_result.overall_confidence,
                    'quality_score': processing_result.quality_score,
                    'japanese_confidence': processing_result.japanese_result.confidence_score if processing_result.japanese_result else None
                }
            )
            session.add(db_result)
            session.commit()

        # Generate output files
        output_dir = get_processed_path(f"doc_{document_id}")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save reconstructed page data for frontend
        try:
            reconstructed_page = page_reconstructor.reconstruct_page(processing_result, filepath)
            with open(output_dir / "page_data.json", 'w', encoding='utf-8') as f:
                json.dump(reconstructed_page.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("Page data saved for frontend rendering")
        except Exception as e:
            logger.warning(f"Page reconstruction failed: {e}")

        # Save complete processing data
        json_data = processing_result.to_dict()
        json_data['document_id'] = document_id
        json_data['processing_date'] = datetime.now().isoformat()
        with open(output_dir / "data.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # Also write a stable HTML artifact for the download route
        try:
            html = processing_result.html_content
            if html:
                (output_dir / "page_1.html").write_text(html, encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to write page_1.html: {e}")

        logger.info("Processing completed successfully")
        return {
            'success': True,
            'ocr_engine': processing_result.best_ocr_result.engine,
            'confidence': processing_result.overall_confidence,
            'has_japanese': processing_result.japanese_result is not None
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

# -------------------------
# Gallery
# -------------------------
@app.route('/gallery')
def gallery():
    """Document gallery with filtering."""
    try:
        status_filter = request.args.get('status')
        with get_db_session() as session:
            q = session.query(Document).order_by(Document.upload_date.desc())
            if status_filter:
                q = q.filter(Document.status == status_filter)
            docs = q.all()
            documents = [{
                "id": d.id,
                "original_filename": d.original_filename,
                "filename": d.filename,
                "upload_date": d.upload_date,
                "processing_date": d.processing_date,
                "status": d.status,
                "file_size": d.file_size,
                "error_message": d.error_message,
            } for d in docs]

        return render_template('gallery.html', documents=documents, pagination=None)

    except Exception as e:
        logger.error(f"Gallery error: {e}")
        flash('Error loading gallery', 'error')
        return render_template('gallery.html', documents=[], pagination=None)

# -------------------------
# Viewer
# -------------------------
@app.route('/view/<int:document_id>')
def view_document(document_id):
    """View processed document using sophisticated frontend templates."""
    try:
        with get_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc:
                flash('Document not found', 'error')
                return redirect(url_for('gallery'))
            document = {
                "id": doc.id,
                "original_filename": doc.original_filename,
                "upload_date": doc.upload_date,
                "processing_date": doc.processing_date,
                "status": doc.status,
                "file_size": doc.file_size,
            }
            page = session.query(Page).filter_by(document_id=document_id).first()
            result = session.query(ProcessingResult).filter_by(page_id=page.id).first() if page else None

        # Load saved processing data for frontend
        output_dir = get_processed_path(f"doc_{document_id}")
        page_data = None
        processing_data = None
        page_data_file = output_dir / "page_data.json"
        data_file = output_dir / "data.json"
        if page_data_file.exists():
            try:
                page_data = json.load(open(page_data_file, 'r', encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Could not load page data: {e}")
        if data_file.exists():
            try:
                processing_data = json.load(open(data_file, 'r', encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Could not load processing data: {e}")

        return render_template('page_view.html',
                               document=document,
                               page=page,
                               result=result,
                               page_data=page_data,
                               processing_data=processing_data)

    except Exception as e:
        logger.error(f"View error: {e}")
        flash('Error loading document', 'error')
        return redirect(url_for('gallery'))

# -------------------------
# Engine status / JP helpers
# -------------------------
@app.route('/api/engines/status')
def api_engines_status():
    """API endpoint for OCR engine status."""
    try:
        status = ocr_processor.get_engine_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Engine status error: {e}")
        return jsonify({'error': 'Failed to get engine status'}), 500

@app.route('/api/romanize', methods=['POST'])
def api_romanize():
    """API endpoint for Japanese romanization."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        romaji = japanese_processor.romanize_text_simple(text)
        return jsonify({'romaji': romaji})
    except Exception as e:
        logger.error(f"Romanization error: {e}")
        return jsonify({'error': 'Romanization failed'}), 500

@app.route('/api/translate', methods=['POST'])
def api_translate():
    """API endpoint for Japanese translation."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = japanese_processor.process_text(text)
        translation = result.overall_translation if result else None
        return jsonify({'translation': translation})
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed'}), 500

# -------------------------
# Document management
# -------------------------
@app.route('/retry/<int:document_id>', methods=['POST'])
def retry_processing(document_id):
    """Reset a failed document to 'uploaded' to allow reprocessing."""
    try:
        with get_db_session() as session:
            document = session.get(Document, document_id)
            if not document:
                return jsonify({'error': 'Document not found'}), 404
            document.status = 'uploaded'
            document.error_message = None
            session.commit()
        # Optionally kick off processing immediately:
        # _kickoff_processing_async(document_id)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Retry processing error: {e}")
        return jsonify({'error': 'Failed to retry processing'}), 500

@app.route('/download/<int:document_id>')
def download_document(document_id):
    """Download processed HTML artifact."""
    try:
        with get_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc or doc.status != 'completed':
                flash('Document not available for download', 'error')
                return redirect(url_for('gallery'))
            original = doc.original_filename  # capture before close

        output_dir = get_processed_path(f"doc_{document_id}")
        html_file = output_dir / "page_1.html"
        if html_file.exists():
            safe_name = f"{Path(original).stem}_processed.html"
            return send_file(html_file, as_attachment=True, download_name=safe_name)
        else:
            flash('Processed file not found', 'error')
            return redirect(url_for('view_document', document_id=document_id))

    except Exception as e:
        logger.error(f"Download error: {e}")
        flash('Download failed', 'error')
        return redirect(url_for('gallery'))

# -------------------------
# Error handlers (safe fallbacks)
# -------------------------
@app.errorhandler(404)
def not_found_error(error):
    try:
        return render_template('404.html'), 404
    except TemplateNotFound:
        return "Not Found", 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    try:
        return render_template('500.html'), 500
    except TemplateNotFound:
        return "Internal Server Error", 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    # For form submits this is fine; /upload returns JSON itself for this case
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

# -------------------------
# Entrypoint
# -------------------------
def main():
    """Main entry point."""
    # Verify template directory exists
    template_dir = Path(app.template_folder)
    if not template_dir.exists():
        logger.error(f"Templates directory not found: {template_dir}")
        logger.error("Please ensure the templates/ directory exists with base.html, index.html, etc.")
        return

    # Verify static directory exists
    static_dir = Path(app.static_folder) if app.static_folder else Path("static")
    if not static_dir.exists():
        logger.warning(f"Static directory not found: {static_dir}")
        static_dir.mkdir(parents=True, exist_ok=True)

    # Ensure necessary directories exist
    directories = [
        config.UPLOAD_FOLDER,
        get_processed_path(""),
        static_dir / "extracted_content"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Log startup information
    logger.info("Starting Martial Arts OCR application")
    logger.info(f"Upload directory: {config.UPLOAD_FOLDER}")
    logger.info(f"Processed directory: {get_processed_path('')}")
    logger.info(f"Debug mode: {config.DEBUG}")

    # Check OCR engine status
    try:
        engine_status = ocr_processor.get_engine_status()
        logger.info(f"Tesseract available: {engine_status['tesseract']['available']}")
        logger.info(f"EasyOCR available: {engine_status['easyocr']['available']}")
        logger.info(f"Japanese processor ready: {engine_status['japanese_processor']['available']}")
    except Exception as e:
        logger.warning(f"Could not check engine status: {e}")

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )

if __name__ == '__main__':
    main()
