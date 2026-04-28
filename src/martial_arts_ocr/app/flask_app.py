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

from martial_arts_ocr.config import (
    allowed_file,
    configure_runtime_paths,
    get_config,
    get_processed_path,
    get_upload_path,
)
from martial_arts_ocr.db.database import configure_database, get_db_session, init_db
from martial_arts_ocr.db.models import Document, Page, ProcessingResult
from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator

config = get_config()
# Initialize Flask app
app = Flask(
    __name__,
    template_folder=str(config.BASE_DIR / "templates"),
    static_folder=str(config.STATIC_DIR),
)
app.config.from_object(config)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnavailableProcessor:
    """Placeholder used when optional runtime processors cannot initialize."""

    def __init__(self, name: str, error: Exception):
        self.name = name
        self.error = error

    def process_document(self, *_args, **_kwargs):
        raise RuntimeError(f"{self.name} is unavailable: {self.error}")

    def get_engine_status(self):
        return {"available": False, "error": str(self.error)}

    def romanize_text_simple(self, *_args, **_kwargs):
        raise RuntimeError(f"{self.name} is unavailable: {self.error}")

    def process_text(self, *_args, **_kwargs):
        raise RuntimeError(f"{self.name} is unavailable: {self.error}")


def _init_processor(name, factory):
    try:
        processor = factory()
        logger.info("%s initialized successfully", name)
        return processor
    except Exception as e:
        logger.warning("%s unavailable: %s", name, e)
        return UnavailableProcessor(name, e)


ocr_processor = None
content_extractor = None
japanese_processor = None
page_reconstructor = None
workflow_orchestrator = None


def get_ocr_processor():
    global ocr_processor
    if ocr_processor is None:
        from martial_arts_ocr.ocr.processor import OCRProcessor

        ocr_processor = _init_processor("OCRProcessor", OCRProcessor)
    return ocr_processor


def get_content_extractor():
    global content_extractor
    if content_extractor is None:
        from martial_arts_ocr.imaging.content_extractor import ContentExtractor

        content_extractor = _init_processor("ContentExtractor", ContentExtractor)
    return content_extractor


def get_japanese_processor():
    global japanese_processor
    if japanese_processor is None:
        from martial_arts_ocr.japanese.processor import JapaneseProcessor

        japanese_processor = _init_processor("JapaneseProcessor", JapaneseProcessor)
    return japanese_processor


def get_page_reconstructor():
    global page_reconstructor
    if page_reconstructor is None:
        from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor

        page_reconstructor = _init_processor("PageReconstructor", PageReconstructor)
    return page_reconstructor


def get_workflow_orchestrator():
    global workflow_orchestrator
    if workflow_orchestrator is None:
        workflow_orchestrator = WorkflowOrchestrator(
            processor=get_ocr_processor(),
            page_reconstructor=get_page_reconstructor(),
            session_factory=get_db_session,
            processed_path_factory=get_processed_path,
            document_model=Document,
            page_model=Page,
            db_processing_result_model=ProcessingResult,
        )
    return workflow_orchestrator


def create_app(config_overrides: dict | None = None):
    """Return the legacy Flask app object with optional config overrides."""
    global workflow_orchestrator

    if config_overrides:
        data_dir = config_overrides.get("DATA_DIR")
        upload_dir = config_overrides.get("UPLOAD_DIR") or config_overrides.get("UPLOAD_FOLDER")
        processed_dir = config_overrides.get("PROCESSED_DIR")
        if data_dir or upload_dir or processed_dir:
            configure_runtime_paths(data_dir=data_dir, upload_dir=upload_dir, processed_dir=processed_dir)

        database_path = config_overrides.get("DATABASE_PATH")
        database_url = config_overrides.get("DATABASE_URL")
        if database_path or database_url or data_dir:
            if database_path is None and database_url is None and data_dir:
                database_path = Path(data_dir) / "martial_arts_ocr.db"
            configure_database(database_path=database_path, database_url=database_url)
            init_db()
            workflow_orchestrator = None

        app.config.update(config_overrides)
        active_config = get_config()
        active_database_url = database_url
        if not active_database_url and database_path:
            active_database_url = f"sqlite:///{database_path}"
        if not active_database_url:
            active_database_url = active_config.DATABASE_URL
        app.config.update(
            DATA_DIR=str(active_config.DATA_DIR),
            UPLOAD_FOLDER=str(active_config.UPLOAD_FOLDER),
            DATABASE_URL=active_database_url,
        )
    return app

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
                with get_db_session() as session:
                    doc = session.get(Document, document_id)
                    if not doc:
                        logger.error("Document %s not found", document_id)
                        return
                    filename = doc.filename

                request = PipelineRequest(document_id=document_id, image_path=get_upload_path(filename))
                get_workflow_orchestrator().process_document(request)

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
            filename = doc.filename  # capture before session closes

        filepath = get_upload_path(filename)
        result = get_workflow_orchestrator().process_document(
            PipelineRequest(document_id=document_id, image_path=filepath)
        )
        if result.success:
            flash('Document processed successfully', 'success')
            return redirect(url_for('view_document', document_id=document_id))

        flash(f'Processing failed: {result.error or "Unknown error"}', 'error')
        return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Processing error: {e}")
        flash('Processing failed. Please try again.', 'error')
        return redirect(url_for('index'))

# -------------------------
# Core processing function
# -------------------------
def process_image_file(filepath: str, document_id: int) -> dict:
    """Compatibility wrapper for the canonical workflow orchestrator."""
    result = get_workflow_orchestrator().process_document(
        PipelineRequest(document_id=document_id, image_path=Path(filepath))
    )
    return result.to_legacy_dict()

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
        status = get_ocr_processor().get_engine_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Engine status error: {e}")
        return jsonify({'available': False, 'error': 'Failed to get engine status'}), 503

@app.route('/api/romanize', methods=['POST'])
def api_romanize():
    """API endpoint for Japanese romanization."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        romaji = get_japanese_processor().romanize_text_simple(text)
        return jsonify({'romaji': romaji})
    except Exception as e:
        logger.error(f"Romanization error: {e}")
        return jsonify({'error': 'Romanization failed'}), 503

@app.route('/api/translate', methods=['POST'])
def api_translate():
    """API endpoint for Japanese translation."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = get_japanese_processor().process_text(text)
        translation = result.overall_translation if result else None
        return jsonify({'translation': translation})
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed'}), 503

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
        engine_status = get_ocr_processor().get_engine_status()
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
