"""
Main Flask application for Martial Arts OCR.
Handles file uploads, OCR processing, and web interface.
"""
import os
import json
from io import BytesIO
from pathlib import Path
from threading import Thread

from flask import (
    current_app,
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from jinja2 import TemplateNotFound
import logging
from datetime import datetime
from PIL import Image, ImageOps

from martial_arts_ocr.config import (
    allowed_file,
    get_config,
    get_processed_path,
    get_upload_path,
)
from martial_arts_ocr.app.dependencies import AppDependencies
from martial_arts_ocr.db.context import DatabaseConfig, DatabaseContext
from martial_arts_ocr.db.database import get_database_context, get_db_session, init_db
from martial_arts_ocr.db.models import Document, Page, ProcessingResult
from martial_arts_ocr.pipeline import PipelineRequest, WorkflowOrchestrator
from martial_arts_ocr.pipeline.document_models import DocumentResult, PageResult
from martial_arts_ocr.pipeline.extraction_service import ExtractionService, ExtractionServiceOptions
from martial_arts_ocr.review import OrientationService, REGION_TYPES, ReviewWorkbenchStore

APP_EXTENSION_KEY = "martial_arts_ocr"

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


def _default_ocr_processor_factory():
    from martial_arts_ocr.ocr.processor import OCRProcessor

    return _init_processor("OCRProcessor", OCRProcessor)


def _default_content_extractor_factory():
    from martial_arts_ocr.imaging.content_extractor import ContentExtractor

    return _init_processor("ContentExtractor", ContentExtractor)


def _default_japanese_processor_factory():
    from martial_arts_ocr.japanese.processor import JapaneseProcessor

    return _init_processor("JapaneseProcessor", JapaneseProcessor)


def _default_page_reconstructor_factory():
    from martial_arts_ocr.reconstruction.page_reconstructor import PageReconstructor

    return _init_processor("PageReconstructor", PageReconstructor)


def _extension_deps(flask_app: Flask | None = None) -> AppDependencies | None:
    try:
        target_app = flask_app or current_app._get_current_object()
    except RuntimeError:
        target_app = app
    return target_app.extensions.get(APP_EXTENSION_KEY)


def _current_app_is_legacy() -> bool:
    try:
        return current_app._get_current_object() is app
    except RuntimeError:
        return True


def _attach_dependencies(
    flask_app: Flask,
    *,
    db_context: DatabaseContext,
    data_dir: Path,
    upload_dir: Path,
    processed_dir: Path,
    orchestrator: WorkflowOrchestrator | None = None,
    processor_factory=None,
    content_extractor_factory=None,
    japanese_processor_factory=None,
    page_reconstructor_factory=None,
    extraction_service: ExtractionService | None = None,
) -> AppDependencies:
    deps = AppDependencies(
        db_context=db_context,
        data_dir=Path(data_dir),
        upload_dir=Path(upload_dir),
        processed_dir=Path(processed_dir),
        orchestrator=orchestrator,
        extraction_service=extraction_service,
        ocr_processor_factory=processor_factory or _default_ocr_processor_factory,
        content_extractor_factory=content_extractor_factory or _default_content_extractor_factory,
        japanese_processor_factory=japanese_processor_factory or _default_japanese_processor_factory,
        page_reconstructor_factory=page_reconstructor_factory or _default_page_reconstructor_factory,
    )
    flask_app.extensions[APP_EXTENSION_KEY] = deps
    return deps


def _build_extraction_service(app_config) -> ExtractionService:
    return ExtractionService(
        ExtractionServiceOptions(
            enable_image_regions=_config_bool(app_config.get("ENABLE_IMAGE_REGION_EXTRACTION", False)),
            save_crops=_config_bool(app_config.get("IMAGE_REGION_EXTRACTION_SAVE_CROPS", True)),
            fail_on_extraction_error=_config_bool(app_config.get("IMAGE_REGION_EXTRACTION_FAIL_ON_ERROR", False)),
            enable_paddle_layout_fusion=_config_bool(app_config.get("ENABLE_PADDLE_LAYOUT_FUSION", False)),
            paddle_layout_model_dir=app_config.get("PADDLE_LAYOUT_MODEL_DIR"),
        )
    )


def _config_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_attach_dependencies(
    app,
    db_context=get_database_context(),
    data_dir=config.DATA_DIR,
    upload_dir=Path(config.UPLOAD_FOLDER),
    processed_dir=get_processed_path(""),
    extraction_service=_build_extraction_service(app.config),
)


def get_ocr_processor():
    global ocr_processor
    deps = _extension_deps()
    if deps is not None and not _current_app_is_legacy():
        return deps.get_ocr_processor()
    if ocr_processor is None:
        ocr_processor = _default_ocr_processor_factory()
    return ocr_processor


def get_content_extractor():
    global content_extractor
    deps = _extension_deps()
    if deps is not None and not _current_app_is_legacy():
        return deps.get_content_extractor()
    if content_extractor is None:
        content_extractor = _default_content_extractor_factory()
    return content_extractor


def get_japanese_processor():
    global japanese_processor
    deps = _extension_deps()
    if deps is not None and not _current_app_is_legacy():
        return deps.get_japanese_processor()
    if japanese_processor is None:
        japanese_processor = _default_japanese_processor_factory()
    return japanese_processor


def get_page_reconstructor():
    global page_reconstructor
    deps = _extension_deps()
    if deps is not None and not _current_app_is_legacy():
        return deps.get_page_reconstructor()
    if page_reconstructor is None:
        page_reconstructor = _default_page_reconstructor_factory()
    return page_reconstructor


def get_workflow_orchestrator():
    global workflow_orchestrator
    deps = _extension_deps()
    if deps is not None and not _current_app_is_legacy():
        return deps.get_orchestrator()
    if workflow_orchestrator is None:
        deps = _extension_deps(app)
        workflow_orchestrator = deps.get_orchestrator() if deps else WorkflowOrchestrator()
    return workflow_orchestrator


def get_current_db_session():
    deps = _extension_deps()
    if deps is not None:
        return deps.db_context.get_db_session()
    return get_db_session()


def get_current_upload_path(filename: str) -> Path:
    deps = _extension_deps()
    if deps is not None:
        return deps.upload_dir / filename
    return get_upload_path(filename)


def get_current_processed_path(filename: str) -> Path:
    deps = _extension_deps()
    if deps is not None:
        return deps.processed_dir / filename
    return get_processed_path(filename)


def _build_flask_app(config_overrides: dict | None = None) -> Flask:
    cfg = get_config()
    flask_app = Flask(
        __name__,
        template_folder=str(cfg.BASE_DIR / "templates"),
        static_folder=str(cfg.STATIC_DIR),
    )
    flask_app.config.from_object(cfg)
    if config_overrides:
        flask_app.config.update(config_overrides)
    return flask_app


def _register_existing_routes(target_app: Flask) -> None:
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        target_app.add_url_rule(
            rule.rule,
            endpoint=rule.endpoint,
            view_func=app.view_functions[rule.endpoint],
            methods=rule.methods,
            defaults=rule.defaults,
            strict_slashes=rule.strict_slashes,
        )
    target_app.context_processor(inject_globals)
    target_app.template_filter("filesizeformat")(filesizeformat)
    target_app.before_request(enforce_allowed_hosts)
    target_app.register_error_handler(404, not_found_error)
    target_app.register_error_handler(500, internal_error)
    target_app.register_error_handler(RequestEntityTooLarge, handle_file_too_large)


def create_app(
    config_overrides: dict | None = None,
    *,
    orchestrator: WorkflowOrchestrator | None = None,
    db_context: DatabaseContext | None = None,
    processor_factory=None,
    content_extractor_factory=None,
    japanese_processor_factory=None,
    page_reconstructor_factory=None,
    extraction_service: ExtractionService | None = None,
):
    """Create an isolated Flask app instance with app-scoped dependencies."""
    flask_app = _build_flask_app(config_overrides)

    overrides = config_overrides or {}
    data_dir = Path(overrides.get("DATA_DIR") or flask_app.config.get("DATA_DIR", config.DATA_DIR))
    runtime_dir = Path(overrides.get("RUNTIME_DIR") or data_dir / "runtime")
    upload_override = overrides.get("UPLOAD_DIR") or overrides.get("UPLOAD_FOLDER")
    processed_override = overrides.get("PROCESSED_DIR")
    if upload_override:
        upload_dir = Path(upload_override)
    elif "DATA_DIR" in overrides:
        upload_dir = runtime_dir / "uploads"
    else:
        upload_dir = Path(flask_app.config.get("UPLOAD_FOLDER") or runtime_dir / "uploads")
    processed_dir = Path(processed_override) if processed_override else runtime_dir / "processed"
    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    database_path = overrides.get("DATABASE_PATH")
    database_url = overrides.get("DATABASE_URL")
    if db_context is None:
        if database_path:
            db_context = DatabaseContext(DatabaseConfig(database_path=Path(database_path)))
        elif database_url:
            db_context = DatabaseContext(DatabaseConfig.from_url(database_url))
        elif "DATA_DIR" in overrides:
            db_context = DatabaseContext(DatabaseConfig(database_path=runtime_dir / "db" / "martial_arts_ocr.db"))
        else:
            db_context = DatabaseContext(DatabaseConfig.from_url(flask_app.config["DATABASE_URL"]))
    db_context.init_db()

    if extraction_service is None:
        extraction_service = _build_extraction_service(flask_app.config)

    _attach_dependencies(
        flask_app,
        db_context=db_context,
        data_dir=data_dir,
        upload_dir=upload_dir,
        processed_dir=processed_dir,
        orchestrator=orchestrator,
        processor_factory=processor_factory,
        content_extractor_factory=content_extractor_factory,
        japanese_processor_factory=japanese_processor_factory,
        page_reconstructor_factory=page_reconstructor_factory,
        extraction_service=extraction_service,
    )
    flask_app.config.update(
        DATA_DIR=str(data_dir),
        RUNTIME_DIR=str(runtime_dir),
        UPLOAD_FOLDER=str(upload_dir),
        PROCESSED_DIR=str(processed_dir),
        DATABASE_URL=db_context.config.url,
    )
    _register_existing_routes(flask_app)
    return flask_app

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
    active_app = current_app._get_current_object()
    allowed = active_app.config.get("ALLOWED_HOSTS")
    if not allowed:
        return  # no restriction
    host = request.headers.get("Host", "")
    host_only = host.split(":", 1)[0].strip().lower()
    if host_only.startswith('[') and host_only.endswith(']'):
        host_only = host_only[1:-1]  # [::1] -> ::1
    if host_only not in allowed:
        active_app.logger.warning("Denied host %s (allowed: %s)", host_only, sorted(allowed))
        abort(403)

# -------------------------
# Health
# -------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}, 200


# -------------------------
# Local research review workbench
# -------------------------
def _review_workbench_store() -> ReviewWorkbenchStore:
    runtime_dir = Path(current_app.config.get("RUNTIME_DIR") or Path(current_app.config["DATA_DIR"]) / "runtime")
    project_root = Path(current_app.config.get("REVIEW_PROJECTS_DIR") or runtime_dir / "review_projects")
    configured_roots = current_app.config.get("REVIEW_ALLOWED_ROOTS")
    if configured_roots is None:
        configured_roots = [current_app.config.get("DATA_DIR"), Path.cwd()]
    allowed_roots = [
        Path(root)
        for root in configured_roots
        if root
    ]
    return ReviewWorkbenchStore(project_root, allowed_roots=allowed_roots)


def _review_orientation_service() -> OrientationService:
    injected_service = current_app.config.get("REVIEW_ORIENTATION_SERVICE")
    if injected_service is not None:
        return injected_service
    return OrientationService(
        model_path=current_app.config.get(
            "REVIEW_ORIENTATION_MODEL_PATH",
            "experiments/orientation_model/checkpoints/orient_convnext_tiny.pth",
        ),
        ensemble_model_path=current_app.config.get(
            "REVIEW_ORIENTATION_ENSEMBLE_MODEL_PATH",
            "experiments/orientation_model/checkpoints/orient_effnetv2s.pth",
        ),
    )


def _review_json_error(exc: Exception, status_code: int = 400):
    return jsonify({"error": str(exc)}), status_code


def _effective_orientation_degrees(page: dict) -> int:
    orientation = page.get("orientation") or {}
    return int(orientation.get("effective_rotation_degrees") or 0)


def _effective_page_image_path(store: ReviewWorkbenchStore, state: dict, page_id: str) -> Path:
    """Return an oriented runtime copy when page orientation is non-zero."""
    page = store.get_page(state, page_id)
    image_path = store.image_path(state, page_id)
    rotation = _effective_orientation_degrees(page)
    if rotation == 0:
        return image_path

    output_dir = store.project_dir(state["project_id"]) / "oriented_pages"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{page_id}_rot{rotation}.png"
    source_mtime = image_path.stat().st_mtime
    if output_path.exists() and output_path.stat().st_mtime >= source_mtime:
        return output_path

    with Image.open(image_path) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB").rotate(-rotation, expand=True)
        oriented.save(output_path)
    return output_path


def _send_effective_page_image(store: ReviewWorkbenchStore, state: dict, page_id: str):
    page = store.get_page(state, page_id)
    image_path = store.image_path(state, page_id)
    rotation = _effective_orientation_degrees(page)
    if rotation == 0:
        return send_file(image_path)
    with Image.open(image_path) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB").rotate(-rotation, expand=True)
        output = BytesIO()
        oriented.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png")


def _detect_review_regions(
    *,
    image_path: Path,
    page: dict,
    output_dir: Path,
) -> dict:
    """Run review-mode region detection without invoking OCR."""
    injected_service = current_app.config.get("REVIEW_RECOGNITION_SERVICE")
    service = injected_service or ExtractionService(
        ExtractionServiceOptions(
            enable_image_regions=True,
            save_crops=False,
            fail_on_extraction_error=False,
            enable_paddle_layout_fusion=_config_bool(current_app.config.get("ENABLE_PADDLE_LAYOUT_FUSION", False)),
            paddle_layout_model_dir=current_app.config.get("PADDLE_LAYOUT_MODEL_DIR"),
        )
    )
    document = DocumentResult(
        document_id=None,
        source_path=image_path,
        pages=[
            PageResult(
                page_number=1,
                width=page.get("effective_width") or page.get("width"),
                height=page.get("effective_height") or page.get("height"),
                raw_text="",
                metadata={
                    "review_workbench_recognition": True,
                    "ocr_executed": False,
                    "orientation_degrees": _effective_orientation_degrees(page),
                },
            )
        ],
        metadata={
            "review_workbench_recognition": True,
            "ocr_executed": False,
            "orientation_degrees": _effective_orientation_degrees(page),
        },
    )
    enriched = service.enrich_document_result(document, output_dir=output_dir)
    if not enriched.pages:
        return {"regions": [], "rejected_count": 0}
    region_records = [
        _review_region_record_from_image_region(region)
        for region in enriched.pages[0].image_regions
        if region.bbox is not None
    ]
    return {
        "regions": region_records,
        "rejected_count": len((enriched.metadata.get("image_extraction") or {}).get("rejected", [])),
    }


def _review_region_record_from_image_region(region) -> dict:
    metadata = dict(region.metadata or {})
    metadata.setdefault("detector", metadata.get("source") or "review_mode_extraction")
    if region.reading_order is not None:
        metadata.setdefault("reading_order", region.reading_order)
    if region.image_path:
        metadata.setdefault("image_path", str(region.image_path))
    return {
        "region_type": region.region_type,
        "bbox": _bbox_as_xywh(region.bbox),
        "confidence": region.confidence,
        "metadata": metadata,
    }


def _bbox_as_xywh(bbox) -> list[int]:
    if isinstance(bbox, dict):
        return [
            int(bbox.get("x") or 0),
            int(bbox.get("y") or 0),
            int(bbox.get("width") or 1),
            int(bbox.get("height") or 1),
        ]
    return [
        int(getattr(bbox, "x", 0)),
        int(getattr(bbox, "y", 0)),
        int(getattr(bbox, "width", 1)),
        int(getattr(bbox, "height", 1)),
    ]


@app.get("/review")
def review_workbench():
    """Local research workbench for page/region review."""
    return render_template("review_workbench.html", region_types=REGION_TYPES)


@app.post("/api/review/projects")
def api_review_create_project():
    data = request.get_json() or {}
    store = _review_workbench_store()
    try:
        source_folder = data.get("source_folder")
        project_id = data.get("project_id")
        if source_folder:
            state = store.create_project(Path(source_folder), project_id=project_id)
        elif project_id:
            state = store.load_project(str(project_id))
        else:
            return _review_json_error(ValueError("source_folder or project_id is required"))
        return jsonify(state), 200
    except PermissionError as exc:
        return _review_json_error(exc, 403)
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.get("/api/review/projects/<project_id>")
def api_review_get_project(project_id):
    store = _review_workbench_store()
    try:
        return jsonify(store.load_project(project_id)), 200
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)


@app.get("/api/review/projects/<project_id>/pages")
def api_review_list_pages(project_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        return jsonify({"project_id": project_id, "pages": state.get("pages", [])}), 200
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)


@app.get("/api/review/projects/<project_id>/pages/<page_id>")
def api_review_get_page(project_id, page_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        return jsonify(store.get_page(state, page_id)), 200
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)
    except KeyError as exc:
        return _review_json_error(exc, 404)


@app.get("/api/review/projects/<project_id>/pages/<page_id>/image")
def api_review_page_image(project_id, page_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        return _send_effective_page_image(store, state, page_id)
    except PermissionError as exc:
        return _review_json_error(exc, 403)
    except (FileNotFoundError, KeyError) as exc:
        return _review_json_error(exc, 404)


@app.post("/api/review/projects/<project_id>/pages/<page_id>/orientation/detect")
def api_review_detect_orientation(project_id, page_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        image_path = store.image_path(state, page_id)
        result = _review_orientation_service().predict(image_path)
        updated_page = store.update_page_orientation(
            state,
            page_id,
            detected_rotation_degrees=result.rotation_degrees,
            detected_confidence=result.confidence,
            source=result.source,
            status="detected" if result.status == "ok" else result.status,
            metadata=result.metadata,
        )
        return jsonify({"page": updated_page, "orientation": updated_page.get("orientation")}), 200
    except PermissionError as exc:
        return _review_json_error(exc, 403)
    except (FileNotFoundError, KeyError) as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.patch("/api/review/projects/<project_id>/pages/<page_id>/orientation")
def api_review_update_orientation(project_id, page_id):
    data = request.get_json() or {}
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        if "reviewed_rotation_degrees" not in data and "rotation_degrees" not in data:
            return _review_json_error(ValueError("reviewed_rotation_degrees is required"))
        rotation = data.get("reviewed_rotation_degrees", data.get("rotation_degrees"))
        updated_page = store.update_page_orientation(
            state,
            page_id,
            reviewed_rotation_degrees=rotation,
            source="reviewer_override",
            status="reviewed",
            metadata={"reviewer_override": True},
        )
        return jsonify({"page": updated_page, "orientation": updated_page.get("orientation")}), 200
    except PermissionError as exc:
        return _review_json_error(exc, 403)
    except (FileNotFoundError, KeyError) as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.post("/api/review/projects/<project_id>/pages/<page_id>/recognize")
def api_review_recognize_page(project_id, page_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        page = store.get_page(state, page_id)
        image_path = _effective_page_image_path(store, state, page_id)
        output_dir = store.project_dir(project_id) / "recognition" / page_id
        recognition_result = _detect_review_regions(
            image_path=image_path,
            page=page,
            output_dir=output_dir,
        )
        detected_regions = recognition_result["regions"]
        updated_page = store.import_detected_regions(state, page_id, detected_regions)
        return jsonify(
            {
                "page": updated_page,
                "detected_count": len(detected_regions),
                "rejected_count": recognition_result.get("rejected_count", 0),
                "rerun_behavior": "replaced_unreviewed_machine_detection_regions",
            }
        ), 200
    except PermissionError as exc:
        return _review_json_error(exc, 403)
    except (FileNotFoundError, KeyError) as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.post("/api/review/projects/<project_id>/pages/<page_id>/regions")
def api_review_add_region(project_id, page_id):
    data = request.get_json() or {}
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        region = store.add_region(state, page_id, data)
        return jsonify({"region": region, "page": store.get_page(state, page_id)}), 201
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)
    except KeyError as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.patch("/api/review/projects/<project_id>/pages/<page_id>/regions/<region_id>")
def api_review_update_region(project_id, page_id, region_id):
    data = request.get_json() or {}
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        region = store.update_region(state, page_id, region_id, data)
        return jsonify({"region": region, "page": store.get_page(state, page_id)}), 200
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)
    except KeyError as exc:
        return _review_json_error(exc, 404)
    except Exception as exc:
        return _review_json_error(exc, 400)


@app.delete("/api/review/projects/<project_id>/pages/<page_id>/regions/<region_id>")
def api_review_delete_region(project_id, page_id, region_id):
    store = _review_workbench_store()
    try:
        state = store.load_project(project_id)
        store.delete_region(state, page_id, region_id)
        return jsonify({"page": store.get_page(state, page_id)}), 200
    except FileNotFoundError as exc:
        return _review_json_error(exc, 404)
    except KeyError as exc:
        return _review_json_error(exc, 404)

# -------------------------
# Background processing
# -------------------------
def _kickoff_processing_async(document_id: int, flask_app: Flask | None = None):
    """Start processing in a background thread (no HTTP redirect)."""
    target_app = flask_app
    if target_app is None:
        try:
            target_app = current_app._get_current_object()
        except RuntimeError:
            target_app = app

    def _run():
        try:
            with target_app.app_context():
                with get_current_db_session() as session:
                    doc = session.get(Document, document_id)
                    if not doc:
                        logger.error("Document %s not found", document_id)
                        return
                    filename = doc.filename

                request = PipelineRequest(document_id=document_id, image_path=get_current_upload_path(filename))
                get_workflow_orchestrator().process_document(request)

        except Exception as e:
            logger.exception("Background processing error for doc %s", document_id)
            try:
                with target_app.app_context(), get_current_db_session() as session:
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
        with get_current_db_session() as db:
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
        with get_current_db_session() as session:
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
        filepath = get_current_upload_path(filename)
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
        with get_current_db_session() as session:
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
        with get_current_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc:
                flash('Document not found', 'error')
                return redirect(url_for('index'))
            filename = doc.filename  # capture before session closes

        filepath = get_current_upload_path(filename)
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
        with get_current_db_session() as session:
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
        with get_current_db_session() as session:
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
        output_dir = get_current_processed_path(f"doc_{document_id}")
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
        with get_current_db_session() as session:
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
        with get_current_db_session() as session:
            doc = session.get(Document, document_id)
            if not doc or doc.status != 'completed':
                flash('Document not available for download', 'error')
                return redirect(url_for('gallery'))
            original = doc.original_filename  # capture before close

        output_dir = get_current_processed_path(f"doc_{document_id}")
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
