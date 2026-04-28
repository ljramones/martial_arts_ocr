"""Flask application factory.

The legacy root ``app.py`` still owns the full UI and processing routes. This
factory provides a lightweight testable entrypoint while preserving the legacy
runtime until the route modules are migrated.
"""

from __future__ import annotations

from typing import Any


def create_app(config_overrides: dict[str, Any] | None = None):
    """Create a Flask app.

    In testing mode this avoids importing the legacy app module, which eagerly
    initializes OCR engines and the database. In runtime mode it returns the
    existing Flask app object from root ``app.py``.
    """

    testing = bool((config_overrides or {}).get("TESTING"))

    if testing:
        from flask import Flask

        app = Flask(__name__, template_folder="../../../templates", static_folder="../../../static")
        app.config.update(config_overrides or {})

        @app.get("/healthz")
        def healthz():
            return {"ok": True}, 200

        return app

    from app import app as legacy_app

    if config_overrides:
        legacy_app.config.update(config_overrides)
    return legacy_app

