"""Vetinari Web UI — thin entry point.

The Flask application and all routes live in ``vetinari.web.app``.
This module re-exports the public API for backward compatibility.
"""

from vetinari.web.app import create_app, app  # noqa: F401

# Re-export shared state that some callers access via web_ui
from vetinari.web.shared import current_config  # noqa: F401

__all__ = ["create_app", "app", "current_config"]

if __name__ == "__main__":
    import os
    _debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    _port = int(os.environ.get("VETINARI_WEB_PORT", 5000))
    _host = os.environ.get("VETINARI_WEB_HOST", "0.0.0.0")
    app.run(host=_host, port=_port, debug=_debug)
