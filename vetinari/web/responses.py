"""Standardized HTTP response helpers for Vetinari web handlers.

All JSON API responses follow a consistent schema (ADR-0072):
- Success: ``{"status": "ok", "data": ..., "code": int, "api_version": "1.0.0"}``
- Error:   ``{"status": "error", "error": str, "code": int, "api_version": "1.0.0"}``

Usage::

    from vetinari.web.responses import success_response, litestar_error_response

    return success_response({"items": [...]})
    return litestar_error_response("Not found", code=404)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Current API version included in every response envelope (ADR-0072).
# Source of truth is config/api_versioning.yaml; this constant is the
# runtime fallback so the module works without YAML parsing at import time.
API_VERSION: str = "1.0.0"


def success_response(data: Any = None, code: int = 200) -> dict[str, Any]:
    """Build a standard success response envelope.

    Returns a plain dict (not a Flask Response) so callers can pass it
    directly to ``jsonify`` or return it from Litestar handlers.
    Includes the ``api_version`` field per ADR-0072.

    Args:
        data: The response payload.  May be any JSON-serializable value.
        code: Nominal HTTP status code included in the envelope body.

    Returns:
        Dict with ``status``, ``data``, ``code``, and ``api_version`` fields.
    """
    return {"status": "ok", "data": data, "code": code, "api_version": API_VERSION}


def litestar_error_response(msg: str, code: int = 400, details: Any = None) -> Any:
    """Build an error response as a Litestar Response object (ADR-0072 envelope).

    Equivalent to ``error_response`` but returns a Litestar Response instead
    of a Flask ``(Response, status_code)`` tuple.  Used by migrated Litestar
    handlers that need to return early with an error.

    The Litestar import is deferred inside the function body so this module
    remains importable in Flask-only environments where Litestar is not
    installed.

    Args:
        msg: Human-readable error description.
        code: HTTP status code.
        details: Optional additional context to include in the envelope.

    Returns:
        Litestar Response with JSON error body following the ADR-0072 envelope.
    """
    from litestar import MediaType
    from litestar import Response as LitestarResponse

    body: dict[str, Any] = {"status": "error", "error": msg, "code": code, "api_version": API_VERSION}
    if details is not None:
        body["details"] = details
    return LitestarResponse(content=body, status_code=code, media_type=MediaType.JSON)
