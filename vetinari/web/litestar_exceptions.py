"""App-level exception handlers for the Litestar ASGI application.

Maps common Python exceptions to appropriate HTTP error responses following
the ADR-0072 envelope format.  Handlers are registered via the
``exception_handlers`` parameter on ``Litestar()`` using the exported
``EXCEPTION_HANDLERS`` dict.

Usage::

    from vetinari.web.litestar_exceptions import EXCEPTION_HANDLERS

    app = Litestar(
        route_handlers=[...],
        exception_handlers=EXCEPTION_HANDLERS,
    )

Exception mapping mirrors the Flask ``handle_errors`` decorator in
``vetinari.web.error_handlers`` so migrated routes behave consistently
regardless of which framework layer handles the exception.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.web.responses import API_VERSION

logger = logging.getLogger(__name__)

# Optional Litestar imports — graceful degradation when not installed
try:
    from litestar import MediaType, Request, Response
    from litestar.exceptions import HTTPException, NotAuthorizedException

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False
    HTTPException = None  # type: ignore[assignment,misc]
    NotAuthorizedException = None  # type: ignore[assignment,misc]


def _error_body(msg: str, code: int) -> dict[str, Any]:
    """Build an ADR-0072 error envelope dict.

    Args:
        msg: Human-readable error description.
        code: HTTP status code to embed in the response body.

    Returns:
        Dict with ``status``, ``error``, ``code``, and ``api_version`` fields.
    """
    return {"status": "error", "error": msg, "code": code, "api_version": API_VERSION}


def value_error_handler(request: Request, exc: ValueError) -> Response:  # type: ignore[type-arg]
    """Convert ``ValueError`` to a 400 Bad Request JSON response.

    Logs the validation failure at WARNING level without leaking the raw
    exception message to the client.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The ``ValueError`` that was raised.

    Returns:
        JSON Response with status 400 and ADR-0072 error envelope.
    """
    logger.warning(
        "ValueError on %s %s — rejecting with 400: %s",
        request.method,
        request.url.path,
        exc,
    )
    return Response(
        content=_error_body("Invalid request parameters", 400),
        status_code=400,
        media_type=MediaType.JSON,
    )


def key_error_handler(request: Request, exc: KeyError) -> Response:  # type: ignore[type-arg]
    """Convert ``KeyError`` to a 400 Bad Request JSON response.

    Logs the missing-key error at WARNING level without exposing the key
    name to the client.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The ``KeyError`` that was raised.

    Returns:
        JSON Response with status 400 and ADR-0072 error envelope.
    """
    logger.warning(
        "KeyError on %s %s — rejecting with 400: missing key %s",
        request.method,
        request.url.path,
        exc,
    )
    return Response(
        content=_error_body("Invalid request parameters", 400),
        status_code=400,
        media_type=MediaType.JSON,
    )


def file_not_found_handler(request: Request, exc: FileNotFoundError) -> Response:  # type: ignore[type-arg]
    """Convert ``FileNotFoundError`` to a 404 Not Found JSON response.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The ``FileNotFoundError`` that was raised.

    Returns:
        JSON Response with status 404 and ADR-0072 error envelope.
    """
    logger.warning(
        "FileNotFoundError on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return Response(
        content=_error_body("Resource not found", 404),
        status_code=404,
        media_type=MediaType.JSON,
    )


def permission_error_handler(request: Request, exc: PermissionError) -> Response:  # type: ignore[type-arg]
    """Convert ``PermissionError`` to a 403 Forbidden JSON response.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The ``PermissionError`` that was raised.

    Returns:
        JSON Response with status 403 and ADR-0072 error envelope.
    """
    logger.warning(
        "PermissionError on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return Response(
        content=_error_body("Permission denied", 403),
        status_code=403,
        media_type=MediaType.JSON,
    )


def not_authorized_handler(request: Request, exc: Exception) -> Response:  # type: ignore[type-arg]
    """Convert ``NotAuthorizedException`` to a 401 Unauthorized JSON response.

    The admin_guard raises this when a request lacks valid admin credentials.
    Without this handler the exception falls through to the generic 500 handler,
    leaking the auth rejection as an internal server error.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The ``NotAuthorizedException`` from the guard.

    Returns:
        JSON Response with status 401 and ADR-0072 error envelope.
    """
    logger.warning(
        "Unauthorized request on %s %s — admin privileges required",
        request.method,
        request.url.path,
    )
    return Response(
        content=_error_body("Admin privileges required", 401),
        status_code=401,
        media_type=MediaType.JSON,
    )


def http_exception_handler(request: Request, exc: Exception) -> Response:  # type: ignore[type-arg]
    """Convert Litestar ``HTTPException`` subclasses to their own status code.

    Litestar raises ``ClientException`` (400), ``ValidationException`` (400),
    ``NotFoundException`` (404), ``MethodNotAllowedException`` (405), and
    similar subclasses for well-defined protocol violations.  Without this
    handler every one of those falls through to ``generic_exception_handler``
    and becomes a 500, masking the real 4xx cause from clients and tests.

    This handler preserves the exception's own ``status_code`` so that, e.g.,
    a malformed JSON body correctly returns 400 rather than 500.

    Args:
        request: The Litestar request that triggered the exception.
        exc: An ``HTTPException`` instance carrying its own ``status_code``.

    Returns:
        JSON Response with the exception's status code and ADR-0072 envelope.
    """
    # exc is typed as Exception here because Litestar's handler protocol does
    # not narrow the type — cast is safe because we only register this handler
    # against HTTPException in EXCEPTION_HANDLERS.
    http_exc = exc  # type: ignore[assignment]
    status = getattr(http_exc, "status_code", 400)
    detail = getattr(http_exc, "detail", str(exc))
    if status >= 500:
        # HTTPException subclasses that map to 5xx (e.g. InternalServerException)
        # should still go through the generic error path so the traceback is logged.
        logger.error(
            "Litestar HTTPException (5xx) on %s %s — status %s: %s",
            request.method,
            request.url.path,
            status,
            detail,
            exc_info=exc,
        )
        return Response(
            content=_error_body("Internal server error", status),
            status_code=status,
            media_type=MediaType.JSON,
        )
    logger.warning(
        "Litestar HTTPException (4xx) on %s %s — status %s: %s",
        request.method,
        request.url.path,
        status,
        detail,
    )
    return Response(
        content=_error_body(detail or "Bad request", status),
        status_code=status,
        media_type=MediaType.JSON,
    )


def generic_exception_handler(request: Request, exc: Exception) -> Response:  # type: ignore[type-arg]
    """Convert any unhandled ``Exception`` to a 500 Internal Server Error response.

    Logs the full traceback at ERROR level via ``logger.exception()`` so the
    root cause is always captured server-side without leaking stack details
    to the client.

    Args:
        request: The Litestar request that triggered the exception.
        exc: The unhandled exception.

    Returns:
        JSON Response with status 500 and ADR-0072 error envelope.
    """
    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return Response(
        content=_error_body("Internal server error", 500),
        status_code=500,
        media_type=MediaType.JSON,
    )


# Mapping of exception type → handler, ready to unpack into Litestar(exception_handlers=...).
# Order matters for specificity — more specific exceptions should appear before broad ones.
# HTTPException must come before the bare Exception catch-all so that Litestar's
# own 4xx/5xx exceptions are returned with their correct status codes instead of
# being swallowed as 500 by the generic handler.
EXCEPTION_HANDLERS: dict[type[Exception], Any] = {
    ValueError: value_error_handler,
    KeyError: key_error_handler,
    FileNotFoundError: file_not_found_handler,
    PermissionError: permission_error_handler,
    Exception: generic_exception_handler,
}

# HTTPException and NotAuthorizedException are only available when Litestar is
# installed — add them after dict construction so the module still loads in
# Flask-only environments. HTTPException must be registered so that its
# subclasses (e.g. ServiceUnavailableException) preserve their own status codes
# instead of being collapsed to 500 by the broad Exception handler.
if HTTPException is not None:
    EXCEPTION_HANDLERS[HTTPException] = http_exception_handler
if NotAuthorizedException is not None:
    EXCEPTION_HANDLERS[NotAuthorizedException] = not_authorized_handler
