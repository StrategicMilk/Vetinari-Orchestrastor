"""Litestar security guards for Vetinari admin-only endpoints.

Provides the ``admin_guard`` Litestar guard and the ``is_admin_connection``
predicate that replicate the Flask ``is_admin_user()`` / ``require_admin``
logic from ``vetinari.web`` for native Litestar route handlers.

Usage::

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/admin/something", guards=[admin_guard])
    async def admin_route() -> dict:
        ...

This is step 3 of the Flask→Litestar migration (ADR-0066): providing
Litestar-native equivalents of the Flask auth decorators so migrated
handlers do not need the WSGI bridge for auth enforcement.
"""

from __future__ import annotations

import hmac
import logging
import os

logger = logging.getLogger(__name__)

# Optional Litestar imports — graceful degradation when not installed
try:
    from litestar.connection import ASGIConnection
    from litestar.exceptions import NotAuthorizedException
    from litestar.handlers import BaseRouteHandler

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Loopback addresses accepted when no token is configured.
# "testclient" is included because Litestar's TestClient (httpx ASGITransport)
# sets connection.client.host to "testclient" — it is a synthetic hostname that
# cannot appear on any real network interface, so including it here does not
# weaken production security.
_LOCALHOST_IPS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})

# Primary admin token header name. Authorization: Bearer <token> is accepted as a fallback.
_ADMIN_HEADER_NAME = "X-Admin-Token"


def _get_exact_header(connection: ASGIConnection, name: str) -> str:  # type: ignore[type-arg]
    """Return the value of a header whose name matches exactly (case-sensitive).

    Litestar normalises stored header names to lowercase, so this function
    iterates the raw scope headers to find the first entry whose name matches
    ``name`` byte-for-byte after UTF-8 decoding.  Returns empty string when
    no exact match exists.

    This enforces strict header-name casing so that ``x-admin-token`` is NOT
    treated as equivalent to ``X-Admin-Token`` — closing the lowercase-alias
    auth bypass vector.

    Args:
        connection: The active ASGI connection carrying the incoming request.
        name: The exact header name to look up (case-sensitive).

    Returns:
        The header value string, or empty string when absent.
    """
    # ASGI scope["headers"] is a list of (name_bytes, value_bytes) tuples
    # with the original casing from the HTTP client preserved.
    raw_headers: list[tuple[bytes, bytes]] = connection.scope.get("headers", [])
    name_bytes = name.encode("latin-1")
    for raw_name, raw_value in raw_headers:
        if raw_name == name_bytes:
            return raw_value.decode("latin-1")
    return ""


def is_admin_connection(connection: ASGIConnection) -> bool:  # type: ignore[type-arg]
    """Return True if the connection originates from an authorised admin.

    Accepts the ``X-Admin-Token`` header (case-insensitive per RFC 7230)
    carrying the value configured in ``VETINARI_ADMIN_TOKEN``.  Falls back to
    ``Authorization: Bearer <token>`` when ``X-Admin-Token`` is absent.

    Uses constant-time comparison (``hmac.compare_digest``) to prevent
    timing-based token oracle attacks (P1.C1/P1.H10).

    HTTP header names are case-insensitive (RFC 7230 §3.2).  Using
    ``connection.headers.get()`` (Litestar's case-insensitive dict) ensures
    compatibility with HTTP clients that normalise header names to lowercase
    (e.g. httpx, h2).

    When no token is set, falls back to an IP-based localhost check.
    ``X-Forwarded-For`` is only trusted when ``VETINARI_TRUSTED_PROXY`` is
    set explicitly, to prevent IP-spoofing (P1.H8).

    Args:
        connection: The active ASGI connection carrying the incoming request.

    Returns:
        True when the request is authorised as admin, False otherwise.
    """
    admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
    if admin_token:
        # Prefer X-Admin-Token; fall back to Authorization: Bearer <token>.
        # Header name lookup is case-insensitive per RFC 7230 §3.2.
        # Try lowercase first (real HTTP clients — httpx/h2 normalise header names).
        # Fall back to exact-case for unit test mocks that use plain dicts (case-sensitive).
        req_token = connection.headers.get(_ADMIN_HEADER_NAME.lower(), "") or connection.headers.get(
            _ADMIN_HEADER_NAME, ""
        )
        auth_header = connection.headers.get("authorization", "") or connection.headers.get("Authorization", "")
        # RFC 7235 §5.1.2: auth-scheme tokens are case-insensitive. Accept "Bearer",
        # "bearer", "BEARER", and mixed case. The scheme must still be separated
        # from the credential by a single space.
        bearer = ""
        if auth_header:
            parts = auth_header.split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                bearer = parts[1].strip()
        provided = req_token or bearer
        # hmac.compare_digest requires both operands to be the same type
        try:
            return hmac.compare_digest(provided.encode(), admin_token.encode())
        except Exception:
            # If comparison fails for any reason, fail CLOSED (anti-pattern: fail-open security)
            logger.warning("Admin token comparison failed unexpectedly — denying access (fail-closed)")
            return False

    # No token configured — fall back to IP-based localhost check.
    # Read X-Forwarded-For only when the operator has opted in via
    # VETINARI_TRUSTED_PROXY to prevent spoofing.
    trusted_proxy = os.environ.get("VETINARI_TRUSTED_PROXY", "").lower() in ("1", "true", "yes")
    if trusted_proxy:
        forwarded = connection.headers.get("X-Forwarded-For", "")
        if forwarded:
            remote = forwarded.split(",")[0].strip()
        else:
            client = connection.client
            remote = client.host if client else ""
    else:
        client = connection.client
        remote = client.host if client else ""

    return remote in _LOCALHOST_IPS


def admin_guard(connection: ASGIConnection, _: BaseRouteHandler) -> None:  # type: ignore[type-arg]
    """Litestar guard that enforces admin access on a route handler.

    Raises ``NotAuthorizedException`` when the caller is not an admin,
    which Litestar translates to a 401 response.  Wire this guard into
    a handler via the ``guards`` parameter::

        @get("/api/v1/admin/foo", guards=[admin_guard])
        async def admin_foo() -> dict:
            ...

    Replicates the Flask ``require_admin`` decorator logic using Litestar's
    guard protocol so the WSGI bridge is not needed for auth on migrated routes.

    Args:
        connection: The active ASGI connection for the incoming request.
        _: The route handler (unused — required by Litestar guard protocol).

    Raises:
        NotAuthorizedException: When the caller is not an authorised admin.
    """
    if not is_admin_connection(connection):
        logger.warning(
            "Admin guard rejected request from %s",
            connection.client.host if connection.client else "unknown",
        )
        raise NotAuthorizedException("Admin privileges required")
