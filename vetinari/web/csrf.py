"""CSRF protection via custom header validation.

Requires X-Requested-With header on all mutation requests (POST/PUT/DELETE/PATCH).
This prevents cross-origin form submissions since browsers cannot add custom
headers to cross-origin requests without CORS preflight approval.

Decision: custom header CSRF for local-first app (ADR-0071).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litestar.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Methods that modify state and need CSRF protection
_UNSAFE_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# Paths exempt from CSRF check (machine-to-machine or health)
_EXEMPT_PATHS = frozenset({"/health", "/api/v1/a2a"})

CSRF_HEADER = "X-Requested-With"


class CSRFMiddleware:
    """ASGI middleware that enforces custom-header CSRF protection.

    For every mutation request (POST, PUT, DELETE, PATCH) the middleware
    checks that the ``X-Requested-With`` header is present and non-empty.
    Browsers cannot attach arbitrary headers to cross-origin requests without
    a CORS preflight, so this header acts as an unforgeable same-origin proof.

    Requests to exempt paths (``/health``, ``/api/v1/a2a``) and safe HTTP
    methods are passed through without inspection.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Store the next ASGI application in the middleware chain.

        Args:
            app: The downstream ASGI application to delegate to.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Intercept HTTP requests and enforce CSRF header presence.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "")
        path: str = scope.get("path", "")

        # Safe methods and exempt paths bypass the check
        if method not in _UNSAFE_METHODS or path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract headers; ASGI delivers them as a list of (name, value) byte pairs
        headers: dict[bytes, bytes] = {name.lower(): value for name, value in scope.get("headers", [])}

        csrf_value = headers.get(CSRF_HEADER.lower().encode())
        if csrf_value:
            await self.app(scope, receive, send)
            return

        # Missing or empty header — block with 403
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        logger.warning(
            "CSRF check failed: %s %s from %s — missing %s header",
            method,
            path,
            client_ip,
            CSRF_HEADER,
        )

        body = json.dumps({
            "error": "CSRF validation failed",
            "detail": f"Mutation requests must include the '{CSRF_HEADER}' header.",
        }).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("utf-8")),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })
