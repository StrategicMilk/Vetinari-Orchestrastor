"""Litestar middleware — security, CORS, correlation, scheduler notify, JSON guard, admin fail-closed.

Provides middleware components wired into ``create_app()`` in ``litestar_app.py``:

1. SecurityHeadersMiddleware       — adds defensive HTTP response headers
2. CORSMiddleware                  — restricts CORS to localhost origins only
3. RequestIdMiddleware             — injects a per-request correlation ID into logs
4. UserActivityMiddleware          — notifies the idle training scheduler on each request
5. JsonDepthGuardMiddleware        — rejects JSON bodies with nesting depth > 5
6. RemoteMutationGuardMiddleware   — fails closed on remote mutation requests when no
   admin token is configured (SESSION-32.2). Localhost requests bypass. Remote
   clients sending POST/PUT/DELETE/PATCH without a valid admin token receive a
   bounded 401. Explicit opt-in via ``VETINARI_ALLOW_UNAUTHENTICATED_REMOTE_MUTATION=1``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from vetinari.constants import DEFAULT_WEB_CLIENT_PORT, DEFAULT_WEB_PORT
from vetinari.web.csrf import CSRF_HEADER

logger = logging.getLogger(__name__)

# Methods whose request bodies mutate server state. Used by RemoteMutationGuardMiddleware
# to decide which requests the fail-closed remote guard must evaluate.
_MUTATION_METHODS: frozenset[str] = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# Hosts/IPs that are treated as local-only and bypass the remote mutation guard.
# Kept in sync with litestar_guards._LOCALHOST_IPS — if that set changes, update here too.
_LOCALHOST_IPS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

# ── Litestar middleware base ──────────────────────────────────────────────────

try:
    from litestar.middleware.base import ASGIMiddleware
    from litestar.types import ASGIApp, Receive, Scope, Send

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# ── Allowed CORS origins (localhost only) ────────────────────────────────────

_ALLOWED_ORIGINS: frozenset[str] = frozenset({
    "http://localhost",
    "http://127.0.0.1",
    f"http://localhost:{DEFAULT_WEB_CLIENT_PORT}",
    f"http://127.0.0.1:{DEFAULT_WEB_CLIENT_PORT}",
    f"http://localhost:{DEFAULT_WEB_PORT}",
    f"http://127.0.0.1:{DEFAULT_WEB_PORT}",
})

_ALLOWED_CORS_HEADERS = ", ".join(("Content-Type", "Authorization", "X-Admin-Token", CSRF_HEADER)).encode("latin-1")

# ── Static CSP header (no per-request nonce — templates not yet on Litestar) ─

_CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net "
    "https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
    "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
    "img-src 'self' data:; connect-src 'self'"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_header(scope: Any, name: bytes) -> str:
    """Extract a single HTTP header value from an ASGI scope.

    Args:
        scope: The ASGI connection scope dict.
        name:  Header name as lowercase bytes (e.g. ``b"origin"``).

    Returns:
        Header value as str, or empty string when absent.
    """
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1", errors="replace")
    return ""


def _is_secure(scope: Any) -> bool:
    """Return True when the connection arrived over HTTPS.

    Args:
        scope: The ASGI connection scope dict.

    Returns:
        True for ``https`` or ``wss`` schemes.
    """
    return scope.get("scheme", "http") in ("https", "wss")


def _inject_header(headers: list[tuple[bytes, bytes]], name: str, value: str) -> None:
    """Append a response header only if the name is not already present.

    Mutates ``headers`` in place.  Uses case-insensitive comparison against
    existing names so that downstream handlers can override a header by setting
    it first.

    Args:
        headers: Mutable list of ``(name_bytes, value_bytes)`` pairs.
        name:    Header name string (e.g. ``"X-Frame-Options"``).
        value:   Header value string.
    """
    name_lower = name.lower().encode("latin-1")
    for existing_name, _ in headers:
        if existing_name.lower() == name_lower:
            return  # Already set — do not override
    headers.append((name.encode("latin-1"), value.encode("latin-1")))


# ── SecurityHeadersMiddleware ─────────────────────────────────────────────────

if _LITESTAR_AVAILABLE:

    class SecurityHeadersMiddleware(ASGIMiddleware):
        """Add defensive HTTP security headers to every response.

        Ports the ``_add_security_headers`` Flask after_request hook from
        ``vetinari/web_ui.py`` (lines 132-153).  Uses a static CSP because
        Litestar does not yet serve Jinja2 templates — the nonce variant will
        be restored in a later session when template rendering is migrated.

        The HSTS header is only added on HTTPS connections.
        """

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Intercept the response and inject security headers.

            Args:
                scope:    ASGI scope dict for the current request.
                receive:  ASGI receive callable.
                send:     ASGI send callable (wrapped to inject headers).
                next_app: The next ASGI app in the middleware stack.
            """
            secure = _is_secure(scope)

            async def send_with_headers(message: Any) -> None:
                """Intercept outgoing ASGI messages and inject security headers.

                Adds X-Content-Type-Options, X-Frame-Options, CSP, and other
                defensive headers to every HTTP response start message before
                forwarding to the original send callable. HSTS is added only
                on HTTPS connections.

                Args:
                    message: ASGI message dict — only ``http.response.start``
                        messages are modified; all other types are forwarded
                        unchanged.
                """
                if message["type"] == "http.response.start":
                    headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                    _inject_header(headers, "X-Content-Type-Options", "nosniff")
                    _inject_header(headers, "X-Frame-Options", "DENY")
                    _inject_header(headers, "X-XSS-Protection", "1; mode=block")
                    _inject_header(headers, "Referrer-Policy", "strict-origin-when-cross-origin")
                    _inject_header(headers, "Content-Security-Policy", _CSP_HEADER)
                    if secure:
                        _inject_header(
                            headers,
                            "Strict-Transport-Security",
                            "max-age=31536000; includeSubDomains",
                        )
                    _inject_header(headers, "Permissions-Policy", "camera=(), microphone=(), geolocation=()")
                    message = {**message, "headers": headers}
                await send(message)

            await next_app(scope, receive, send_with_headers)

    # ── CORSMiddleware ────────────────────────────────────────────────────────

    class CORSMiddleware(ASGIMiddleware):
        """Restrict CORS to localhost origins only — no wildcard.

        Ports the ``_restrict_cors`` Flask after_request hook from
        ``vetinari/web_ui.py`` (lines 158-177).  When the ``Origin`` header
        is present and matches an allowed localhost origin the appropriate
        CORS response headers are added.  Origins not in the allowlist receive
        no CORS headers at all, so browsers will block cross-origin requests
        from untrusted origins.
        """

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Intercept the response and apply CORS headers.

            Args:
                scope:    ASGI scope dict carrying the request headers.
                receive:  ASGI receive callable.
                send:     ASGI send callable (wrapped to inject CORS headers).
                next_app: The next ASGI app in the middleware stack.
            """
            origin = _get_header(scope, b"origin")
            is_allowed = origin in _ALLOWED_ORIGINS

            async def send_with_cors(message: Any) -> None:
                """Intercept outgoing ASGI messages and inject CORS headers.

                Adds Access-Control-Allow-Origin and related CORS headers when
                the request Origin matches an allowed localhost origin. Requests
                from unlisted origins receive no CORS headers, causing browsers
                to block the cross-origin request.

                Args:
                    message: ASGI message dict — only ``http.response.start``
                        messages are modified; all other types are forwarded
                        unchanged.
                """
                if message["type"] == "http.response.start":
                    headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                    if is_allowed:
                        headers.extend([
                            (b"Access-Control-Allow-Origin", origin.encode("latin-1")),
                            (b"Access-Control-Allow-Methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                            (b"Access-Control-Allow-Headers", _ALLOWED_CORS_HEADERS),
                            (b"Vary", b"Origin"),
                        ])
                    else:
                        # Strip any wildcard CORS header that a downstream handler may have set
                        headers = [(k, v) for k, v in headers if k.lower() != b"access-control-allow-origin"]
                    message = {**message, "headers": headers}
                await send(message)

            await next_app(scope, receive, send_with_cors)

    # ── RequestIdMiddleware ───────────────────────────────────────────────────

    class RequestIdMiddleware(ASGIMiddleware):
        """Inject a per-request correlation ID into the structured logging context.

        Ports the ``_set_correlation_request_id`` Flask before_request hook from
        ``vetinari/web_ui.py`` (lines 102-118).  Reads the ``X-Request-ID``
        header when present, or generates a new UUID, then calls
        ``vetinari.structured_logging.set_request_id`` so every log line emitted
        during the request carries the same trace identifier.

        Failures are logged at WARNING but never block the request.
        """

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Set correlation ID before forwarding the request.

            Args:
                scope:    ASGI scope dict carrying the request headers.
                receive:  ASGI receive callable.
                send:     ASGI send callable.
                next_app: The next ASGI app in the middleware stack.
            """
            if scope["type"] == "http":
                request_id = _get_header(scope, b"x-request-id") or str(uuid.uuid4())
                try:
                    from vetinari.structured_logging import set_request_id

                    set_request_id(request_id)
                except Exception:
                    logger.warning("set_request_id unavailable — skipping request ID injection for this request")
            await next_app(scope, receive, send)

    # ── UserActivityMiddleware ────────────────────────────────────────────────

    class UserActivityMiddleware(ASGIMiddleware):
        """Notify the idle training scheduler that a user request arrived.

        Ports the ``_record_user_activity`` Flask before_request hook from
        ``vetinari/web_ui.py`` (lines 84-99).  Without this hook the
        IdleDetector would consider the system idle 5 minutes after server
        start regardless of actual traffic — training could fire while the
        user is actively working.

        Failures are logged at WARNING but never block the request.
        """

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Record user activity before forwarding the request.

            Args:
                scope:    ASGI scope dict.
                receive:  ASGI receive callable.
                send:     ASGI send callable.
                next_app: The next ASGI app in the middleware stack.
            """
            if scope["type"] == "http":
                try:
                    from vetinari.training.idle_scheduler import get_idle_detector

                    detector = get_idle_detector()
                    if detector is not None:
                        detector.record_activity()
                except Exception:
                    logger.warning("Idle activity recording unavailable")
            await next_app(scope, receive, send)

    # ── RemoteMutationGuardMiddleware ─────────────────────────────────────────

    class RemoteMutationGuardMiddleware(ASGIMiddleware):
        """Block mutation requests from non-localhost origins when no admin token is configured.

        When ``VETINARI_ADMIN_TOKEN`` is not set the server has no bearer-auth
        layer, so any remote client could call mutation endpoints
        (POST/PUT/DELETE/PATCH) without credentials.  This middleware closes
        that hole by returning 401 for all such requests unless:

        - The request originates from a loopback address (127.0.0.1 / ::1), OR
        - ``VETINARI_ALLOW_UNAUTHENTICATED_REMOTE_MUTATION=1`` is set (explicit
          operator opt-in — useful for trusted internal networks).

        When ``VETINARI_ADMIN_TOKEN`` IS set the middleware is transparent and
        passes all requests through; the token-based admin guard on protected
        routes handles auth independently.

        Decision: fail-closed no-admin-token remote mutation policy (ADR-0098).
        """

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Intercept mutation requests and enforce the no-token remote-access policy.

            Args:
                scope:    ASGI scope dict for the current request.
                receive:  ASGI receive callable.
                send:     ASGI send callable (used to send 401 when blocked).
                next_app: The next ASGI app in the middleware stack.
            """
            if scope["type"] != "http":
                await next_app(scope, receive, send)
                return

            method: str = scope.get("method", "")
            if method not in _MUTATION_METHODS:
                await next_app(scope, receive, send)
                return

            # Transparent when a token is configured — the admin_guard handles auth.
            admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
            if admin_token:
                await next_app(scope, receive, send)
                return

            # Operator has explicitly opted in to unauthenticated remote mutation.
            opt_in = os.environ.get("VETINARI_ALLOW_UNAUTHENTICATED_REMOTE_MUTATION", "")
            if opt_in.lower() in ("1", "true", "yes"):
                await next_app(scope, receive, send)
                return

            # Determine the real client IP.  X-Forwarded-For is only trusted
            # when VETINARI_TRUSTED_PROXY is set to prevent spoofing.
            trusted_proxy = os.environ.get("VETINARI_TRUSTED_PROXY", "").lower() in ("1", "true", "yes")
            client_tuple = scope.get("client")
            direct_ip = client_tuple[0] if client_tuple else ""

            if trusted_proxy:
                forwarded = _get_header(scope, b"x-forwarded-for")
                remote_ip = forwarded.split(",")[0].strip() if forwarded else direct_ip
            else:
                remote_ip = direct_ip

            if remote_ip in _LOCALHOST_IPS:
                await next_app(scope, receive, send)
                return

            # Remote mutation with no token configured — block fail-closed.
            path: str = scope.get("path", "")
            try:
                from vetinari.security.redaction import redact_text

                path_for_log = redact_text(path)
                remote_ip_for_log = redact_text(remote_ip)
            except Exception:
                path_for_log = path
                remote_ip_for_log = remote_ip
            logger.warning(
                "RemoteMutationGuard: blocked %s %s from %s — no VETINARI_ADMIN_TOKEN configured"
                " (set VETINARI_ALLOW_UNAUTHENTICATED_REMOTE_MUTATION=1 to permit)",
                method,
                path_for_log,
                remote_ip_for_log,
            )
            body = json.dumps({
                "status_code": 401,
                "detail": (
                    "Remote mutation requires VETINARI_ADMIN_TOKEN to be configured. "
                    "Set VETINARI_ALLOW_UNAUTHENTICATED_REMOTE_MUTATION=1 to disable this check."
                ),
            }).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("utf-8")),
                    (b"www-authenticate", b'Bearer realm="vetinari"'),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })

    # ── JsonDepthGuardMiddleware ──────────────────────────────────────────────

    async def _replay_and_forward(
        scope: Scope,
        raw_body: bytes,
        send: Send,
        next_app: ASGIApp,
    ) -> None:
        """Replay a pre-consumed body and forward the request to the next ASGI app.

        ASGI receive channels are single-use streams.  When middleware reads the
        body it must replace the ``receive`` callable with one that re-emits the
        buffered bytes so downstream handlers see a complete body.

        Args:
            scope:    ASGI scope dict for the current request.
            raw_body: The already-consumed raw request body bytes.
            send:     ASGI send callable.
            next_app: The next ASGI app in the middleware stack.
        """
        body_sent = False

        async def replay_receive() -> dict:
            """Replay the pre-consumed request body to the ASGI app.

            On the first call, returns the buffered body. Subsequent calls return
            a disconnect message to indicate the stream is exhausted.

            Returns:
                An ASGI message dict with type "http.request" (first call) or
                "http.disconnect" (subsequent calls).
            """
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            # Subsequent calls block; in practice Litestar reads body exactly once.
            return {"type": "http.disconnect"}

        await next_app(scope, replay_receive, send)

    # Maximum JSON nesting depth allowed in request bodies.  The deeply-nested
    # fuzz case {"a":{"b":{"c":{"d":{"e":{"f":{"g":"deep"}}}}}}} has depth 7.
    # Legitimate API payloads (task inputs, agent configs) go at most 3-4 deep.
    # A threshold of 5 rejects all depth-≥6 bodies while leaving real payloads
    # untouched.  Decision: systemic JSON depth guard (ADR-0099).
    _MAX_JSON_DEPTH: int = 5

    def _json_depth(obj: Any, current: int = 0) -> int:
        """Return the maximum nesting depth of a parsed JSON value.

        Args:
            obj:     Parsed Python value (dict, list, str, int, etc.).
            current: Depth of the current node (0 = top-level).

        Returns:
            Maximum depth found in the subtree rooted at ``obj``.
        """
        if isinstance(obj, dict):
            if not obj:
                return current
            return max(_json_depth(v, current + 1) for v in obj.values())
        if isinstance(obj, list):
            if not obj:
                return current
            return max(_json_depth(v, current + 1) for v in obj)
        return current

    # Maximum allowed length for any individual JSON string key or value.
    # Protects against key-bombing attacks ({"x"*10000: "v"}) that are
    # syntactically valid JSON but pathological for logging, storage, and
    # downstream parsing.
    _MAX_STRING_LENGTH: int = 4096

    def _contains_control_chars(obj: Any) -> bool:
        """Return True when any string value in a parsed JSON tree contains ASCII control characters.

        Control characters in the range U+0000-U+001F (including null bytes) are not
        valid in API string fields and are a common vector for fuzzing and injection
        attacks.  Rejecting them at the middleware layer protects every handler
        uniformly without per-handler validation logic.

        Args:
            obj: Parsed Python value from ``json.loads()`` — dict, list, str, int,
                float, bool, or None.

        Returns:
            True when ``obj`` or any nested value contains a string with a character
            whose code point is less than 0x20 (space).  False otherwise.
        """
        if isinstance(obj, str):
            return any(ord(c) < 0x20 for c in obj)
        if isinstance(obj, dict):
            return any(_contains_control_chars(k) or _contains_control_chars(v) for k, v in obj.items())
        if isinstance(obj, list):
            return any(_contains_control_chars(v) for v in obj)
        return False

    def _has_oversized_strings(obj: Any) -> bool:
        """Return True when any string key or value exceeds ``_MAX_STRING_LENGTH``.

        Very long keys and values are syntactically valid JSON but pathological for
        logging, database storage, and downstream processing.  Rejecting them here
        ensures handlers never receive unbounded string fields.

        Args:
            obj: Parsed Python value — dict, list, str, or scalar.

        Returns:
            True when any string in the tree exceeds ``_MAX_STRING_LENGTH`` characters.
        """
        if isinstance(obj, str):
            return len(obj) > _MAX_STRING_LENGTH
        if isinstance(obj, dict):
            return any(_has_oversized_strings(k) or _has_oversized_strings(v) for k, v in obj.items())
        if isinstance(obj, list):
            return any(_has_oversized_strings(v) for v in obj)
        return False

    def _has_non_finite_floats(obj: Any) -> bool:
        """Return True when any numeric value in a parsed JSON tree is non-finite.

        Python's ``json.loads`` accepts the non-standard tokens ``NaN``,
        ``Infinity``, and ``-Infinity``.  RFC 8259 forbids them.  Rejecting
        non-finite floats prevents them from reaching handlers that may
        serialise them back to JSON and trigger a ``ValueError``.

        Args:
            obj: Parsed Python value — dict, list, float, or scalar.

        Returns:
            True when any float in the tree is ``nan``, ``inf``, or ``-inf``.
        """
        import math

        if isinstance(obj, float):
            return not math.isfinite(obj)
        if isinstance(obj, dict):
            return any(_has_non_finite_floats(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_non_finite_floats(v) for v in obj)
        return False

    def _has_duplicate_keys(raw_body: bytes) -> bool:
        """Return True when the JSON object at the top level (or nested) has duplicate keys.

        Python's ``json.loads`` silently discards all but the last value for a
        repeated key, making ``{"a":1,"a":2}`` indistinguishable from ``{"a":2}``
        after parsing.  Detecting duplicates requires re-parsing with a custom
        ``object_pairs_hook`` that preserves the raw key list.

        Args:
            raw_body: The raw request body bytes.

        Returns:
            True when any object in the JSON document contains a repeated key.
        """
        seen_duplicates: list[bool] = []

        def _check_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            keys = [k for k, _ in pairs]
            if len(keys) != len(set(keys)):
                seen_duplicates.append(True)
            return dict(pairs)

        try:
            json.loads(raw_body, object_pairs_hook=_check_pairs)
        except (ValueError, UnicodeDecodeError):
            logger.warning("Could not parse JSON body for duplicate-key check — malformed request body, skipping check")
            return False
        return bool(seen_duplicates)

    class JsonDepthGuardMiddleware(ASGIMiddleware):
        """Reject JSON request bodies whose nesting depth exceeds ``_MAX_JSON_DEPTH``.

        Litestar's dataclass and dict deserialization silently ignores unknown
        keys, so a deeply-nested body like ``{"a":{"b":{"c":...}}}`` passes
        validation and reaches the handler with no error.  This middleware
        intercepts the raw body *before* deserialization and returns 422 for
        any ``application/json`` body whose parse-tree depth exceeds the
        configured threshold.

        Only POST and PUT requests are checked; DELETE/GET/PATCH carry no
        body in the normal case.  Non-JSON content types are forwarded as-is.
        Non-parseable bodies (truncated JSON, binary garbage) are forwarded
        so that Litestar's own deserialization layer can produce the 422.

        Decision: systemic JSON depth guard (ADR-0099).
        """

        _JSON_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})

        async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
            """Enforce the JSON depth limit before dispatching to the next layer.

            Args:
                scope:    ASGI scope dict for the current request.
                receive:  ASGI receive callable.
                send:     ASGI send callable.
                next_app: The next ASGI app in the middleware stack.
            """
            if scope["type"] != "http":
                await next_app(scope, receive, send)
                return

            method: str = scope.get("method", "")
            if method not in self._JSON_METHODS:
                await next_app(scope, receive, send)
                return

            content_type = _get_header(scope, b"content-type")
            if "application/json" not in content_type:
                await next_app(scope, receive, send)
                return

            # Consume the full body so we can inspect it, then replay it.
            body_chunks: list[bytes] = []
            more_body = True
            while more_body:
                message = await receive()
                body_chunks.append(message.get("body", b""))
                more_body = message.get("more_body", False)
            raw_body = b"".join(body_chunks)

            # Try to parse and measure depth.  If the body is not valid JSON
            # we forward it and let Litestar return 422 via its own decoder.
            try:
                parsed = json.loads(raw_body)
            except (ValueError, UnicodeDecodeError) as exc:
                # Let Litestar handle the decode error (returns 422 already).
                path = scope.get("path", "")
                logger.warning("Malformed JSON body on %s — forwarding to Litestar for standard 422 response (%s)", path, type(exc).__name__)
                await _replay_and_forward(scope, raw_body, send, next_app)
                return

            # Reject non-object JSON bodies (null, arrays, bare strings/numbers).
            # All mutating endpoints in this API expect a JSON object at the top
            # level.  Litestar's dict-typed handlers silently accept or coerce
            # other JSON types, so we enforce the object constraint here.
            if not isinstance(parsed, dict):
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body is not a JSON object (got %s)",
                    method,
                    path,
                    type(parsed).__name__,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": "Request body must be a JSON object.",
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            if _json_depth(parsed) > _MAX_JSON_DEPTH:
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body nesting depth exceeds %d",
                    method,
                    path,
                    _MAX_JSON_DEPTH,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": (f"Request body nesting depth exceeds the maximum of {_MAX_JSON_DEPTH}."),
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            if _contains_control_chars(parsed):
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body contains ASCII control characters",
                    method,
                    path,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": "Request body contains invalid control characters.",
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            if _has_non_finite_floats(parsed):
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body contains non-finite float (NaN/Infinity)",
                    method,
                    path,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": "Request body contains non-finite numeric values (NaN or Infinity).",
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            if _has_oversized_strings(parsed):
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body contains string exceeding %d chars",
                    method,
                    path,
                    _MAX_STRING_LENGTH,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": f"Request body contains a string field exceeding the maximum length of {_MAX_STRING_LENGTH}.",
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            if _has_duplicate_keys(raw_body):
                path = scope.get("path", "")
                logger.warning(
                    "JsonDepthGuard: rejected %s %s — body contains duplicate JSON object keys",
                    method,
                    path,
                )
                error_body = json.dumps({
                    "status_code": 422,
                    "detail": "Request body contains duplicate object keys.",
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(error_body)).encode("utf-8")),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                    "more_body": False,
                })
                return

            # All checks passed — replay body and forward.
            await _replay_and_forward(scope, raw_body, send, next_app)

else:
    # Stubs so the module is importable when Litestar is not installed.
    # create_app() guards on _LITESTAR_AVAILABLE before using these.
    SecurityHeadersMiddleware = None  # type: ignore[assignment,misc]
    CORSMiddleware = None  # type: ignore[assignment,misc]
    RequestIdMiddleware = None  # type: ignore[assignment,misc]
    UserActivityMiddleware = None  # type: ignore[assignment,misc]
    RemoteMutationGuardMiddleware = None  # type: ignore[assignment,misc]
    JsonDepthGuardMiddleware = None  # type: ignore[assignment,misc]


__all__ = [
    "CORSMiddleware",
    "JsonDepthGuardMiddleware",
    "RemoteMutationGuardMiddleware",
    "RequestIdMiddleware",
    "SecurityHeadersMiddleware",
    "UserActivityMiddleware",
]
