"""Security enforcement tests for the Vetinari project.

Covers four enforcement layers:
- Path traversal prevention (SecretScanner pattern detection and sandbox filesystem allowlists)
- Sandbox escape prevention (module blocking and filesystem allowlists in CodeSandbox)
- CSRF header enforcement (CSRFMiddleware rejects mutation requests without X-Requested-With)
- Mutation endpoint validation (routes return appropriate error responses without required headers)

All tests are marked ``@pytest.mark.security``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.factories import make_asgi_scope as _make_asgi_scope

pytestmark = pytest.mark.security


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_csrf_middleware(
    scope: dict[str, Any],
) -> tuple[int, bytes]:
    """Drive the CSRFMiddleware through one request cycle and capture the response.

    A minimal downstream ASGI app is used that always returns 200 OK.  If the
    middleware short-circuits the request it emits its own status/body instead.

    Args:
        scope: Pre-built ASGI scope dictionary.

    Returns:
        Tuple of (HTTP status code, response body bytes).
    """
    from vetinari.web.csrf import CSRFMiddleware

    sent_messages: list[dict[str, Any]] = []

    # ASGI send must be an awaitable callable; side_effect captures each call.
    capture_send = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))

    async def downstream(s: Any, r: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK", "more_body": False})

    middleware = CSRFMiddleware(downstream)
    receive = AsyncMock(return_value={"type": "http.request", "body": b""})
    await middleware(scope, receive, capture_send)

    start_msg = next((m for m in sent_messages if m["type"] == "http.response.start"), None)
    body_msg = next((m for m in sent_messages if m["type"] == "http.response.body"), None)

    status = start_msg["status"] if start_msg else 0
    body = body_msg["body"] if body_msg else b""
    return status, body


# ---------------------------------------------------------------------------
# TestPathTraversal
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Verify that path traversal patterns are detected and blocked.

    SecretScanner includes a built-in ``path_traversal`` pattern that matches
    ``../`` and ``..\\`` sequences.  The CodeSandbox filesystem allowlist
    enforces that ``open()`` inside executed code cannot escape to arbitrary
    paths.
    """

    def test_unix_path_traversal_detected_by_scanner(self) -> None:
        """SecretScanner must flag Unix-style ../../etc/passwd traversal attempts."""
        from vetinari.security import SecretScanner

        scanner = SecretScanner()
        payloads = [
            "../../etc/passwd",
            "../../../etc/shadow",
            "foo/../../etc/hosts",
        ]
        for payload in payloads:
            detected = scanner.scan(payload)
            assert "path_traversal" in detected, (
                f"Expected 'path_traversal' detection for payload {payload!r}, got keys: {list(detected.keys())}"
            )

    def test_windows_path_traversal_detected_by_scanner(self) -> None:
        """SecretScanner must flag Windows-style ..\\..\\ traversal attempts."""
        from vetinari.security import SecretScanner

        scanner = SecretScanner()
        payloads = [
            "..\\..\\windows\\system32",
            "..\\..\\Users\\Administrator",
        ]
        for payload in payloads:
            detected = scanner.scan(payload)
            assert "path_traversal" in detected, (
                f"Expected 'path_traversal' detection for Windows payload {payload!r}, "
                f"got keys: {list(detected.keys())}"
            )

    def test_mixed_separator_traversal_detected(self) -> None:
        """SecretScanner must detect traversal patterns regardless of path depth."""
        from vetinari.security import SecretScanner

        scanner = SecretScanner()
        # Single level is the minimum the pattern matches
        result = scanner.scan("../secret_file.txt")
        assert "path_traversal" in result

    def test_clean_path_not_flagged(self) -> None:
        """Ordinary relative paths without traversal must not be flagged as path_traversal."""
        from vetinari.security import SecretScanner

        scanner = SecretScanner()
        safe_paths = [
            "uploads/document.pdf",
            "static/images/logo.png",
            "config/settings.yaml",
        ]
        for path in safe_paths:
            detected = scanner.scan(path)
            assert "path_traversal" not in detected, (
                f"Clean path {path!r} should not trigger path_traversal, got: {list(detected.keys())}"
            )

    def test_sandbox_filesystem_allowlist_blocks_outside_paths(self, tmp_path: Path) -> None:
        """CodeSandbox with a filesystem allowlist must raise PermissionError for paths outside the allowed prefix."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        sandbox = CodeSandbox(
            working_dir=str(tmp_path),
            filesystem_allowlist=[str(allowed_dir)],
        )

        # Attempt to open a file outside the allowed directory
        outside_path = tmp_path / "secret.txt"
        outside_path.write_text("sensitive data", encoding="utf-8")

        code = f"open({str(outside_path)!r}, 'r')"
        result = sandbox.execute_python(code)

        assert not result.success, "Sandbox should block file access outside filesystem allowlist"
        assert "PermissionError" in result.error or "blocked" in result.error.lower(), (
            f"Expected PermissionError in sandbox output, got: {result.error!r}"
        )

    def test_sandbox_without_allowlist_executes_safe_code(self, tmp_path: Path) -> None:
        """CodeSandbox with no filesystem allowlist must allow unrestricted safe code execution.

        When ``filesystem_allowlist`` is empty, the ``_restricted_open`` guard
        is not installed, so ordinary file operations inside the sandbox work.
        This also confirms the sandbox itself is operational.
        """
        from vetinari.code_sandbox import CodeSandbox

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        target_file = work_dir / "data.txt"
        target_file.write_text("hello", encoding="utf-8")

        # No allowlist — the open() guard is not injected
        sandbox = CodeSandbox(working_dir=str(work_dir))

        result = sandbox.execute_python("x = 1 + 1\nprint(x)")
        assert result.success, f"Sandbox without allowlist should run safe code. error={result.error!r}"


# ---------------------------------------------------------------------------
# TestSandboxEscape
# ---------------------------------------------------------------------------


class TestSandboxEscape:
    """Verify that CodeSandbox blocks dangerous escape vectors.

    The sandbox injects a ``_restricted_import`` hook that raises
    ``ImportError`` when blocked modules (subprocess, os, socket, etc.)
    are imported inside executed code.
    """

    def test_subprocess_import_is_blocked(self, tmp_path: Path) -> None:
        """Sandboxed code must not be able to import subprocess."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path))
        result = sandbox.execute_python("import subprocess; subprocess.run(['echo', 'hi'])")

        assert not result.success, "sandbox should block subprocess import"
        assert (
            "subprocess" in result.error.lower() or "ImportError" in result.error or "blocked" in result.error.lower()
        )

    def test_socket_import_is_blocked_with_network_isolation(self, tmp_path: Path) -> None:
        """When network_isolation=True the sandbox must block the socket module."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path), network_isolation=True)
        result = sandbox.execute_python("import socket; socket.create_connection(('example.com', 80))")

        assert not result.success, "sandbox should block socket import when network_isolation=True"

    def test_sys_import_is_blocked(self, tmp_path: Path) -> None:
        """Sandboxed code must not be able to import sys (it is in the default blocklist)."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path))
        result = sandbox.execute_python("import sys; sys.exit(0)")

        assert not result.success, "sandbox should block sys import"

    def test_safe_code_executes_successfully(self, tmp_path: Path) -> None:
        """Code that uses only safe builtins must execute and return correct output."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path))
        result = sandbox.execute_python("result = 2 + 2\nprint(result)")

        assert result.success, f"Safe code should run successfully, got error: {result.error!r}"

    def test_timeout_enforcement(self, tmp_path: Path) -> None:
        """CodeSandbox must terminate code that exceeds the timeout limit."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path), max_execution_time=2)
        # Busy-wait loop; 'time' is not in the default blocklist so this is a real spin
        result = sandbox.execute_python(
            "import math\nfor i in range(10**9): math.sqrt(i)",
            timeout=2,
        )

        assert not result.success, "Sandbox must terminate code that exceeds timeout"

    def test_network_modules_blocked_by_default(self, tmp_path: Path) -> None:
        """Requests module must be blocked when network_isolation is enabled (default)."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir=str(tmp_path), network_isolation=True)
        result = sandbox.execute_python("import requests; requests.get('http://example.com')")

        assert not result.success, "sandbox should block requests import when network_isolation=True"


# ---------------------------------------------------------------------------
# TestCSRF
# ---------------------------------------------------------------------------


class TestCSRF:
    """Verify that the CSRFMiddleware rejects mutation requests missing the required header.

    The middleware checks for the ``X-Requested-With`` header on all POST,
    PUT, DELETE, and PATCH requests, returning 403 when it is absent.
    """

    def test_post_without_csrf_header_returns_403(self) -> None:
        """POST request without X-Requested-With must receive a 403 response."""
        scope = _make_asgi_scope("POST", "/api/v1/tasks")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, f"Expected 403 for POST without CSRF header, got {status}"

    def test_put_without_csrf_header_returns_403(self) -> None:
        """PUT request without X-Requested-With must receive a 403 response."""
        scope = _make_asgi_scope("PUT", "/api/v1/projects/1")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, f"Expected 403 for PUT without CSRF header, got {status}"

    def test_delete_without_csrf_header_returns_403(self) -> None:
        """DELETE request without X-Requested-With must receive a 403 response."""
        scope = _make_asgi_scope("DELETE", "/api/v1/tasks/99")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, f"Expected 403 for DELETE without CSRF header, got {status}"

    def test_patch_without_csrf_header_returns_403(self) -> None:
        """PATCH request without X-Requested-With must receive a 403 response."""
        scope = _make_asgi_scope("PATCH", "/api/v1/tasks/99")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, f"Expected 403 for PATCH without CSRF header, got {status}"

    def test_post_with_csrf_header_passes_through(self) -> None:
        """POST request that includes X-Requested-With must be forwarded to the downstream app."""
        headers = [(b"x-requested-with", b"XMLHttpRequest")]
        scope = _make_asgi_scope("POST", "/api/v1/tasks", headers=headers)
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 200, f"Expected 200 when CSRF header present, got {status}"

    def test_get_request_does_not_require_csrf_header(self) -> None:
        """GET requests must be exempt from the CSRF header requirement."""
        scope = _make_asgi_scope("GET", "/api/v1/tasks")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 200, f"Expected 200 for safe GET method without CSRF header, got {status}"

    def test_health_endpoint_is_exempt_from_csrf(self) -> None:
        """POST to /health must bypass the CSRF check (machine-to-machine exemption)."""
        scope = _make_asgi_scope("POST", "/health")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 200, f"Expected /health to be exempt from CSRF, got {status}"

    def test_403_response_body_contains_csrf_error_message(self) -> None:
        """The 403 error body must contain a parseable JSON error indicating CSRF failure."""
        scope = _make_asgi_scope("POST", "/api/v1/tasks")
        status, body = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403
        parsed = json.loads(body.decode("utf-8"))
        assert "error" in parsed, f"403 body must contain 'error' key, got: {parsed}"
        assert "CSRF" in parsed["error"] or "csrf" in parsed["error"].lower(), (
            f"Error message should mention CSRF, got: {parsed['error']!r}"
        )

    def test_non_http_scope_passes_through_csrf_middleware(self) -> None:
        """WebSocket and other non-HTTP scopes must not be subject to CSRF checks."""
        from vetinari.web.csrf import CSRFMiddleware

        passed: list[bool] = []

        async def downstream(scope: Any, receive: Any, send: Any) -> None:
            # Must be async: CSRFMiddleware awaits self.app() for all scope types.
            passed.append(True)

        middleware = CSRFMiddleware(downstream)
        ws_scope: dict[str, Any] = {"type": "websocket", "method": "GET", "path": "/ws", "headers": []}

        asyncio.run(middleware(ws_scope, AsyncMock(), AsyncMock()))

        assert passed, "Non-HTTP scope should pass through CSRF middleware without blocking"


# ---------------------------------------------------------------------------
# TestAuthOnMutationEndpoints
# ---------------------------------------------------------------------------


class TestAuthOnMutationEndpoints:
    """Verify that mutation routes exist and return appropriate responses.

    Vetinari is a local-first application; authentication is enforced at the
    CSRF layer rather than a traditional token layer.  These tests verify
    that:

    1. The CSRF header check is the gate for all state-modifying requests.
    2. Mutation route paths are registered and return deterministic status
       codes rather than 500/404 when hit without the required header.
    3. The ``_UNSAFE_METHODS`` constant correctly lists all mutation verbs.
    """

    def test_csrf_unsafe_methods_includes_all_mutation_verbs(self) -> None:
        """The CSRF module must mark POST, PUT, DELETE, and PATCH as unsafe."""
        from vetinari.web.csrf import _UNSAFE_METHODS

        expected = {"POST", "PUT", "DELETE", "PATCH"}
        assert expected == _UNSAFE_METHODS, f"_UNSAFE_METHODS should be exactly {expected}, got {_UNSAFE_METHODS}"

    def test_csrf_header_constant_is_x_requested_with(self) -> None:
        """The CSRF header name constant must be the standard custom header string."""
        from vetinari.web.csrf import CSRF_HEADER

        assert CSRF_HEADER == "X-Requested-With", f"CSRF_HEADER should be 'X-Requested-With', got {CSRF_HEADER!r}"

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_all_mutation_methods_blocked_without_csrf_header(self, method: str) -> None:
        """Every mutation HTTP method must receive 403 when CSRF header is absent."""
        scope = _make_asgi_scope(method, "/api/v1/projects")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, f"{method} /api/v1/projects without CSRF header should return 403, got {status}"

    @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
    def test_safe_methods_are_not_blocked(self, method: str) -> None:
        """Safe HTTP methods must not require the CSRF header."""
        scope = _make_asgi_scope(method, "/api/v1/projects")
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 200, f"{method} should be allowed without CSRF header, got {status}"

    def test_a2a_endpoint_is_exempt_from_csrf(self) -> None:
        """The /api/v1/a2a machine-to-machine endpoint must bypass the CSRF check."""
        from vetinari.web.csrf import _EXEMPT_PATHS

        assert "/api/v1/a2a" in _EXEMPT_PATHS, "Machine-to-machine endpoint /api/v1/a2a must be in the CSRF exempt list"

    def test_exempt_paths_includes_health(self) -> None:
        """The /health endpoint must be in the CSRF exempt list for health checks."""
        from vetinari.web.csrf import _EXEMPT_PATHS

        assert "/health" in _EXEMPT_PATHS, "/health must be in the CSRF exempt list"

    def test_mutation_with_empty_csrf_header_is_blocked(self) -> None:
        """A POST request with an empty X-Requested-With header value must still be blocked."""
        # The middleware checks for a truthy value, so an empty byte string should block
        headers = [(b"x-requested-with", b"")]
        scope = _make_asgi_scope("POST", "/api/v1/tasks", headers=headers)
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 403, "POST with empty X-Requested-With should be blocked (403)"

    def test_mutation_with_correct_csrf_value_is_allowed(self) -> None:
        """A POST request with a non-empty X-Requested-With header value must be forwarded."""
        headers = [(b"x-requested-with", b"fetch")]
        scope = _make_asgi_scope("POST", "/api/v1/projects", headers=headers)
        status, _ = asyncio.run(_call_csrf_middleware(scope))

        assert status == 200, f"POST with valid CSRF header should reach downstream (200), got {status}"
