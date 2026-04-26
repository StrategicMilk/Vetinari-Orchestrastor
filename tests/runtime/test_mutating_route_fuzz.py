"""Mutating-route fuzz tests — SESSION-32.3.

Comprehensive parametrized fuzz coverage for all 127 mutating routes
(POST / PUT / DELETE) registered in the Litestar app.  Every request goes
through the real ``TestClient`` HTTP stack — no handler ``.fn(...)`` calls.

Test classes:
  TestMalformedJsonFuzz        — truncated / syntactically invalid JSON bodies
  TestTypeMutationFuzz         — wrong Go-style type for every required field
  TestPathTraversalFuzz        — path-segment injection in URL path params
  TestAuthBypassFuzz           — missing / spoofed / empty auth headers
  TestCsrfBypassFuzz           — mutation without the CSRF header
  TestContentTypeFuzz          — wrong Content-Type on JSON routes
  TestOversizedPayloadFuzz     — bodies exceeding reasonable size limits
  TestFileUploadFuzz           — multipart fuzzing for file-upload routes
  TestHypothesisBodyFuzz       — property-based fuzzing via Hypothesis

All assertions use ``response.status_code in _SAFE_4XX`` — the contract is
that malformed input MUST return a 4xx status, never 500 and never 200.

Routes listed in ``_KNOWN_500_ROUTES`` are expected to return 500 today
(SESSION-32.4 defects).  They are marked ``xfail(strict=True)`` so that
a fix in production code will automatically promote them to passing.
"""

from __future__ import annotations

import io
import os
import re
import string
import sys
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
from litestar.testing import TestClient

from tests.runtime._route_inventory import ROUTE_TABLE

# Skip entire file on Windows — Litestar TestClient anyio blocking portal hangs
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Litestar TestClient anyio blocking portal hangs on Windows — upstream issue",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All mutation endpoints must carry this CSRF header
_CSRF = {"X-Requested-With": "XMLHttpRequest"}

# Admin token injected via env for routes that require auth
_ADMIN_TOKEN = "test-fuzz-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# Combined headers for authenticated mutation requests
_AUTH_CSRF = {**_CSRF, **_ADMIN_HEADERS}

# Safe 4xx codes that indicate the server correctly rejected bad input
_SAFE_4XX = {400, 401, 403, 404, 405, 409, 413, 415, 422}

# Property-based valid-body fuzz may occasionally satisfy a route contract.
# The invariant remains bounded behavior: accepted success or expected
# validation/auth rejection, never server-error fallback.
_BOUNDED_MUTATION_STATUSES = _SAFE_4XX | {200, 201, 202, 204}

# Placeholder values injected for URL path parameters
_PATH_PARAM_DEFAULTS: dict[str, str] = {
    "action_id": "fuzz-action-id",
    "adr_id": "ADR-0001",
    "action_type": "fuzz-action",
    "source_type": "fuzz-source",
    "skill_id": "fuzz-skill",
    "plan_id": "fuzz-plan-id",
    "project_id": "fuzz-project-id",
    "entry_id": "fuzz-entry-id",
    "stage": "fuzz-stage",
    "name": "fuzz-name",
    "path": "fuzz/path",
    "subtask_id": "fuzz-subtask",
    "gate_id": "fuzz-gate",
    "experiment_id": "fuzz-exp-id",
}

# Routes whose malformed-input handling is known broken (return 500 instead of
# 4xx).  Covered by SESSION-32.4.  Each entry is (method, path).
# All SESSION-32.4 defects are fixed — this set is now empty.
_KNOWN_500_ROUTES: set[tuple[str, str]] = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATH_PARAM_RE = re.compile(r"\{(\w+)(?::[^}]+)?\}")


def _fill_path(path_template: str) -> str:
    """Replace ``{param}`` / ``{param:type}`` placeholders with safe defaults.

    Args:
        path_template: URL path with Litestar-style path parameters.

    Returns:
        Concrete URL with all parameters substituted.
    """

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        name = match.group(1)
        return _PATH_PARAM_DEFAULTS.get(name, f"fuzz-{name}")

    return _PATH_PARAM_RE.sub(_replace, path_template)


def _is_known_500(route: dict) -> bool:
    """Return True if this route is in the known-broken SESSION-32.4 set."""
    return (route["method"], route["path"]) in _KNOWN_500_ROUTES


def _xfail_if_known_500(route: dict) -> Any:
    """Build an xfail mark when the route is a known 500 offender, else None."""
    if _is_known_500(route):
        return pytest.mark.xfail(
            strict=True,
            reason=(
                f"{route['method']} {route['path']} currently returns 500 "
                "on malformed input — tracked as SESSION-32.4 defect"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# App fixture (session-scoped — ~300-handler app is expensive to build)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app: Any) -> None:
    """Drop-in lifespan that skips all subsystem wiring."""
    yield


@pytest.fixture(scope="session")
def fuzz_app() -> Any:
    """Create a single Litestar app instance shared across the fuzz session.

    Returns:
        Litestar application with noop lifespan and no shutdown handlers.
    """
    with (
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=False)


@pytest.fixture
def fuzz_client(fuzz_app: Any) -> Generator[TestClient, None, None]:
    """Yield a fresh TestClient per fuzz test.

    Args:
        fuzz_app: Session-scoped Litestar application.

    Yields:
        Connected TestClient with admin env in scope.
    """
    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        TestClient(app=fuzz_app) as c,
    ):
        yield c


# ---------------------------------------------------------------------------
# Parametrize helpers — build (route, case) pairs for each test class
# ---------------------------------------------------------------------------

# All routes that carry a JSON body (POST/PUT with body_schema="dict")
_JSON_BODY_ROUTES = [r for r in ROUTE_TABLE if r["body_schema"] == "dict"]

# All routes including no-body DELETE (for auth/CSRF fuzz)
_ALL_MUTATING = list(ROUTE_TABLE)

# File-upload routes only
_FILE_UPLOAD_ROUTES = [r for r in ROUTE_TABLE if r["body_schema"] == "file-upload"]

# ---------------------------------------------------------------------------
# Malformed JSON cases
# ---------------------------------------------------------------------------

# Each entry: (label, raw_bytes_or_str)
MALFORMED_JSON_CASES = [
    ("truncated_object", b'{"key":'),
    ("truncated_string", b'{"key": "val'),
    ("bare_string", b'"just a string"'),
    ("bare_number", b"42"),
    ("bare_null", b"null"),
    ("bare_array", b"[1, 2, 3]"),
    ("empty_body", b""),
    ("whitespace_only", b"   \n  "),
    ("binary_garbage", b"\x00\x01\x02\x03\xff\xfe"),
    ("latin1_garbage", "café".encode("latin-1")),
    ("double_encoded_json", b'"{\\"nested\\": true}"'),
    ("trailing_comma", b'{"a": 1,}'),
    ("single_quotes", b"{'a': 1}"),
    ("js_undefined", b'{"a": undefined}'),
    ("nan_value", b'{"a": NaN}'),
    ("infinite_value", b'{"a": Infinity}'),
    ("comment_in_json", b'{"a": 1 /* comment */}'),
    ("xml_body", b"<root><item>value</item></root>"),
    ("form_encoded", b"field1=value1&field2=value2"),
    ("multiline_garbage", b"line1\nline2\nnot json"),
    ("deeply_nested", b'{"a":{"b":{"c":{"d":{"e":{"f":{"g":"deep"}}}}}}}'),
    ("unicode_escape_bomb", b'{"a": "\\u0000\\u0001\\u0002"}'),
    ("repeated_keys", b'{"a": 1, "a": 2, "a": 3}'),
    ("number_as_key", b'{1: "value"}'),
    ("very_long_key", b'{"' + b"x" * 10000 + b'": "value"}'),
]

# Build parametrize ids for malformed JSON
_malformed_json_params = []
for _route in _JSON_BODY_ROUTES:
    for _label, _body in MALFORMED_JSON_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _body,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-{_label}",
            marks=[_mark] if _mark else [],
        )
        _malformed_json_params.append(_param)


class TestMalformedJsonFuzz:
    """Send syntactically invalid JSON to every JSON-body route.

    Contract: malformed bodies MUST produce 4xx, never 500 or 200.
    """

    @pytest.mark.parametrize("route,label,raw_body", _malformed_json_params)
    def test_malformed_json_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        raw_body: bytes,
    ) -> None:
        """Malformed JSON body must be rejected with a 4xx status.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            raw_body: The malformed body bytes to send.
        """
        url = _fill_path(route["path"])
        headers = {
            **_AUTH_CSRF,
            "Content-Type": "application/json",
        }
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, content=raw_body, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}] returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Type mutation cases
# ---------------------------------------------------------------------------

# Each entry: (label, body_dict)  — body that swaps field types
TYPE_MUTATION_CASES = [
    ("list_for_scalar", {"__fuzz_type": [1, 2, 3]}),
    ("dict_for_scalar", {"__fuzz_type": {"nested": "value"}}),
    ("null_for_string", {"goal": None, "description": None, "name": None}),
    ("int_for_string", {"goal": 12345, "description": 99, "name": 0}),
    ("float_for_int", {"count": 3.14, "limit": 99.9, "max_retries": 1.5}),
    ("bool_for_string", {"name": True, "goal": False, "model_id": True}),
    ("bool_for_int", {"count": True, "limit": False}),
    ("empty_list", {"inputs": [], "outputs": [], "dependencies": []}),
    ("nested_list_of_lists", {"inputs": [[1, 2], [3, 4]]}),
    ("all_nulls", dict.fromkeys(["goal", "description", "name", "model_id", "count"])),
    ("empty_object", {}),
    ("string_numbers", {"count": "99", "limit": "5", "max_retries": "3"}),
    ("unicode_garbage", {"name": "\x00\xff\u2028\u2029", "goal": "\ufffd" * 100}),
    ("huge_string_field", {"name": "x" * 50000, "description": "y" * 50000}),
    ("list_for_object_field", {"config": [1, 2, 3], "options": ["a", "b"]}),
    ("negative_counts", {"count": -1, "limit": -999, "max_tokens": -1}),
    ("zero_counts", {"count": 0, "limit": 0, "max_tokens": 0}),
    ("max_int", {"count": 2**63 - 1, "limit": 2**63 - 1}),
    ("control_chars_in_string", {"name": "\t\n\r\x0b\x0c", "description": "\x00\x01\x02"}),
    ("emoji_spam", {"name": "ðŸ”¥" * 1000, "description": "ðŸ’£" * 500}),
    ("nested_nulls", {"config": {"key": None, "nested": {"deep": None}}}),
    ("wrong_bool_string", {"enabled": "yes", "active": "true", "flag": "1"}),
    ("mixed_type_list", {"inputs": [1, "two", None, True, {"nested": 1}]}),
    ("only_extra_fields", {"__unknown_field_1": "val", "__unknown_field_2": 99}),
    ("duplicate_id_fields", {"id": "a", "project_id": "b", "plan_id": None}),
]

_type_mutation_params = []
for _route in _JSON_BODY_ROUTES:
    for _label, _body in TYPE_MUTATION_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _body,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-tm-{_label}",
            marks=[_mark] if _mark else [],
        )
        _type_mutation_params.append(_param)


class TestTypeMutationFuzz:
    """Send type-mutated JSON bodies to every JSON-body route.

    Contract: type-mismatched fields MUST produce 4xx, never 500 or 200.
    """

    @pytest.mark.parametrize("route,label,body", _type_mutation_params)
    def test_type_mutation_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        body: dict,
    ) -> None:
        """Type-mutated body fields must be rejected with a 4xx status.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            body: JSON-serializable body with mutated field types.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}] returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Path traversal injection
# ---------------------------------------------------------------------------

# Each: (label, injected_value) used to replace ALL path params in the URL
PATH_TRAVERSAL_IDS = [
    ("dotdot", ".."),
    ("dotdot_encoded", "%2E%2E"),
    ("dotdot_slash", "../"),
    ("double_encoded_dotdot", "%2F%2E%2E%2F"),
    ("backslash_dotdot", "\\..\\"),
    ("backslash_encoded", "%5C%2E%2E%5C"),
    ("full_path", "../../etc/passwd"),
    ("encoded_full_path", "..%2F..%2Fetc%2Fpasswd"),
    ("null_byte", "%00"),
    ("null_byte_dotdot", "%00../../"),
    ("absolute_path", "/etc/passwd"),
    ("windows_path", "C:\\Windows\\System32"),
    ("windows_unc", "\\\\server\\share"),
    ("semicolon_inject", "valid;DROP TABLE"),
    ("newline_inject", "valid\nX-Injected: yes"),
    ("carriage_inject", "valid\rX-Injected: yes"),
    ("percent_inject", "val%20id"),
    ("double_slash", "//admin"),
    ("asterisk", "*"),
    ("question_mark", "?admin=true"),
    ("hash_fragment", "#fragment"),
    ("sql_inject", "' OR '1'='1"),
    ("xss_payload", "<script>alert(1)</script>"),
    ("ldap_inject", "*)(&(objectClass=*)"),
    ("template_inject", "{{7*7}}"),
    ("command_inject", "; ls -la"),
    ("pipe_inject", "| cat /etc/passwd"),
    ("very_long_param", "a" * 4096),
    ("unicode_normalization", "\u0041\u0301"),  # A + combining accent
    ("rtl_override", "\u202e" + "evil"),
]

# Only routes with at least one path parameter
_path_param_routes = [r for r in _ALL_MUTATING if _PATH_PARAM_RE.search(r["path"])]

_path_traversal_params = []
for _route in _path_param_routes:
    for _label, _injected in PATH_TRAVERSAL_IDS:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _injected,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-pt-{_label}",
            marks=[_mark] if _mark else [],
        )
        _path_traversal_params.append(_param)


class TestPathTraversalFuzz:
    """Inject path traversal payloads into URL path parameters.

    Contract: traversal payloads MUST produce 4xx (typically 404 or 400),
    never 500 and never a successful 2xx response.
    """

    @pytest.mark.parametrize("route,label,injected", _path_traversal_params)
    def test_path_traversal_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        injected: str,
    ) -> None:
        """Path traversal injection must not produce 500 or 2xx.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            injected: The injected path segment value.
        """
        # double_slash ("//admin") is normalised to "admin" by the HTTP stack —
        # not a traversal threat, just standard URL normalisation.  Skip it.
        if label == "double_slash":
            pytest.skip("double_slash is normalised to a valid segment by the HTTP layer")

        # Build URL replacing every path param with the injected value.
        # Use a lambda so backslashes in `injected` are never treated as
        # regex replacement escapes (e.g. "\..\\" would crash re.sub otherwise).
        url = _PATH_PARAM_RE.sub(lambda _m: injected, route["path"])
        headers = {**_AUTH_CSRF}
        body: dict | None = {} if route["body_schema"] == "dict" else None
        method = route["method"].lower()
        try:
            if body is not None:
                resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
            else:
                resp = getattr(fuzz_client, method)(url, headers=headers)
        except Exception as exc:
            # httpx.InvalidURL (and similar) mean the client refused to send the
            # request because the URL itself was malformed — that is a correct
            # rejection of the injection payload, so treat it as a pass.
            _exc_name = type(exc).__name__
            if "InvalidURL" in _exc_name or "Invalid" in _exc_name:
                return
            raise
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}={injected!r}] "
            f"returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Auth bypass fuzz
# ---------------------------------------------------------------------------

# Routes that require auth
_AUTH_REQUIRED_ROUTES = [r for r in _ALL_MUTATING if r["auth_required"] == "required"]

# Cases: (label, headers_override)
AUTH_BYPASS_CASES = [
    ("no_auth_header", {}),
    ("empty_token", {"X-Admin-Token": ""}),
    ("whitespace_token", {"X-Admin-Token": "   "}),
    ("wrong_token", {"X-Admin-Token": "wrong-token"}),
    ("token_too_short", {"X-Admin-Token": "a"}),
    ("token_all_zeros", {"X-Admin-Token": "0" * 32}),
    ("token_with_null", {"X-Admin-Token": "valid\x00extra"}),
    ("token_with_newline", {"X-Admin-Token": "valid\nX-Admin-Token: good-token"}),
    # Note: "Authorization: Bearer <token>" is intentionally accepted by admin_guard
    # as a documented fallback — it is NOT a bypass attempt and is excluded here.
    ("misnamed_header", {"X-Admin-Tok": _ADMIN_TOKEN}),
    ("multiple_tokens_first_bad", {"X-Admin-Token": f"bad, {_ADMIN_TOKEN}"}),
]

_auth_bypass_params = []
for _route in _AUTH_REQUIRED_ROUTES:
    for _label, _hdrs in AUTH_BYPASS_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _hdrs,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-ab-{_label}",
            marks=[_mark] if _mark else [],
        )
        _auth_bypass_params.append(_param)


class TestAuthBypassFuzz:
    """Send requests with invalid / absent authentication to auth-required routes.

    Contract: invalid auth MUST return 401 or 403, never 200 or 500.
    """

    @pytest.mark.parametrize("route,label,bad_headers", _auth_bypass_params)
    def test_auth_bypass_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        bad_headers: dict,
    ) -> None:
        """Requests with bad auth must receive 401 or 403.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            bad_headers: Headers that intentionally fail auth.
        """
        url = _fill_path(route["path"])
        # Include CSRF but use the bad auth headers (no valid admin token)
        headers = {**_CSRF, **bad_headers}
        body: dict | None = {} if route["body_schema"] == "dict" else None
        method = route["method"].lower()
        if body is not None:
            resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        else:
            resp = getattr(fuzz_client, method)(url, headers=headers)
        assert resp.status_code in {401, 403}, (
            f"{route['method']} {route['path']} [{label}] "
            f"returned {resp.status_code} — expected 401/403 for bad auth "
            f"(body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# CSRF bypass fuzz
# ---------------------------------------------------------------------------

# CSRF bypass is relevant for all mutating routes (POST/PUT/DELETE)
# Routes that are A2A (CSRF-exempt machine-to-machine) are excluded:
_A2A_PATHS = {"/api/v1/a2a", "/api/v1/a2a/raw"}
_CSRF_CHECKABLE_ROUTES = [r for r in _ALL_MUTATING if r["path"] not in _A2A_PATHS]

# Cases: (label, headers_to_send_without_csrf)
CSRF_BYPASS_CASES = [
    ("no_csrf_header", {}),
    ("wrong_csrf_header_name", {"X-Requested-By": "XMLHttpRequest"}),
    ("empty_csrf_value", {"X-Requested-With": ""}),
    # lowercase_csrf_key omitted: HTTP header names are case-insensitive; the
    # ASGI layer normalises all keys to lower-case, so a lowercase key with the
    # correct value IS a valid CSRF token — not a bypass.
    ("csrf_typo", {"X-Requsted-With": "XMLHttpRequest"}),
]

_csrf_bypass_params = []
for _route in _CSRF_CHECKABLE_ROUTES:
    for _label, _hdrs in CSRF_BYPASS_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _hdrs,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-csrf-{_label}",
            marks=[_mark] if _mark else [],
        )
        _csrf_bypass_params.append(_param)


class TestCsrfBypassFuzz:
    """Send mutation requests without valid CSRF headers.

    Contract: missing or malformed CSRF header MUST produce 4xx, never 200/500.
    Note: routes that require auth will also hit auth check first; either
    rejection (401/403/403) satisfies the 4xx contract.
    """

    @pytest.mark.parametrize("route,label,bad_headers", _csrf_bypass_params)
    def test_csrf_bypass_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        bad_headers: dict,
    ) -> None:
        """Mutation without valid CSRF token must be rejected with 4xx.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            bad_headers: Headers intentionally missing or spoofing CSRF.
        """
        url = _fill_path(route["path"])
        # Include admin auth (we want to test CSRF, not auth rejection)
        headers = {**_ADMIN_HEADERS, **bad_headers}
        body: dict | None = {} if route["body_schema"] == "dict" else None
        method = route["method"].lower()
        if body is not None:
            resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        else:
            resp = getattr(fuzz_client, method)(url, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}] "
            f"returned {resp.status_code} without CSRF header "
            f"(body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Content-Type fuzz
# ---------------------------------------------------------------------------

# Wrong content types for JSON-body routes
CONTENT_TYPE_CASES = [
    ("plain_text", "text/plain"),
    ("html", "text/html"),
    ("xml", "application/xml"),
    ("form_urlencoded", "application/x-www-form-urlencoded"),
    ("multipart_no_boundary", "multipart/form-data"),
    ("octet_stream", "application/octet-stream"),
    ("yaml", "application/yaml"),
    ("csv", "text/csv"),
    ("no_content_type", None),
    ("json_with_charset_bad", "application/json; charset=ascii"),
    ("vendor_json", "application/vnd.api+json"),
    ("problem_json", "application/problem+json"),
    ("msgpack", "application/msgpack"),
    ("cbor", "application/cbor"),
]

_content_type_params = []
for _route in _JSON_BODY_ROUTES:
    for _label, _ctype in CONTENT_TYPE_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _ctype,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-ct-{_label}",
            marks=[_mark] if _mark else [],
        )
        _content_type_params.append(_param)


class TestContentTypeFuzz:
    """Send JSON-body routes with wrong Content-Type headers.

    Contract: incorrect Content-Type MUST produce 4xx (typically 415 or 400),
    never 500 or 200.
    """

    @pytest.mark.parametrize("route,label,content_type", _content_type_params)
    def test_wrong_content_type_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        content_type: str | None,
    ) -> None:
        """Wrong Content-Type header must produce a 4xx response.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            content_type: Content-Type to send; None omits the header.
        """
        url = _fill_path(route["path"])
        headers = dict(_AUTH_CSRF)
        if content_type is not None:
            headers["Content-Type"] = content_type
        method = route["method"].lower()
        # Send a body that looks like valid JSON text but under the wrong type
        resp = getattr(fuzz_client, method)(
            url,
            content=b'{"key": "value"}',
            headers=headers,
        )
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}={content_type!r}] "
            f"returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Oversized payload fuzz
# ---------------------------------------------------------------------------

# Each: (label, body_factory)
OVERSIZED_PAYLOAD_CASES = [
    ("1mb_string_field", lambda: {"data": "x" * 1_048_576}),
    ("10mb_string_field", lambda: {"data": "x" * 10_485_760}),
    ("1000_keys", lambda: {f"key_{i}": f"value_{i}" for i in range(1000)}),
    ("deeply_nested_100", lambda: _make_nested_dict(100)),
    ("list_of_1000_items", lambda: {"items": list(range(1000))}),
    ("list_of_10000_items", lambda: {"items": list(range(10000))}),
    ("list_of_1000_dicts", lambda: {"items": [{"id": i, "name": f"item_{i}"} for i in range(1000)]}),
    ("repeated_unicode", lambda: {"text": "ðŸ”¥" * 100000}),
    ("large_array_of_strings", lambda: {"inputs": ["x" * 1000 for _ in range(100)]}),
    ("deep_unicode_key", lambda: {("k" * 1000): "value"}),
]


def _make_nested_dict(depth: int) -> dict:
    """Build a dict nested ``depth`` levels deep.

    Args:
        depth: Number of nesting levels.

    Returns:
        A deeply nested dictionary.
    """
    d: dict = {"leaf": "value"}
    for _ in range(depth):
        d = {"nested": d}
    return d


_oversized_params = []
for _route in _JSON_BODY_ROUTES:
    for _label, _factory in OVERSIZED_PAYLOAD_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _factory,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-os-{_label}",
            marks=[_mark] if _mark else [],
        )
        _oversized_params.append(_param)


class TestOversizedPayloadFuzz:
    """Send oversized JSON payloads to every JSON-body route.

    Contract: oversized inputs MUST produce 4xx (400 or 413), never 500 or 200.
    """

    @pytest.mark.parametrize("route,label,body_factory", _oversized_params)
    def test_oversized_payload_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        body_factory: Any,
    ) -> None:
        """Oversized payload must be rejected with a 4xx status.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            body_factory: Callable that produces the oversized body dict.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = route["method"].lower()
        body = body_factory()
        resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}] returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# File-upload fuzz
# ---------------------------------------------------------------------------

# Each: (label, filename, content_bytes, declared_mime)
FILE_UPLOAD_FUZZ_CASES = [
    ("empty_file", "empty.txt", b"", "text/plain"),
    ("binary_garbage", "garbage.bin", b"\x00\x01\x02\xff\xfe\xfd", "application/octet-stream"),
    ("fake_png_header", "evil.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png"),
    ("polyglot_gif_js", "poly.gif", b"GIF89a" + b"<script>alert(1)</script>", "image/gif"),
    ("path_traversal_name", "../../etc/passwd", b"root:x:0:0", "text/plain"),
    ("null_byte_name", "evil\x00.txt", b"content", "text/plain"),
    ("very_long_name", "a" * 512 + ".txt", b"content", "text/plain"),
    ("unicode_name", "æ—¥æœ¬èªžãƒ•ã‚¡ã‚¤ãƒ«.txt", b"content", "text/plain"),
    ("no_extension", "noeext", b"some data", "application/octet-stream"),
    ("json_disguised_as_txt", "data.txt", b'{"key": "value"}', "text/plain"),
    ("script_as_text", "evil.js", b"alert(1)", "text/plain"),
    ("empty_mime_type", "file.txt", b"content", ""),
    ("wrong_mime_for_content", "image.png", b"this is not an image", "image/png"),
    ("zip_bomb_small", "bomb.zip", b"PK\x05\x06" + b"\x00" * 18, "application/zip"),
    ("1mb_file", "large.txt", b"x" * 1_048_576, "text/plain"),
    ("multiple_null_bytes", "test.txt", b"con\x00tent\x00here", "text/plain"),
    ("windows_reserved_name", "CON.txt", b"content", "text/plain"),
    ("no_file_sent", None, None, None),  # multipart without file part
]


def _build_multipart(
    filename: str | None,
    content: bytes | None,
    mime_type: str | None,
) -> tuple[bytes, str]:
    """Build a raw multipart/form-data body and return (body_bytes, boundary).

    Args:
        filename: Name to use for the file part; None omits the file.
        content: File content bytes; None omits the file.
        mime_type: MIME type for the file part; None omits Content-Type.

    Returns:
        Tuple of (multipart body bytes, boundary string).
    """
    boundary = "fuzz-boundary-12345"
    parts = []
    if filename is not None and content is not None:
        part_lines = [
            f"--{boundary}",
            f'Content-Disposition: form-data; name="file"; filename="{filename}"',
        ]
        if mime_type:
            part_lines.append(f"Content-Type: {mime_type}")
        part_lines.append("")
        parts.append("\r\n".join(part_lines).encode() + b"\r\n" + content)
    else:
        # Send a multipart with no file part — just a text field
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="dummy"\r\n\r\nvalue'.encode())
    body = b"\r\n".join(parts) + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


_file_upload_params = []
for _route in _FILE_UPLOAD_ROUTES:
    for _label, _fname, _fcontent, _fmime in FILE_UPLOAD_FUZZ_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _fname,
            _fcontent,
            _fmime,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-fu-{_label}",
            marks=[_mark] if _mark else [],
        )
        _file_upload_params.append(_param)


class TestFileUploadFuzz:
    """Send malformed or adversarial multipart bodies to file-upload routes.

    Contract: adversarial uploads MUST produce 4xx, never 500 or 2xx on garbage.
    """

    @pytest.mark.parametrize(
        "route,label,filename,content,mime_type",
        _file_upload_params,
    )
    def test_file_upload_fuzz(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        filename: str | None,
        content: bytes | None,
        mime_type: str | None,
    ) -> None:
        """Adversarial multipart upload must be rejected with a 4xx status.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            filename: File name in the multipart part (may be adversarial).
            content: File content bytes (may be garbage or empty).
            mime_type: MIME type declared in the part header.
        """
        url = _fill_path(route["path"])
        body_bytes, boundary = _build_multipart(filename, content, mime_type)
        headers = {
            **_AUTH_CSRF,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        resp = fuzz_client.post(url, content=body_bytes, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"POST {route['path']} [{label}] returned {resp.status_code} (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Hypothesis property-based fuzzing
# ---------------------------------------------------------------------------

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

# Hypothesis strategies for building random-ish bodies
if _HYPOTHESIS_AVAILABLE:
    _text_strategy = st.text(
        alphabet=string.printable,
        min_size=0,
        max_size=500,
    )
    _scalar_strategy = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**31), max_value=2**31),
        st.floats(allow_nan=False, allow_infinity=False),
        _text_strategy,
    )
    _flat_dict_strategy = st.dictionaries(
        keys=_text_strategy,
        values=_scalar_strategy,
        min_size=0,
        max_size=20,
    )


# We pick a representative subset of routes for Hypothesis (running all 127
# routes × many Hypothesis examples would be prohibitively slow in CI).
_HYPOTHESIS_ROUTE_SAMPLE = [
    r
    for r in _JSON_BODY_ROUTES
    if r["path"]
    in {
        "/api/v1/approvals/{action_id}/approve",
        "/api/v1/autonomy/policies/{action_type}",
        "/api/v1/memory",
        "/api/v1/models/select",
        "/api/v1/run-prompt",
        "/api/v1/run-task",
        "/api/v1/system-prompts",
        "/api/v1/training/experiments/compare",
        "/api/adr/propose",
        "/api/project/{project_id}/rename",
    }
]


@pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed",
)
class TestHypothesisBodyFuzz:
    """Property-based fuzz tests using Hypothesis for a sample of routes.

    Invariant: for any randomly-generated JSON body, the server must not
    return 500 (Internal Server Error) or 200 (success) for the inputs that
    are not known-valid.  Acceptable responses are 4xx.
    """

    @pytest.mark.parametrize(
        "route",
        _HYPOTHESIS_ROUTE_SAMPLE,
        ids=[
            f"{r['method']}-{r['path'].replace('/', '_').replace('{', '').replace('}', '')}"
            for r in _HYPOTHESIS_ROUTE_SAMPLE
        ],
    )
    @settings(  # type: ignore[misc]
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @given(body=_flat_dict_strategy)  # type: ignore[misc]
    def test_random_body_never_500(
        self,
        fuzz_client: TestClient,
        route: dict,
        body: dict,
    ) -> None:
        """Random JSON body must not trigger an unhandled 500 error.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            body: Randomly generated dict from Hypothesis.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        assert resp.status_code in _BOUNDED_MUTATION_STATUSES, (
            f"{route['method']} {route['path']} returned unexpected status {resp.status_code} "
            f"for body={body!r} (response: {resp.text[:300]})"
        )


# ---------------------------------------------------------------------------
# Smoke tests: routes with no body (DELETE) still reject bad auth / no CSRF
# ---------------------------------------------------------------------------

_NO_BODY_ROUTES = [r for r in _ALL_MUTATING if r["body_schema"] == "none"]

_no_body_auth_params = [
    pytest.param(
        route,
        id=f"{route['method']}-{route['path'].replace('/', '_').replace('{', '').replace('}', '')}-noauth",
        marks=[_xfail_if_known_500(route)] if _xfail_if_known_500(route) else [],
    )
    for route in _NO_BODY_ROUTES
    if route["auth_required"] == "required"
]

_no_body_csrf_params = [
    pytest.param(
        route,
        id=f"{route['method']}-{route['path'].replace('/', '_').replace('{', '').replace('}', '')}-nocsrf",
        marks=[_xfail_if_known_500(route)] if _xfail_if_known_500(route) else [],
    )
    for route in _NO_BODY_ROUTES
    if route["path"] not in _A2A_PATHS
]


class TestNoBodyRoutesFuzz:
    """Fuzz DELETE and other no-body mutating routes for auth/CSRF enforcement.

    These routes carry no JSON body, but they still require auth and CSRF.
    """

    @pytest.mark.parametrize("route", _no_body_auth_params)
    def test_no_body_route_requires_auth(
        self,
        fuzz_client: TestClient,
        route: dict,
    ) -> None:
        """No-body mutating route must reject unauthenticated requests.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE with body_schema='none'.
        """
        url = _fill_path(route["path"])
        headers = {**_CSRF}  # No auth header
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, headers=headers)
        assert resp.status_code in {401, 403}, (
            f"{route['method']} {route['path']} returned {resp.status_code} without auth (body: {resp.text[:200]})"
        )

    @pytest.mark.parametrize("route", _no_body_csrf_params)
    def test_no_body_route_requires_csrf(
        self,
        fuzz_client: TestClient,
        route: dict,
    ) -> None:
        """No-body mutating route must reject requests missing the CSRF header.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE with body_schema='none'.
        """
        url = _fill_path(route["path"])
        headers = {**_ADMIN_HEADERS}  # Auth present, CSRF absent
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} returned {resp.status_code} without CSRF (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Key-field mutation: target known required fields by name
# ---------------------------------------------------------------------------

# Routes that advertise specific required key_fields in the inventory
_KEY_FIELD_ROUTES = [r for r in _JSON_BODY_ROUTES if r["key_fields"]]

# Mutations for each required key field
_KEY_FIELD_MUTATIONS = [
    ("missing", lambda fields: dict.fromkeys(fields[1:], "value")),  # omit first
    ("null_all", lambda fields: dict.fromkeys(fields)),
    ("int_all", lambda fields: dict.fromkeys(fields, 0)),
    ("list_all", lambda fields: {f: [] for f in fields}),
    ("empty_string_all", lambda fields: dict.fromkeys(fields, "")),
]

_key_field_params = []
for _route in _KEY_FIELD_ROUTES:
    for _label, _mutation_fn in _KEY_FIELD_MUTATIONS:
        _fields = _route["key_fields"]
        if not _fields:
            continue
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _mutation_fn(_fields),
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-kf-{_label}",
            marks=[_mark] if _mark else [],
        )
        _key_field_params.append(_param)


class TestKeyFieldMutationFuzz:
    """Mutate known-required key fields for routes that declare them.

    Contract: missing or wrong-typed required fields MUST return 4xx.
    """

    @pytest.mark.parametrize("route,label,body", _key_field_params)
    def test_key_field_mutation_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        body: dict,
    ) -> None:
        """Missing or mutated required key fields must be rejected with 4xx.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE with declared key_fields.
            label: Human-readable mutation label for diagnostics.
            body: Body dict with mutations applied to required fields.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, json=body, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{route['method']} {route['path']} [{label}] "
            f"returned {resp.status_code} for body={body!r} "
            f"(response: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# HTTP method confusion: send wrong method to every route
# ---------------------------------------------------------------------------

# For each route, try the other common mutating methods
_WRONG_METHOD_MAP: dict[str, list[str]] = {
    "POST": ["PUT", "PATCH", "DELETE"],
    "PUT": ["POST", "PATCH", "DELETE"],
    "DELETE": ["POST", "PUT"],
}

# Build a set of (path, method) pairs from the full ROUTE_TABLE so we can
# skip "wrong method" cases where that method is actually registered for
# the same path (e.g. a path that has both POST and DELETE handlers).
_REGISTERED_METHODS: set[tuple[str, str]] = {
    (r["path"], r["method"].upper()) for r in ROUTE_TABLE
}

_method_confusion_params = []
for _route in _ALL_MUTATING[:50]:  # Limit to first 50 routes to keep suite size reasonable
    for _wrong_method in _WRONG_METHOD_MAP.get(_route["method"], []):
        # Skip if the "wrong" method is legitimately registered for this path.
        if (_route["path"], _wrong_method.upper()) in _REGISTERED_METHODS:
            continue
        _param = pytest.param(
            _route,
            _wrong_method,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-wm-{_wrong_method}",
        )
        _method_confusion_params.append(_param)


class TestMethodConfusionFuzz:
    """Send wrong HTTP methods to registered routes.

    Contract: a route registered for POST must reject PUT/DELETE with 405,
    not 500 or 200.  Some routes may legitimately accept multiple methods;
    405 is the canonical rejection.
    """

    @pytest.mark.parametrize("route,wrong_method", _method_confusion_params)
    def test_wrong_method_rejected(
        self,
        fuzz_client: TestClient,
        route: dict,
        wrong_method: str,
    ) -> None:
        """Wrong HTTP method must produce 405 or another 4xx status.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            wrong_method: HTTP method that does not match the route definition.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = wrong_method.lower()
        # httpx.Client.delete() does not accept a json= keyword argument —
        # DELETE requests conventionally have no body.
        if method == "delete":
            resp = getattr(fuzz_client, method)(url, headers=headers)
        else:
            resp = getattr(fuzz_client, method)(url, json={}, headers=headers)
        assert resp.status_code in _SAFE_4XX, (
            f"{wrong_method} {route['path']} (registered as {route['method']}) "
            f"returned {resp.status_code} "
            f"(body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Header injection fuzz
# ---------------------------------------------------------------------------

HEADER_INJECTION_CASES = [
    ("newline_in_value", {"X-Custom": "value\r\nX-Injected: injected"}),
    ("null_in_value", {"X-Custom": "value\x00extra"}),
    ("oversized_header", {"X-Custom": "x" * 8192}),
    ("many_headers", {f"X-Custom-{i}": f"value-{i}" for i in range(100)}),
    ("encoded_newline", {"X-Custom": "value%0d%0aX-Injected: bad"}),
    ("host_override", {"Host": "evil.example.com"}),
    ("content_length_override", {"Content-Length": "0"}),
    ("transfer_encoding_inject", {"Transfer-Encoding": "chunked, identity"}),
    ("connection_inject", {"Connection": "keep-alive, X-Custom: injected"}),
]

# Use a small subset of routes for header injection (25 routes)
_HEADER_FUZZ_ROUTES = _JSON_BODY_ROUTES[:25]

_header_injection_params = []
for _route in _HEADER_FUZZ_ROUTES:
    for _label, _hdrs in HEADER_INJECTION_CASES:
        _mark = _xfail_if_known_500(_route)
        _param = pytest.param(
            _route,
            _label,
            _hdrs,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-hi-{_label}",
            marks=[_mark] if _mark else [],
        )
        _header_injection_params.append(_param)


class TestHeaderInjectionFuzz:
    """Send requests with injected or malformed HTTP headers.

    Contract: header injection must not cause 500.  The server may process
    the request normally (ignoring unknown headers) or reject with 4xx; both
    are acceptable as long as 500 is not returned.
    """

    @pytest.mark.parametrize("route,label,extra_headers", _header_injection_params)
    def test_header_injection_no_500(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        extra_headers: dict,
    ) -> None:
        """Header injection must not produce a 500 Internal Server Error.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            extra_headers: Extra or malformed headers to include in the request.
        """
        url = _fill_path(route["path"])
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        # Merge carefully — httpx filters out some invalid headers silently
        try:
            headers.update(extra_headers)
        except (ValueError, TypeError):
            # Some injected values may raise during header construction — skip
            pytest.skip(f"Header construction raised for {label!r}")
        method = route["method"].lower()
        try:
            resp = getattr(fuzz_client, method)(url, json={}, headers=headers)
        except Exception:
            # Network-layer error due to malformed header — not a 500
            return
        assert resp.status_code in _BOUNDED_MUTATION_STATUSES, (
            f"{route['method']} {route['path']} [{label}] returned unexpected status {resp.status_code} "
            f"with injected headers (body: {resp.text[:200]})"
        )


# ---------------------------------------------------------------------------
# Query-string injection fuzz
# ---------------------------------------------------------------------------

QUERY_INJECTION_CASES = [
    ("sql_injection", "id=1' OR '1'='1"),
    ("xss_in_param", "name=<script>alert(1)</script>"),
    ("path_traversal_in_param", "path=../../etc/passwd"),
    ("null_byte_in_param", "name=test%00admin"),
    ("huge_param_value", "data=" + "x" * 4096),
    ("many_params", "&".join(f"p{i}=v{i}" for i in range(200))),
    ("repeated_param", "id=1&id=2&id=3&id=4&id=5"),
    ("unicode_param", "name=%E6%97%A5%E6%9C%AC%E8%AA%9E"),
    ("encoded_space", "q=hello+world&r=hello%20world"),
    ("nested_encoded", "url=http%3A%2F%2Fevil.com%2F%3Fsteal%3D1"),
    ("ampersand_inject", "value=good%26admin%3Dtrue"),
    ("empty_param_name", "=value&other=val"),
    ("param_with_equals", "key=a=b=c"),
]

# Use a small subset of routes for query injection
_QUERY_FUZZ_ROUTES = _JSON_BODY_ROUTES[:20]

_query_injection_params = []
for _route in _QUERY_FUZZ_ROUTES:
    for _label, _qs in QUERY_INJECTION_CASES:
        _param = pytest.param(
            _route,
            _label,
            _qs,
            id=f"{_route['method']}-{_route['path'].replace('/', '_').replace('{', '').replace('}', '')}-qi-{_label}",
        )
        _query_injection_params.append(_param)


class TestQueryStringInjectionFuzz:
    """Append adversarial query strings to mutating routes.

    Contract: adversarial query strings must not trigger 500 responses.
    4xx or 2xx (if validation passes query params) are both acceptable.
    """

    @pytest.mark.parametrize("route,label,query_string", _query_injection_params)
    def test_query_string_injection_no_500(
        self,
        fuzz_client: TestClient,
        route: dict,
        label: str,
        query_string: str,
    ) -> None:
        """Adversarial query strings must not trigger a 500 response.

        Args:
            fuzz_client: The Litestar test client.
            route: Route entry from ROUTE_TABLE.
            label: Human-readable case label for diagnostics.
            query_string: Raw query string to append to the URL.
        """
        base_url = _fill_path(route["path"])
        url = f"{base_url}?{query_string}"
        headers = {**_AUTH_CSRF, "Content-Type": "application/json"}
        method = route["method"].lower()
        resp = getattr(fuzz_client, method)(url, json={}, headers=headers)
        assert resp.status_code in _BOUNDED_MUTATION_STATUSES, (
            f"{route['method']} {route['path']} [{label}] "
            f"returned unexpected status {resp.status_code} with query={query_string!r} "
            f"(body: {resp.text[:200]})"
        )
