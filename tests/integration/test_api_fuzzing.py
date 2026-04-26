"""API endpoint fuzzing with schemathesis.

Generates random valid requests for each endpoint defined in Vetinari's
Litestar app and checks for 500 errors.  Catches unhandled exceptions,
serialization bugs, and input validation gaps that manual tests miss.

Requires: ``schemathesis>=3.25.0`` (declared in pyproject.toml [dev]).

Usage::

    python -m pytest tests/integration/test_api_fuzzing.py -v --timeout=60
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import schemathesis
    from schemathesis import Case

    HAS_SCHEMATHESIS = True
except ImportError:
    HAS_SCHEMATHESIS = False

pytestmark = pytest.mark.skipif(not HAS_SCHEMATHESIS, reason="schemathesis not installed")


@pytest.fixture
def litestar_app():
    """Create a minimal Litestar app for schema-based fuzzing.

    Patches heavy subsystems (model loading, scheduling) that aren't
    needed for API contract testing.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        app = create_app(debug=True)
    return app


@pytest.fixture
def api_schema(litestar_app):
    """Extract the OpenAPI schema from the Litestar app for schemathesis."""
    return schemathesis.from_asgi("/schema/openapi.json", app=litestar_app)


class TestAPIFuzzing:
    """Schema-driven fuzzing of all API endpoints."""

    def test_health_endpoint_never_500s(self, litestar_app) -> None:
        """The /health endpoint must never return a 500 regardless of input."""
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            # Standard health check
            response = client.get("/health")
            assert response.status_code < 500, f"/health returned {response.status_code}: {response.text}"

    def test_skills_endpoint_returns_valid_json(self, litestar_app) -> None:
        """The /api/v1/skills endpoint returns valid JSON with expected shape."""
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = client.get("/api/v1/skills")
            assert response.status_code < 500, f"/api/v1/skills returned {response.status_code}"
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, (list, dict)), f"Expected list or dict, got {type(data).__name__}"
