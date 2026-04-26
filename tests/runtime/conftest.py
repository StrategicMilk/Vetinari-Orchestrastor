"""Shared fixtures for the tests/runtime/ config-matrix test suite.

Provides the noop lifespan helper, session-scoped Litestar app, per-test
TestClient, and common auth constants used by all test_config_matrix_*.py
files in this directory.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
from litestar.testing import TestClient

# -- Shared auth constants used by HTTP tests that require admin access ----------

# Admin token accepted by the admin_guard security guard in tests.
_ADMIN_TOKEN = "test-matrix-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# CSRF header required by all Litestar mutation endpoints.
_CSRF = {"X-Requested-With": "XMLHttpRequest"}


# -- Lifespan helper and app/client fixtures ------------------------------------


@asynccontextmanager
async def _noop_lifespan(app: Any):
    """Drop-in lifespan that skips all subsystem wiring during tests."""
    yield


@pytest.fixture(scope="session")
def matrix_app() -> Any:
    """Create a single Litestar app instance shared across the config matrix session.

    The patches disable real startup/shutdown I/O so tests stay fast and isolated.
    Both patches are applied only during create_app(); the app then runs without them.

    Returns:
        Litestar application with noop lifespan and no shutdown handlers.
    """
    with (
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=False)


@pytest.fixture
def client(matrix_app: Any) -> Generator[TestClient, None, None]:
    """Yield a fresh TestClient per test.

    Args:
        matrix_app: The session-scoped Litestar application.

    Yields:
        A connected TestClient instance.
    """
    with TestClient(app=matrix_app) as c:
        yield c
