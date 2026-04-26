"""Mounted governance tests for audit trail read routes.

Verifies that every audit route returns 503 (not a raw unhandled 500) when
its backing subsystem raises an exception. All requests go through the real
Litestar TestClient stack so router registration, middleware, guards, and
serialisation are all exercised.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

# ---------------------------------------------------------------------------
# Admin auth constants
# ---------------------------------------------------------------------------

_ADMIN_TOKEN = "test-audit-governance-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app: Any):
    """Drop-in lifespan that skips all subsystem wiring."""
    yield


@pytest.fixture(scope="module")
def app():
    """Create a Litestar app with startup/shutdown wiring suppressed.

    Module scope avoids rebuilding the ~300-handler app per test.
    """
    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=True)


@pytest.fixture
def client(app):
    """Yield a fresh TestClient per test."""
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app=app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Audit manifests
# ---------------------------------------------------------------------------


class TestAuditManifestsGovernance:
    """GET /api/v1/audit/manifests — must return 503 when memory store fails."""

    def test_manifests_returns_503_when_store_raises(self, client: TestClient) -> None:
        """Memory store error must produce 503, not an unhandled 500."""
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("memory store down"),
        ):
            resp = client.get("/api/v1/audit/manifests", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_manifests_returns_200_with_list_on_success(self, client: TestClient) -> None:
        """Successful call must return 200 with a ``manifests`` list."""
        mock_store = MagicMock()
        mock_store.recall_episodes.return_value = []
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            resp = client.get("/api/v1/audit/manifests", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "manifests" in body
        assert isinstance(body["manifests"], list)


# ---------------------------------------------------------------------------
# Audit decisions
# ---------------------------------------------------------------------------


class TestAuditDecisionsGovernance:
    """GET /api/v1/audit/decisions — must return 503 when audit logger fails."""

    def test_decisions_returns_503_when_logger_raises(self, client: TestClient) -> None:
        """Audit logger error must produce 503, not an unhandled 500."""
        with patch(
            "vetinari.audit.get_audit_logger",
            side_effect=RuntimeError("audit logger down"),
        ):
            resp = client.get("/api/v1/audit/decisions", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_decisions_returns_200_with_list_on_success(self, client: TestClient) -> None:
        """Successful call must return 200 with a ``decisions`` list."""
        mock_audit = MagicMock()
        mock_audit.read_decisions.return_value = []
        with patch(
            "vetinari.audit.get_audit_logger",
            return_value=mock_audit,
        ):
            resp = client.get("/api/v1/audit/decisions", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "decisions" in body
        assert isinstance(body["decisions"], list)


# ---------------------------------------------------------------------------
# Audit task trail
# ---------------------------------------------------------------------------


class TestAuditTaskGovernance:
    """GET /api/v1/audit/tasks/{task_id} — must return 503 when store fails."""

    def test_task_audit_returns_503_when_store_raises(self, client: TestClient) -> None:
        """Memory store error must produce 503, not an unhandled 500."""
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("memory store down"),
        ):
            resp = client.get("/api/v1/audit/tasks/task-abc", headers=_ADMIN_HEADERS)
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error"

    def test_task_audit_returns_200_with_trail_on_success(self, client: TestClient) -> None:
        """Successful call must return 200 with ``task_id``, ``memories``, and ``episodes``."""
        mock_mem = MagicMock()
        mock_mem.to_dict.return_value = {"memory_id": "m1", "content": "test"}

        mock_ep = MagicMock()
        mock_ep.to_dict.return_value = {"episode_id": "e1"}

        mock_store = MagicMock()
        mock_store.search.return_value = [mock_mem]
        mock_store.recall_episodes.return_value = [mock_ep]

        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            resp = client.get("/api/v1/audit/tasks/task-xyz", headers=_ADMIN_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-xyz"
        assert isinstance(body["memories"], list)
        assert isinstance(body["episodes"], list)
