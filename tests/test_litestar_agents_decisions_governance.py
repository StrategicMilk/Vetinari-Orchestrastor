"""Mounted request-level governance tests for agents and decisions routes.

Proves every route returns bounded 503 (not raw 500) when the backing
subsystem raises, and validates input contracts.

Agent routes:
  GET  /api/v1/agents/status
  POST /api/v1/agents/initialize
  GET  /api/v1/agents/active
  GET  /api/v1/agents/tasks
  GET  /api/v1/agents/memory

Decision routes:
  GET  /api/v1/decisions/pending
  GET  /api/v1/decisions/history
  POST /api/v1/decisions
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.

_ADMIN_TOKEN = "test-admin-governance"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}
_MUTATION_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN, "X-Requested-With": "XMLHttpRequest"}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is built only once; each
    test creates its own TestClient context so connection state does not leak.

    Returns:
        A Litestar application instance.
    """
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Args:
        app: The module-scoped Litestar application fixture.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
        with TestClient(app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert *response* is a bounded 503 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.

    Raises:
        AssertionError: When status_code is not 503 or the body lacks the
            ``"status": "error"`` field.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


def _assert_error_with_code(response: object, expected_code: int) -> None:
    """Assert *response* has *expected_code* and an ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
        expected_code: The expected HTTP status code.

    Raises:
        AssertionError: When status_code does not match or body lacks envelope.
    """
    assert response.status_code == expected_code, (
        f"Expected {expected_code}, got {response.status_code}. Body: {response.text[:400]}"
    )
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


# ===========================================================================
# Agents API tests
# ===========================================================================


class TestAgentsStatusRoute:
    """Tests for GET /api/v1/agents/status on subsystem failure and normal path."""

    def test_orchestrator_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/agents/status returns 503 when MultiAgentOrchestrator raises.

        The handler does a local import each call; patching the class on its
        home module intercepts that import and causes get_instance to raise.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.side_effect = RuntimeError("orchestrator down")
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/status")
        _assert_503_error(response)

    def test_no_orchestrator_returns_empty_agents(self, client: object) -> None:
        """GET /api/v1/agents/status returns 200 with empty list when instance is None.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.return_value = None
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/status")
        assert response.status_code == 200
        body = response.json()
        assert "agents" in body
        assert body["agents"] == []


class TestAgentsInitializeRoute:
    """Tests for POST /api/v1/agents/initialize on subsystem failure and normal path."""

    def test_initialize_raises_returns_503(self, client: object) -> None:
        """POST /api/v1/agents/initialize returns 503 when orchestrator raises.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.side_effect = RuntimeError("init failed")
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.post("/api/v1/agents/initialize", headers=_MUTATION_HEADERS)
        _assert_503_error(response)

    def test_initialize_success_returns_agent_names(self, client: object) -> None:
        """POST /api/v1/agents/initialize returns 201 with agent names when successful.

        Litestar defaults POST handlers to 201 Created, so we assert 201 here,
        not 200.

        Args:
            client: The TestClient fixture.
        """
        mock_agent_a = MagicMock()
        mock_agent_a.name = "foreman-1"
        mock_agent_b = MagicMock()
        mock_agent_b.name = "worker-1"

        mock_orch = MagicMock()
        mock_orch.agents = {"a": mock_agent_a, "b": mock_agent_b}
        mock_cls = MagicMock()
        mock_cls.get_instance.return_value = mock_orch

        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.post("/api/v1/agents/initialize", headers=_MUTATION_HEADERS)
        assert response.status_code == 201
        body = response.json()
        assert body.get("status") == "initialized"
        assert "foreman-1" in body["agents"]
        assert "worker-1" in body["agents"]


class TestAgentsActiveRoute:
    """Tests for GET /api/v1/agents/active on subsystem failure and normal path."""

    def test_orchestrator_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/agents/active returns 503 when orchestrator raises.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.side_effect = RuntimeError("active agents unavailable")
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/active")
        _assert_503_error(response)

    def test_no_orchestrator_returns_empty_agents(self, client: object) -> None:
        """GET /api/v1/agents/active returns 200 with empty list when instance is None.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.return_value = None
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/active")
        assert response.status_code == 200
        body = response.json()
        assert body.get("agents") == []


class TestAgentsTasksRoute:
    """Tests for GET /api/v1/agents/tasks on subsystem failure and normal path."""

    def test_orchestrator_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/agents/tasks returns 503 when orchestrator raises.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.side_effect = RuntimeError("task queue unavailable")
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/tasks")
        _assert_503_error(response)

    def test_no_orchestrator_returns_empty_tasks(self, client: object) -> None:
        """GET /api/v1/agents/tasks returns 200 with empty list when instance is None.

        Args:
            client: The TestClient fixture.
        """
        mock_cls = MagicMock()
        mock_cls.get_instance.return_value = None
        with patch(
            "vetinari.agents.multi_agent_orchestrator.MultiAgentOrchestrator",
            mock_cls,
        ):
            response = client.get("/api/v1/agents/tasks")
        assert response.status_code == 200
        body = response.json()
        assert body.get("tasks") == []


class TestAgentsMemoryRoute:
    """Tests for GET /api/v1/agents/memory on subsystem failure."""

    def test_memory_store_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/agents/memory returns 503 when get_unified_memory_store raises.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("store down"),
        ):
            response = client.get("/api/v1/agents/memory")
        _assert_503_error(response)

    def test_timeline_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/agents/memory returns 503 when store.timeline raises.

        Args:
            client: The TestClient fixture.
        """
        mock_store = MagicMock()
        mock_store.timeline.side_effect = RuntimeError("timeline unavailable")
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            response = client.get("/api/v1/agents/memory")
        _assert_503_error(response)


# ===========================================================================
# Decisions API tests
# ===========================================================================


class TestDecisionsPendingRoute:
    """Tests for GET /api/v1/decisions/pending on subsystem failure."""

    def test_memory_store_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/decisions/pending returns 503 when get_unified_memory_store raises.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("store down"),
        ):
            response = client.get("/api/v1/decisions/pending")
        _assert_503_error(response)

    def test_search_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/decisions/pending returns 503 when store.search raises.

        Args:
            client: The TestClient fixture.
        """
        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("search unavailable")
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            response = client.get("/api/v1/decisions/pending")
        _assert_503_error(response)


class TestDecisionsHistoryRoute:
    """Tests for GET /api/v1/decisions/history on subsystem failure."""

    def test_approval_queue_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/decisions/history returns 503 when get_approval_queue raises.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.autonomy.approval_queue.get_approval_queue",
            side_effect=RuntimeError("approval queue down"),
        ):
            response = client.get("/api/v1/decisions/history", headers=_ADMIN_HEADERS)
        _assert_503_error(response)

    def test_decision_journal_raises_returns_503(self, client: object) -> None:
        """GET /api/v1/decisions/history returns 503 when get_decision_journal raises.

        The approval queue succeeds (returns empty log) but the pipeline journal
        raises, so the handler must still produce a 503 rather than a partial result.

        Args:
            client: The TestClient fixture.
        """
        mock_aq = MagicMock()
        mock_aq.get_decision_log.return_value = []
        with (
            patch(
                "vetinari.autonomy.approval_queue.get_approval_queue",
                return_value=mock_aq,
            ),
            patch(
                "vetinari.observability.decision_journal.get_decision_journal",
                side_effect=RuntimeError("journal down"),
            ),
        ):
            response = client.get("/api/v1/decisions/history", headers=_ADMIN_HEADERS)
        _assert_503_error(response)


class TestDecisionsSubmitRoute:
    """Tests for POST /api/v1/decisions  -  input validation and subsystem failure."""

    def test_empty_body_returns_400(self, client: object) -> None:
        """POST /api/v1/decisions with empty body returns 400 missing decision_id.

        An empty JSON object must be rejected before touching the memory store.

        Args:
            client: The TestClient fixture.
        """
        response = client.post("/api/v1/decisions", json={}, headers=_MUTATION_HEADERS)
        _assert_error_with_code(response, 400)
        body = response.json()
        assert "decision_id" in body.get("error", ""), (
            f"Expected error mentioning 'decision_id', got: {body.get('error')!r}"
        )

    def test_missing_decision_returns_404(self, client: object) -> None:
        """POST /api/v1/decisions returns 404 when decision_id does not exist in store.

        Args:
            client: The TestClient fixture.
        """
        mock_store = MagicMock()
        mock_store.search.return_value = []  # No matching decision
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            response = client.post(
                "/api/v1/decisions",
                json={"decision_id": "nonexistent-decision-id", "choice": "option_a"},
                headers=_MUTATION_HEADERS,
            )
        _assert_error_with_code(response, 404)

    def test_store_raises_returns_503(self, client: object) -> None:
        """POST /api/v1/decisions returns 503 when get_unified_memory_store raises.

        Args:
            client: The TestClient fixture.
        """
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("store down"),
        ):
            response = client.post(
                "/api/v1/decisions",
                json={"decision_id": "some-decision-id", "choice": "option_a"},
                headers=_MUTATION_HEADERS,
            )
        _assert_503_error(response)

    def test_success_returns_resolved_status(self, client: object) -> None:
        """POST /api/v1/decisions returns 201 with resolved status when decision is found.

        Litestar defaults POST handlers to 201 Created when the handler returns
        a plain dict (not a Response object).

        Args:
            client: The TestClient fixture.
        """
        mock_entry = MagicMock()
        mock_entry.id = "decision-abc-123"
        mock_entry.content = "Should we proceed?"

        mock_store = MagicMock()
        mock_store.search.return_value = [mock_entry]

        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            response = client.post(
                "/api/v1/decisions",
                json={"decision_id": "decision-abc-123", "choice": "yes"},
                headers=_MUTATION_HEADERS,
            )
        assert response.status_code == 201
        body = response.json()
        assert body.get("status") == "resolved"
        assert body.get("choice") == "yes"
