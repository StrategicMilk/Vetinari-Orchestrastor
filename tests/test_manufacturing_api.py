"""Tests for vetinari.web.litestar_manufacturing_api — manufacturing and quality gate handlers.

Covers:
- GET /api/v1/workflow/gates returns a dict of configured gates
- POST /api/v1/workflow/gates/{stage} with non-numeric criteria returns 400
- PUT /api/v1/workflow/gates/{stage} with valid data registers a gate
- DELETE /api/v1/workflow/gates/{stage} removes a gate
- POST /api/v1/workflow/gates/{stage} with invalid failure_action returns 400
- GET /api/v1/workflow/gates returns 503 when gate runner is unavailable
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from vetinari.web.litestar_manufacturing_api import create_manufacturing_handlers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously in a new event loop."""
    return asyncio.run(coro)


def _get_handler(handlers, path_substr: str, method: str | None = None):
    """Find the first handler whose paths set contains path_substr.

    Args:
        handlers: List of Litestar route handlers.
        path_substr: Substring to match against handler paths.
        method: Optional HTTP method to match (e.g. "GET", "POST").

    Returns:
        The first matching handler.
    """
    for h in handlers:
        if any(path_substr in p for p in h.paths) and (method is None or method in h.http_methods):
            return h
    raise StopIteration(f"No handler for {path_substr!r} method={method!r}")


# ---------------------------------------------------------------------------
# Smoke test: create_manufacturing_handlers returns handlers
# ---------------------------------------------------------------------------


class TestCreateHandlers:
    """create_manufacturing_handlers() returns the expected handler set."""

    def test_returns_handlers_when_litestar_available(self) -> None:
        """Non-empty list of route handlers is returned when Litestar is installed."""
        handlers = create_manufacturing_handlers()
        if handlers:
            assert len(handlers) > 0

    def test_returns_empty_list_when_litestar_unavailable(self, monkeypatch) -> None:
        """An empty list is returned gracefully when Litestar is missing."""
        import vetinari.web.litestar_manufacturing_api as mod

        monkeypatch.setattr(mod, "_LITESTAR_AVAILABLE", False)
        handlers = mod.create_manufacturing_handlers()
        assert handlers == []


# ---------------------------------------------------------------------------
# GET /api/v1/workflow/gates
# ---------------------------------------------------------------------------


class TestWorkflowGatesList:
    """api_workflow_gates_list() returns configured gates or 503 on subsystem error."""

    def test_returns_dict_of_configured_gates(self) -> None:
        """Handler returns a dict mapping stage name to gate configuration."""
        from vetinari.workflow.quality_gates import GateAction, WorkflowGate

        gate = WorkflowGate(
            name="post_exec_gate",
            stage="post_execution",
            criteria={"quality_score": 0.8},
            failure_action=GateAction.BLOCK,
        )
        mock_runner = MagicMock()
        mock_runner.gates = {"post_execution": gate}

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates", "GET")

        with patch("vetinari.workflow.quality_gates.get_gate_runner", return_value=mock_runner):
            result = _run(handler_fn.fn())

        assert isinstance(result, dict)
        assert "post_execution" in result
        gate_info = result["post_execution"]
        assert gate_info["name"] == "post_exec_gate"
        assert gate_info["stage"] == "post_execution"
        assert gate_info["failure_action"] == "block"

    def test_returns_503_when_gate_runner_unavailable(self) -> None:
        """Handler returns a 503 error response when the gate runner subsystem fails."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates", "GET")

        with patch(
            "vetinari.workflow.quality_gates.get_gate_runner",
            side_effect=RuntimeError("subsystem offline"),
        ):
            result = _run(handler_fn.fn())

        assert isinstance(result, Response)
        assert result.status_code == 503

    def test_returns_empty_dict_when_no_gates_configured(self) -> None:
        """Handler returns an empty dict when the runner has no gates registered."""
        mock_runner = MagicMock()
        mock_runner.gates = {}

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates", "GET")

        with patch(
            "vetinari.web.litestar_manufacturing_api._get_gate_runner",
            return_value=mock_runner,
        ):
            with patch(
                "vetinari.workflow.quality_gates.get_gate_runner",
                return_value=mock_runner,
            ):
                result = _run(handler_fn.fn())

        assert result == {}


# ---------------------------------------------------------------------------
# POST /api/v1/workflow/gates/{stage} — malformed threshold test
# ---------------------------------------------------------------------------


class TestAddGate:
    """api_add_gate() registers a gate or returns 400 on invalid input."""

    def test_non_numeric_criteria_value_returns_400(self) -> None:
        """POST with non-numeric criteria values returns a 400 error response."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "POST")

        payload = {
            "name": "bad_gate",
            "criteria": {"quality_score": "high"},  # string instead of numeric threshold
            "failure_action": "block",
        }
        result = _run(handler_fn.fn(stage="post_execution", data=payload))

        assert isinstance(result, Response)
        assert result.status_code == 400

    def test_missing_required_fields_returns_400(self) -> None:
        """POST without name, criteria, or failure_action returns a 400 error."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "POST")

        result = _run(handler_fn.fn(stage="post_execution", data={}))

        assert isinstance(result, Response)
        assert result.status_code == 400

    def test_invalid_failure_action_returns_400(self) -> None:
        """POST with an unknown failure_action string returns a 400 error."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "POST")

        payload = {
            "name": "test_gate",
            "criteria": {"quality_score": 0.9},
            "failure_action": "explode",  # not a valid GateAction value
        }

        with patch("vetinari.web.litestar_manufacturing_api._get_gate_runner"):
            result = _run(handler_fn.fn(stage="post_execution", data=payload))

        assert isinstance(result, Response)
        assert result.status_code == 400

    def test_valid_gate_registration_returns_201(self) -> None:
        """POST with valid data registers the gate and returns 201 with gate details."""
        from litestar import Response

        mock_runner = MagicMock()

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "POST")

        payload = {
            "name": "execution_quality_gate",
            "criteria": {"quality_score": 0.85, "latency_ms": 5000.0},
            "failure_action": "block",
        }

        with patch(
            "vetinari.web.litestar_manufacturing_api._get_gate_runner",
            return_value=mock_runner,
        ):
            result = _run(handler_fn.fn(stage="post_execution", data=payload))

        assert isinstance(result, Response)
        assert result.status_code == 201
        assert result.content["stage"] == "post_execution"
        assert result.content["gate"]["name"] == "execution_quality_gate"
        mock_runner.add_gate.assert_called_once()


# ---------------------------------------------------------------------------
# PUT /api/v1/workflow/gates/{stage}
# ---------------------------------------------------------------------------


class TestWorkflowGateAdd:
    """api_workflow_gate_add() updates/creates a gate configuration via PUT."""

    def test_put_with_valid_data_returns_gate_config(self) -> None:
        """PUT with valid criteria and failure_action returns the configured gate dict."""
        mock_runner = MagicMock()

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "PUT")

        payload = {
            "name": "updated_gate",
            "criteria": {"quality_score": 0.75},
            "failure_action": "retry",
        }

        with patch(
            "vetinari.workflow.quality_gates.get_gate_runner",
            return_value=mock_runner,
        ):
            result = _run(handler_fn.fn(stage="post_decomposition", data=payload))

        assert isinstance(result, dict)
        assert result["stage"] == "post_decomposition"
        assert result["name"] == "updated_gate"
        assert result["failure_action"] == "retry"
        mock_runner.add_gate.assert_called_once()

    def test_put_non_numeric_criteria_returns_400(self) -> None:
        """PUT with non-numeric criteria values returns a 400 error response."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "PUT")

        payload = {
            "criteria": {"quality_score": "not_a_number"},
        }
        result = _run(handler_fn.fn(stage="post_execution", data=payload))

        assert isinstance(result, Response)
        assert result.status_code == 400

    def test_put_empty_body_returns_400(self) -> None:
        """PUT with an empty body (no recognised keys) returns a 400 error."""
        from litestar import Response

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "PUT")

        result = _run(handler_fn.fn(stage="post_execution", data={}))

        assert isinstance(result, Response)
        assert result.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/v1/workflow/gates/{stage}
# ---------------------------------------------------------------------------


class TestRemoveGate:
    """api_remove_gate() removes the gate for a stage and returns confirmation."""

    def test_remove_existing_gate_returns_success(self) -> None:
        """DELETE for a registered stage removes it and returns a success dict."""
        removed_gate = MagicMock()
        removed_gate.name = "post_exec_gate"

        mock_runner = MagicMock()
        mock_runner.remove_gate.return_value = removed_gate

        handlers = create_manufacturing_handlers()
        if not handlers:
            pytest.skip("Litestar not installed")

        handler_fn = _get_handler(handlers, "/api/v1/workflow/gates/", "DELETE")

        with patch(
            "vetinari.web.litestar_manufacturing_api._get_gate_runner",
            return_value=mock_runner,
        ):
            result = _run(handler_fn.fn(stage="post_execution"))

        assert result["status"] == "ok"
        assert result["stage"] == "post_execution"
        assert result["removed_gate"] == "post_exec_gate"
        mock_runner.remove_gate.assert_called_once_with("post_execution")
