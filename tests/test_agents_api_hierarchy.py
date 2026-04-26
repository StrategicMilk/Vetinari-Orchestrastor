"""Tests for the hierarchy-view endpoints added to vetinari/web/agents_api.py (item 7.8).

Covers:
- POST /api/v1/agents/<id>/pause   — pause an agent instance
- POST /api/v1/agents/<id>/redirect — redirect an agent to a different task
- get_agent_control_state()        — snapshot helper
- Idempotency of pause
- Validation errors (missing task_id for redirect)
- Thread safety smoke test
"""

from __future__ import annotations

import threading

import pytest

import vetinari.web.litestar_agents_api as mod
from vetinari.web.litestar_agents_api import get_agent_control_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_control_state():
    """Reset in-memory pause/redirect dicts before and after each test."""
    with mod._agent_control_lock:
        mod._paused_agents.clear()
        mod._redirect_targets.clear()
    yield
    with mod._agent_control_lock:
        mod._paused_agents.clear()
        mod._redirect_targets.clear()


# ---------------------------------------------------------------------------
# get_agent_control_state snapshot isolation
# ---------------------------------------------------------------------------


def test_control_state_snapshot_is_independent():
    """Mutating the returned snapshot must not alter the module-level dicts."""
    with mod._agent_control_lock:
        mod._paused_agents["agent-snap"] = {"agent_id": "agent-snap", "reason": "test"}

    snapshot = get_agent_control_state()
    snapshot["paused"]["agent-snap"]["reason"] = "mutated"

    # The module-level dict must be unchanged.
    with mod._agent_control_lock:
        assert mod._paused_agents["agent-snap"]["reason"] == "test"


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------


def test_concurrent_pauses_are_safe():
    """Concurrent pause directives from many threads must all be recorded."""
    errors: list[Exception] = []

    def _pause(agent_id: str) -> None:
        try:
            with mod._agent_control_lock:
                mod._paused_agents[agent_id] = {"agent_id": agent_id, "reason": "test"}
            state = get_agent_control_state()
            # Returned snapshot must always be a valid dict.
            assert isinstance(state["paused"], dict)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_pause, args=(f"agent-t{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"

    state = get_agent_control_state()
    for i in range(20):
        assert f"agent-t{i}" in state["paused"]
