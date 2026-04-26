"""Tests for Foreman pure coordinator enforcement (ADR-0093).

Verifies that ForemanAgent._infer() only allows planning/coordination
inference and blocks task-execution inference attempts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.planner_agent import ForemanAgent
from vetinari.exceptions import JurisdictionViolation


@pytest.fixture
def foreman() -> ForemanAgent:
    """Create a ForemanAgent with mocked dependencies for unit testing."""
    with patch("vetinari.agents.planner_agent.get_plan_cache"):
        agent = ForemanAgent(config={"model_id": "test-model"})
    # Stub out adapter manager so _infer's super() doesn't need a real LLM
    agent._adapter_manager = MagicMock()
    return agent


class TestForemanInferenceGuard:
    """Foreman._infer() must only work in planning modes."""

    def test_infer_blocked_when_no_mode_set(self, foreman: ForemanAgent) -> None:
        """_infer() raises JurisdictionViolation when _current_mode is None."""
        assert foreman._current_mode is None
        with pytest.raises(JurisdictionViolation, match="not a planning mode"):
            foreman._infer("Generate some code")

    def test_infer_blocked_for_non_planning_mode(self, foreman: ForemanAgent) -> None:
        """_infer() raises JurisdictionViolation for a non-planning mode."""
        foreman._current_mode = "execute_code"
        with pytest.raises(JurisdictionViolation, match="execute_code"):
            foreman._infer("Generate some code")

    @pytest.mark.parametrize("mode", sorted(ForemanAgent.MODES.keys()))
    def test_infer_allowed_for_all_planning_modes(self, foreman: ForemanAgent, mode: str) -> None:
        """_infer() passes through to super() for every defined planning mode."""
        foreman._current_mode = mode
        with patch.object(ForemanAgent.__bases__[0], "_infer", return_value="mocked response") as mock_infer:
            result = foreman._infer("Plan this task")
        assert result == "mocked response"
        mock_infer.assert_called_once()

    def test_planning_modes_matches_all_defined_modes(self) -> None:
        """_PLANNING_MODES covers every mode in MODES — no gaps."""
        assert frozenset(ForemanAgent.MODES.keys()) == ForemanAgent._PLANNING_MODES

    def test_jurisdiction_error_message_includes_allowed_modes(self, foreman: ForemanAgent) -> None:
        """Error message lists all allowed modes for debuggability."""
        foreman._current_mode = "bad_mode"
        with pytest.raises(JurisdictionViolation) as exc_info:
            foreman._infer("Do something")
        msg = str(exc_info.value)
        assert "bad_mode" in msg
        assert "planning mode" in msg
        # All 6 modes should appear in the allowed list
        for mode in ForemanAgent.MODES:
            assert mode in msg
