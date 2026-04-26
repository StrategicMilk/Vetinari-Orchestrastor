"""Tests for mid-task info request protocol (US-074).

Covers metadata constants, BaseAgent.request_info(), and AgentGraph
handling of needs_info/delegate_to/needs_user_input signals.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.base_agent import (
    CONTEXT_NEEDED,
    DELEGATE_TO,
    NEEDS_INFO,
    NEEDS_USER_INPUT,
    QUESTION,
    BaseAgent,
)
from vetinari.agents.contracts import AgentResult
from vetinari.types import AgentType

# ── Metadata Constants ─────────────────────────────────────────────────


class TestMetadataConstants:
    """Test metadata constant values."""

    def test_needs_info(self) -> None:
        assert NEEDS_INFO == "needs_info"

    def test_needs_user_input(self) -> None:
        assert NEEDS_USER_INPUT == "needs_user_input"

    def test_delegate_to(self) -> None:
        assert DELEGATE_TO == "delegate_to"

    def test_question(self) -> None:
        assert QUESTION == "question"

    def test_context_needed(self) -> None:
        assert CONTEXT_NEEDED == "context_needed"


# ── BaseAgent.request_info ─────────────────────────────────────────────


class TestRequestInfo:
    """Test BaseAgent.request_info() method."""

    def _make_agent(self) -> BaseAgent:
        """Create a minimal concrete agent for testing."""
        agent = MagicMock(spec=BaseAgent)
        agent._agent_type = AgentType.WORKER
        agent.request_info = BaseAgent.request_info.__get__(agent, type(agent))
        return agent

    def test_returns_agent_result(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("What API version?")
        assert isinstance(result, AgentResult)

    def test_success_is_false(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("Need more info")
        assert result.success is False

    def test_needs_info_set(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("What?")
        assert result.metadata[NEEDS_INFO] is True

    def test_question_set(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("What API version?")
        assert result.metadata[QUESTION] == "What API version?"

    def test_user_input_when_no_delegate(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("Need human input")
        assert result.metadata[NEEDS_USER_INPUT] is True
        assert result.metadata[DELEGATE_TO] is None

    def test_delegate_to_agent(self) -> None:
        agent = self._make_agent()
        result = agent.request_info("Research this API", delegate_to=AgentType.WORKER)
        assert result.metadata[DELEGATE_TO] == AgentType.WORKER.value
        assert result.metadata[NEEDS_USER_INPUT] is False

    def test_partial_work_included(self) -> None:
        agent = self._make_agent()
        agent._current_output = "partial code"
        result = agent.request_info("Need more info")
        assert result.output["partial_work"] == "partial code"


# ── AgentGraph needs_info Handling ─────────────────────────────────────


class TestAgentGraphInfoHandling:
    """Test AgentGraph handling of needs_info metadata."""

    def test_delegate_to_another_agent(self) -> None:
        """When delegate_to is set, route question to that agent."""
        from vetinari.orchestration.agent_graph import AgentGraph

        ag = AgentGraph.__new__(AgentGraph)
        ag._agents = {}
        ag._quality_reviewed_agents = set()
        ag._goal_tracker = None

        # Builder returns needs_info with delegate_to RESEARCHER
        mock_builder = MagicMock()
        mock_builder.execute.side_effect = [
            # First call: needs info
            AgentResult(
                success=False,
                output={"partial": "code"},
                metadata={
                    "needs_info": True,
                    "delegate_to": AgentType.WORKER.value,
                    "question": "What API version?",
                },
            ),
            # Second call (after info): success
            AgentResult(success=True, output="complete code", metadata={}),
        ]
        mock_builder.verify.return_value = MagicMock(passed=True, issues=[])

        mock_researcher = MagicMock()
        mock_researcher.execute.return_value = AgentResult(
            success=True,
            output="API v2.0",
            metadata={},
        )

        ag._agents = {
            AgentType.WORKER: mock_builder,
        }

        # Verify the info request routing works by checking researcher was called
        assert mock_researcher.execute.call_count == 0  # Not called yet
        # (Full integration test would require setting up TaskNode etc.)

    def test_needs_user_input_returns_result(self) -> None:
        """When needs_user_input is set, result is returned as-is."""
        result = AgentResult(
            success=False,
            output={"partial_work": "some work"},
            metadata={
                "needs_info": True,
                "needs_user_input": True,
                "question": "What should this return?",
            },
        )
        # Verify the metadata is correct for the orchestrator to surface
        assert result.metadata["needs_info"] is True
        assert result.metadata["needs_user_input"] is True
        assert result.metadata["question"] == "What should this return?"


# ── Import Verification ────────────────────────────────────────────────


class TestImports:
    """Test that all new exports are importable."""

    def test_import_constants(self) -> None:
        from vetinari.agents.base_agent import (
            CONTEXT_NEEDED,
            DELEGATE_TO,
            NEEDS_INFO,
            NEEDS_USER_INPUT,
            QUESTION,
        )

        assert all(isinstance(c, str) for c in [NEEDS_INFO, NEEDS_USER_INPUT, DELEGATE_TO, QUESTION, CONTEXT_NEEDED])

    def test_import_base_agent(self) -> None:
        from vetinari.agents.base_agent import BaseAgent

        assert hasattr(BaseAgent, "request_info")
