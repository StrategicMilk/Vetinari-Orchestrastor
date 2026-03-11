"""Tests for BaseAgent tool registry access methods (_use_tool, _has_tool, _list_tools).

Phase 1 of the tool layer integration plan.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult
from vetinari.execution_context import ExecutionMode
from vetinari.tool_interface import Tool, ToolMetadata, ToolCategory, ToolResult, ToolRegistry


# ---------------------------------------------------------------------------
# Concrete agent stub for testing (BaseAgent is abstract)
# ---------------------------------------------------------------------------

class _StubAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent helpers."""

    def __init__(self):
        super().__init__(AgentType.BUILDER)

    def execute(self, task: AgentTask) -> AgentResult:
        return AgentResult(success=True, output="stub")

    def verify(self, output: Any) -> VerificationResult:
        return VerificationResult(passed=True, issues=[])

    def get_system_prompt(self) -> str:
        return "You are a stub agent."


# ---------------------------------------------------------------------------
# Fake tool for registry tests
# ---------------------------------------------------------------------------

class _FakeTool(Tool):
    """A simple tool that echoes its kwargs."""

    def __init__(self, name: str = "echo_tool"):
        metadata = ToolMetadata(
            name=name,
            description="Echoes kwargs",
            category=ToolCategory.SYSTEM_OPERATIONS,
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=kwargs)


class _FailingTool(Tool):
    """A tool that always raises."""

    def __init__(self):
        metadata = ToolMetadata(
            name="failing_tool",
            description="Always fails",
            category=ToolCategory.SYSTEM_OPERATIONS,
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_with_registry(*tools: Tool) -> _StubAgent:
    """Create a StubAgent initialized with a registry containing the given tools."""
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    agent = _StubAgent()
    agent.initialize({"tool_registry": registry})
    return agent


def _make_agent_without_registry() -> _StubAgent:
    """Create a StubAgent initialized without a tool registry."""
    agent = _StubAgent()
    agent.initialize({})
    return agent


# ===================================================================
# _has_tool tests
# ===================================================================

class TestHasTool:

    def test_returns_true_for_registered_tool(self):
        agent = _make_agent_with_registry(_FakeTool("my_tool"))
        assert agent._has_tool("my_tool") is True

    def test_returns_false_for_missing_tool(self):
        agent = _make_agent_with_registry(_FakeTool("my_tool"))
        assert agent._has_tool("nonexistent") is False

    def test_returns_false_when_registry_is_none(self):
        agent = _make_agent_without_registry()
        assert agent._has_tool("anything") is False

    def test_returns_false_for_empty_registry(self):
        agent = _make_agent_with_registry()  # no tools
        assert agent._has_tool("anything") is False


# ===================================================================
# _list_tools tests
# ===================================================================

class TestListTools:

    def test_lists_all_registered_tool_names(self):
        agent = _make_agent_with_registry(
            _FakeTool("alpha"), _FakeTool("beta"), _FakeTool("gamma"),
        )
        names = agent._list_tools()
        assert sorted(names) == ["alpha", "beta", "gamma"]

    def test_returns_empty_list_when_registry_is_none(self):
        agent = _make_agent_without_registry()
        assert agent._list_tools() == []

    def test_returns_empty_list_for_empty_registry(self):
        agent = _make_agent_with_registry()
        assert agent._list_tools() == []


# ===================================================================
# _use_tool tests
# ===================================================================

class TestUseTool:

    def test_returns_result_dict_on_success(self):
        agent = _make_agent_with_registry(_FakeTool("echo"))
        result = agent._use_tool("echo", greeting="hello")
        assert result is not None
        assert result["success"] is True
        assert result["output"]["greeting"] == "hello"

    def test_returns_none_when_registry_is_none(self):
        agent = _make_agent_without_registry()
        result = agent._use_tool("echo", greeting="hello")
        assert result is None

    def test_returns_none_for_missing_tool(self):
        agent = _make_agent_with_registry(_FakeTool("echo"))
        result = agent._use_tool("nonexistent")
        assert result is None

    def test_returns_error_dict_when_tool_raises(self):
        agent = _make_agent_with_registry(_FailingTool())
        result = agent._use_tool("failing_tool")
        assert result is not None
        assert result["success"] is False
        assert "intentional failure" in result["error"]

    def test_result_contains_expected_keys(self):
        agent = _make_agent_with_registry(_FakeTool("echo"))
        result = agent._use_tool("echo")
        assert result is not None
        expected_keys = {"success", "output", "error", "execution_time_ms", "metadata"}
        assert expected_keys == set(result.keys())

    def test_multiple_tools_selects_correct_one(self):
        agent = _make_agent_with_registry(
            _FakeTool("tool_a"), _FakeTool("tool_b"),
        )
        result = agent._use_tool("tool_b", x=42)
        assert result is not None
        assert result["success"] is True
        assert result["output"]["x"] == 42


# ===================================================================
# Integration: uninitialized agent
# ===================================================================

class TestUninitializedAgent:

    def test_has_tool_false_before_initialize(self):
        agent = _StubAgent()
        # _tool_registry is set in initialize(), not in __init__
        assert agent._has_tool("anything") is False

    def test_use_tool_none_before_initialize(self):
        agent = _StubAgent()
        assert agent._use_tool("anything") is None

    def test_list_tools_empty_before_initialize(self):
        agent = _StubAgent()
        assert agent._list_tools() == []
