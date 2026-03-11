"""Tests for Phase 4: Agent tool wiring.

Verifies that BuilderAgent uses file_operations tool and
ConsolidatedResearcherAgent uses web_search tool when available,
and falls back gracefully when they are not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.contracts import AgentResult, AgentTask, AgentType
from vetinari.execution_context import ExecutionMode
from vetinari.tool_interface import (
    Tool, ToolMetadata, ToolCategory, ToolResult, ToolRegistry,
)


# ---------------------------------------------------------------------------
# Fake tools for testing
# ---------------------------------------------------------------------------

class _FakeFileOpsTool(Tool):
    """Tracks file write calls for assertion."""

    def __init__(self):
        metadata = ToolMetadata(
            name="file_operations",
            description="Fake file operations",
            category=ToolCategory.FILE_OPERATIONS,
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
        )
        super().__init__(metadata)
        self.calls: list[dict] = []

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, output={"written": True})


class _FakeWebSearchTool(Tool):
    """Returns canned search results."""

    def __init__(self):
        metadata = ToolMetadata(
            name="web_search",
            description="Fake web search",
            category=ToolCategory.SEARCH_ANALYSIS,
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
        )
        super().__init__(metadata)
        self.call_count = 0

    def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=True,
            output={
                "results": [
                    {"title": "Result 1", "url": "http://example.com", "snippet": "A snippet",
                     "source_reliability": "high"},
                ],
                "query": kwargs.get("query", ""),
                "total_results": 1,
                "citations": [],
            },
        )


def _make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


# ===================================================================
# BuilderAgent._tool_write_file tests
# ===================================================================

class TestBuilderToolWriteFile:

    def test_uses_file_operations_tool_when_available(self):
        from vetinari.agents.builder_agent import BuilderAgent

        fake_tool = _FakeFileOpsTool()
        registry = _make_registry(fake_tool)

        agent = BuilderAgent()
        agent.initialize({"tool_registry": registry})

        result = agent._tool_write_file("/tmp/test_file.py", "print('hello')")
        assert result is True
        assert len(fake_tool.calls) == 1
        assert fake_tool.calls[0]["operation"] == "write"
        assert fake_tool.calls[0]["path"] == "/tmp/test_file.py"
        assert fake_tool.calls[0]["content"] == "print('hello')"

    def test_falls_back_to_direct_io_without_registry(self):
        from vetinari.agents.builder_agent import BuilderAgent

        agent = BuilderAgent()
        agent.initialize({})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "fallback.py")
            result = agent._tool_write_file(path, "# fallback")
            assert result is True
            assert Path(path).read_text(encoding="utf-8") == "# fallback"

    def test_falls_back_when_tool_fails(self):
        from vetinari.agents.builder_agent import BuilderAgent

        class _FailingFileOps(Tool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="file_operations",
                    description="Always fails",
                    category=ToolCategory.FILE_OPERATIONS,
                    allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
                )
                super().__init__(metadata)

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=False, output=None, error="disk full")

        registry = _make_registry(_FailingFileOps())
        agent = BuilderAgent()
        agent.initialize({"tool_registry": registry})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "fallback2.py")
            result = agent._tool_write_file(path, "# recovered")
            assert result is True
            assert Path(path).read_text(encoding="utf-8") == "# recovered"


# ===================================================================
# ConsolidatedResearcherAgent._tool_search tests
# ===================================================================

class TestResearcherToolSearch:

    def test_uses_web_search_tool_when_available(self):
        from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent

        fake_tool = _FakeWebSearchTool()
        registry = _make_registry(fake_tool)

        agent = ConsolidatedResearcherAgent()
        agent.initialize({"tool_registry": registry})

        results = agent._tool_search("test query", max_results=3)
        assert fake_tool.call_count == 1
        assert len(results) >= 1
        assert results[0]["title"] == "Result 1"

    def test_falls_back_to_direct_search_without_registry(self):
        from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent

        agent = ConsolidatedResearcherAgent()
        agent.initialize({})

        # _search() will return [] since no web_search is configured
        results = agent._tool_search("test query")
        assert isinstance(results, list)

    def test_falls_back_when_tool_returns_failure(self):
        from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent

        class _FailingSearch(Tool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="web_search",
                    description="Always fails",
                    category=ToolCategory.SEARCH_ANALYSIS,
                    allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
                )
                super().__init__(metadata)

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=False, output=None, error="network error")

        registry = _make_registry(_FailingSearch())
        agent = ConsolidatedResearcherAgent()
        agent.initialize({"tool_registry": registry})

        # Should fall back to _search() which returns []
        results = agent._tool_search("test query")
        assert isinstance(results, list)

    def test_result_format_matches_search_helper(self):
        """Tool-based search should return dicts with the same keys as _search()."""
        from vetinari.agents.consolidated.researcher_agent import ConsolidatedResearcherAgent

        fake_tool = _FakeWebSearchTool()
        registry = _make_registry(fake_tool)

        agent = ConsolidatedResearcherAgent()
        agent.initialize({"tool_registry": registry})

        results = agent._tool_search("format check")
        assert len(results) >= 1
        expected_keys = {"title", "url", "snippet", "source_reliability"}
        assert expected_keys == set(results[0].keys())
