"""
Unit tests for the Librarian Skill Tool
"""

import pytest
from unittest.mock import Mock

from vetinari.tools.librarian_skill import (
    LibrarianSkillTool,
    LibrarianCapability,
    ThinkingMode,
    ResearchRequest,
    ResearchResult,
)
from vetinari.execution_context import ExecutionMode, ToolPermission


class TestLibrarianToolMetadata:
    def test_initialization(self):
        tool = LibrarianSkillTool()
        assert tool.metadata.name == "librarian"
        assert tool.metadata.version == "1.0.0"

    def test_capabilities_defined(self):
        tool = LibrarianSkillTool()
        param_names = {p.name for p in tool.metadata.parameters}
        assert "capability" in param_names
        assert "query" in param_names
        assert "thinking_mode" in param_names

    def test_permissions(self):
        tool = LibrarianSkillTool()
        assert ToolPermission.NETWORK_REQUEST in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_modes(self):
        tool = LibrarianSkillTool()
        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes


class TestLibrarianToolExecution:
    def setup_method(self):
        self.tool = LibrarianSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock()
        self.mock_context.mode = ExecutionMode.EXECUTION
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_docs_lookup_execution(self):
        result = self.tool.execute(capability="docs_lookup", query="React")
        assert result.success is True

    def test_github_examples_execution(self):
        result = self.tool.execute(capability="github_examples", query="Python")
        assert result.success is True

    def test_api_reference_execution(self):
        result = self.tool.execute(capability="api_reference", query="GET /users")
        assert result.success is True

    def test_package_info_execution(self):
        result = self.tool.execute(capability="package_info", query="requests")
        assert result.success is True

    def test_best_practices_execution(self):
        result = self.tool.execute(capability="best_practices", query="error handling")
        assert result.success is True

    def test_planning_mode(self):
        self.mock_context.mode = ExecutionMode.PLANNING
        result = self.tool.execute(capability="docs_lookup", query="React")
        assert result.success is True
        assert "Planning mode" in result.output["summary"]

    def test_invalid_capability(self):
        result = self.tool.execute(capability="invalid", query="test")
        assert result.success is False
        assert "Invalid capability" in result.error

    def test_missing_query(self):
        result = self.tool.execute(capability="docs_lookup")
        assert result.success is False

    def test_invalid_thinking_mode(self):
        result = self.tool.execute(capability="docs_lookup", query="test", thinking_mode="invalid")
        assert result.success is False


class TestResearchDataclasses:
    def test_research_request_creation(self):
        request = ResearchRequest(
            capability=LibrarianCapability.DOCS_LOOKUP,
            query="test query",
        )
        assert request.query == "test query"
        assert request.capability == LibrarianCapability.DOCS_LOOKUP

    def test_research_result_creation(self):
        result = ResearchResult(success=True, summary="Test summary")
        assert result.success is True
        assert result.summary == "Test summary"

    def test_to_dict(self):
        result = ResearchResult(success=True, summary="Test", citations=["Source 1"])
        d = result.to_dict()
        assert d["success"] is True
        assert d["summary"] == "Test"
        assert "Source 1" in d["citations"]


class TestCapabilityRouting:
    def setup_method(self):
        self.tool = LibrarianSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_all_capabilities_route(self):
        caps = [c.value for c in LibrarianCapability]
        for cap in caps:
            result = self.tool.execute(capability=cap, query="test")
            assert result.success is True, f"Failed for capability: {cap}"


class TestThinkingModes:
    def setup_method(self):
        self.tool = LibrarianSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_all_thinking_modes(self):
        modes = [m.value for m in ThinkingMode]
        for mode in modes:
            result = self.tool.execute(capability="docs_lookup", query="test", thinking_mode=mode)
            assert result.success is True


class TestEdgeCases:
    def setup_method(self):
        self.tool = LibrarianSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(mode=ExecutionMode.EXECUTION)
        self.mock_ctx_manager.current_context = self.mock_context
        self.tool._context_manager = self.mock_ctx_manager

    def test_empty_query(self):
        result = self.tool.execute(capability="docs_lookup", query="")
        assert result.success is False

    def test_unicode_query(self):
        result = self.tool.execute(capability="docs_lookup", query="你好世界")
        assert result.success is True

    def test_special_chars_query(self):
        result = self.tool.execute(capability="docs_lookup", query="test!@#$%")
        assert result.success is True

    def test_with_context(self):
        result = self.tool.execute(capability="docs_lookup", query="React", context="v18")
        assert result.success is True

    def test_with_focus_areas(self):
        result = self.tool.execute(capability="best_practices", query="security", focus_areas=["xss"])
        assert result.success is True
