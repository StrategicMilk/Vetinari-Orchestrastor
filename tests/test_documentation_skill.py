"""Tests for DocumentationSkill."""
import pytest
from vetinari.tools.documentation_skill import DocumentationSkill
from vetinari.tool_interface import ToolCategory


class TestDocumentationSkill:
    def test_instantiation(self):
        tool = DocumentationSkill()
        assert tool is not None
        assert tool.metadata.name == "documentation"

    def test_metadata_has_required_fields(self):
        tool = DocumentationSkill()
        assert tool.metadata.description
        assert tool.metadata.parameters
        assert tool.metadata.tags
        assert tool.metadata.category == ToolCategory.SEARCH_ANALYSIS

    def test_has_target_parameter(self):
        tool = DocumentationSkill()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "target" in param_names

    def test_target_is_required(self):
        tool = DocumentationSkill()
        param = next(p for p in tool.metadata.parameters if p.name == "target")
        assert param.required is True

    def test_execute_without_agent_returns_failure(self):
        tool = DocumentationSkill()
        result = tool.execute(target="my_module.py")
        assert result is not None
        assert result.success is False
        assert result.error is not None
