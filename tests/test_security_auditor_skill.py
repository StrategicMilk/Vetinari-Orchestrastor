"""Tests for SecurityAuditorSkill."""
import pytest
from vetinari.tools.security_auditor_skill import SecurityAuditorSkill
from vetinari.tool_interface import ToolCategory

pytestmark = pytest.mark.security


class TestSecurityAuditorSkill:
    def test_instantiation(self):
        tool = SecurityAuditorSkill()
        assert tool is not None
        assert tool.metadata.name == "security_auditor"

    def test_metadata_has_required_fields(self):
        tool = SecurityAuditorSkill()
        assert tool.metadata.description
        assert tool.metadata.parameters
        assert tool.metadata.tags
        assert tool.metadata.category == ToolCategory.SEARCH_ANALYSIS

    def test_has_target_parameter(self):
        tool = SecurityAuditorSkill()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "target" in param_names

    def test_target_is_required(self):
        tool = SecurityAuditorSkill()
        param = next(p for p in tool.metadata.parameters if p.name == "target")
        assert param.required is True

    def test_execute_without_agent_returns_failure(self):
        tool = SecurityAuditorSkill()
        result = tool.execute(target="src/app.py")
        assert result is not None
        assert result.success is False
        assert result.error is not None
