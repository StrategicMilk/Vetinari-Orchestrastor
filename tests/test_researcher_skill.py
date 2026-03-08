"""
Unit tests for Researcher Skill Tool
"""

import pytest
from unittest.mock import Mock
from vetinari.tools.researcher_skill import ResearcherSkillTool, ResearcherCapability, ThinkingMode
from vetinari.execution_context import ExecutionMode


class TestResearcherMetadata:
    def test_init(self):
        t = ResearcherSkillTool()
        assert t.metadata.name == "researcher"
        assert ExecutionMode.EXECUTION in t.metadata.allowed_modes

    def test_params(self):
        t = ResearcherSkillTool()
        names = {p.name for p in t.metadata.parameters}
        assert "capability" in names
        assert "topic" in names


def _assert_researcher_ok(result):
    """Assert researcher result is either LLM success or honest LLM-unavailable failure."""
    assert result.output is not None
    if not result.success:
        assert "unavailable" in result.output.get("summary", "").lower()


class TestResearcherExecution:
    def setup_method(self):
        self.tool = ResearcherSkillTool()
        self.mock_ctx = Mock()
        self.mock_ctx.mode = ExecutionMode.EXECUTION
        self.tool._context_manager = Mock(current_context=self.mock_ctx)

    def test_deep_dive(self):
        r = self.tool.execute(capability="deep_dive", topic="AI")
        _assert_researcher_ok(r)

    def test_source_verification(self):
        r = self.tool.execute(capability="source_verification", topic="source")
        _assert_researcher_ok(r)

    def test_comparative_analysis(self):
        r = self.tool.execute(capability="comparative_analysis", topic="A vs B", criteria=["cost", "features"])
        _assert_researcher_ok(r)

    def test_fact_finding(self):
        r = self.tool.execute(capability="fact_finding", topic="facts")
        _assert_researcher_ok(r)

    def test_comprehensive_report(self):
        r = self.tool.execute(capability="comprehensive_report", topic="report")
        _assert_researcher_ok(r)

    def test_planning_mode(self):
        self.mock_ctx.mode = ExecutionMode.PLANNING
        r = self.tool.execute(capability="deep_dive", topic="AI")
        assert r.success is True
        assert "Planning" in r.output["summary"]

    def test_invalid_capability(self):
        r = self.tool.execute(capability="invalid", topic="test")
        assert r.success is False

    def test_missing_topic(self):
        r = self.tool.execute(capability="deep_dive")
        assert r.success is False

    def test_all_capabilities(self):
        for c in ResearcherCapability:
            r = self.tool.execute(capability=c.value, topic="test")
            _assert_researcher_ok(r)
