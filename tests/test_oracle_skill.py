"""
Unit tests for Oracle Skill Tool
"""

import pytest
from unittest.mock import Mock
from vetinari.skills.oracle import OracleSkillTool, OracleCapability, ThinkingMode
from vetinari.execution_context import ExecutionMode


class TestOracleMetadata:
    def test_init(self):
        t = OracleSkillTool()
        assert t.metadata.name == "oracle"
        assert ExecutionMode.EXECUTION in t.metadata.allowed_modes

    def test_params(self):
        t = OracleSkillTool()
        names = {p.name for p in t.metadata.parameters}
        assert "capability" in names
        assert "question" in names


class TestOracleExecution:
    def setup_method(self):
        self.tool = OracleSkillTool()
        self.mock_ctx = Mock()
        self.mock_ctx.mode = ExecutionMode.EXECUTION
        self.tool._context_manager = Mock(current_context=self.mock_ctx)

    def test_architecture_analysis(self):
        r = self.tool.execute(capability="architecture_analysis", question="How to structure app?")
        assert r.success is True

    def test_trade_off_evaluation(self):
        r = self.tool.execute(capability="trade_off_evaluation", question="Which is better?", options=["A", "B"])
        assert r.success is True

    def test_debugging_strategy(self):
        r = self.tool.execute(capability="debugging_strategy", question="Why slow?")
        assert r.success is True

    def test_pattern_suggestion(self):
        r = self.tool.execute(capability="pattern_suggestion", question="What pattern for DB?")
        assert r.success is True

    def test_planning_mode(self):
        self.mock_ctx.mode = ExecutionMode.PLANNING
        r = self.tool.execute(capability="architecture_analysis", question="Test")
        assert r.success is True
        assert "Planning" in r.output["recommendation"]

    def test_invalid_capability(self):
        r = self.tool.execute(capability="invalid", question="Test")
        assert r.success is False

    def test_missing_question(self):
        r = self.tool.execute(capability="architecture_analysis")
        assert r.success is False

    def test_all_capabilities(self):
        for c in OracleCapability:
            r = self.tool.execute(capability=c.value, question="Test")
            assert r.success is True

    def test_all_thinking_modes(self):
        for m in ThinkingMode:
            r = self.tool.execute(capability="architecture_analysis", question="Test", thinking_mode=m.value)
            assert r.success is True
