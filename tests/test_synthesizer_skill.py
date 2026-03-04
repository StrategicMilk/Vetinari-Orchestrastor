"""
Unit tests for Synthesizer Skill Tool
"""

import pytest
from unittest.mock import Mock
from vetinari.tools.synthesizer_skill import SynthesizerSkillTool, SynthesizerCapability, ThinkingMode
from vetinari.execution_context import ExecutionMode


class TestSynthesizerMetadata:
    def test_init(self):
        t = SynthesizerSkillTool()
        assert t.metadata.name == "synthesizer"

    def test_params(self):
        t = SynthesizerSkillTool()
        names = {p.name for p in t.metadata.parameters}
        assert "capability" in names
        assert "content" in names


class TestSynthesizerExecution:
    def setup_method(self):
        self.tool = SynthesizerSkillTool()
        self.mock_ctx = Mock()
        self.mock_ctx.mode = ExecutionMode.EXECUTION
        self.tool._context_manager = Mock(current_context=self.mock_ctx)

    def test_result_combination(self):
        r = self.tool.execute(capability="result_combination", content="data1, data2")
        assert r.success is True

    def test_summarization(self):
        r = self.tool.execute(capability="summarization", content="long content")
        assert r.success is True

    def test_report_generation(self):
        r = self.tool.execute(capability="report_generation", content="findings")
        assert r.success is True
        assert r.output.get("report") is not None

    def test_insight_extraction(self):
        r = self.tool.execute(capability="insight_extraction", content="data")
        assert r.success is True
        assert len(r.output.get("insights", [])) > 0

    def test_consolidation(self):
        r = self.tool.execute(capability="consolidation", content="info")
        assert r.success is True

    def test_presentation(self):
        r = self.tool.execute(capability="presentation", content="slides")
        assert r.success is True
        assert r.output.get("report") is not None

    def test_planning_mode(self):
        self.mock_ctx.mode = ExecutionMode.PLANNING
        r = self.tool.execute(capability="summarization", content="test")
        assert r.success is True
        assert "Planning" in r.output["summary"]

    def test_invalid_capability(self):
        r = self.tool.execute(capability="invalid", content="test")
        assert r.success is False

    def test_missing_content(self):
        r = self.tool.execute(capability="summarization")
        assert r.success is False
