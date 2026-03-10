"""Tests for CostPlannerSkill."""
import pytest
from unittest.mock import MagicMock, patch
from vetinari.tools.cost_planner_skill import CostPlannerSkill
from vetinari.tool_interface import ToolCategory


class TestCostPlannerSkill:
    def test_instantiation(self):
        tool = CostPlannerSkill()
        assert tool is not None
        assert tool.metadata.name == "cost_planner"

    def test_metadata_has_required_fields(self):
        tool = CostPlannerSkill()
        assert tool.metadata.description
        assert tool.metadata.parameters
        assert tool.metadata.tags
        assert tool.metadata.category == ToolCategory.SEARCH_ANALYSIS

    def test_has_task_description_parameter(self):
        tool = CostPlannerSkill()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "task_description" in param_names

    def test_task_description_is_required(self):
        tool = CostPlannerSkill()
        param = next(p for p in tool.metadata.parameters if p.name == "task_description")
        assert param.required is True

    def test_execute_without_agent_returns_result(self):
        tool = CostPlannerSkill()
        # Agent unavailable in test environment; expect ToolResult
        result = tool.execute(task_description="test cost planning")
        assert result is not None
        # Either success via fallback or failure with error message
        assert isinstance(result.success, bool)
