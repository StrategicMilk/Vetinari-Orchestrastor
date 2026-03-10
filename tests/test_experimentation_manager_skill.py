"""Tests for ExperimentationManagerSkill."""
import pytest
from vetinari.tools.experimentation_manager_skill import ExperimentationManagerSkill
from vetinari.tool_interface import ToolCategory


class TestExperimentationManagerSkill:
    def test_instantiation(self):
        tool = ExperimentationManagerSkill()
        assert tool is not None
        assert tool.metadata.name == "experimentation_manager"

    def test_metadata_has_required_fields(self):
        tool = ExperimentationManagerSkill()
        assert tool.metadata.description
        assert tool.metadata.parameters
        assert tool.metadata.tags
        assert tool.metadata.category == ToolCategory.SEARCH_ANALYSIS

    def test_has_experiment_name_parameter(self):
        tool = ExperimentationManagerSkill()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "experiment_name" in param_names

    def test_experiment_name_is_required(self):
        tool = ExperimentationManagerSkill()
        param = next(p for p in tool.metadata.parameters if p.name == "experiment_name")
        assert param.required is True

    def test_execute_without_agent_returns_failure(self):
        tool = ExperimentationManagerSkill()
        result = tool.execute(experiment_name="test_experiment")
        assert result is not None
        assert result.success is False
        assert result.error is not None
