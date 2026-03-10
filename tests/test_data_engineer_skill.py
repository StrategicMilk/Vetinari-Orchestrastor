"""Tests for DataEngineerSkill."""
import pytest
from vetinari.tools.data_engineer_skill import DataEngineerSkill
from vetinari.tool_interface import ToolCategory


class TestDataEngineerSkill:
    def test_instantiation(self):
        tool = DataEngineerSkill()
        assert tool is not None
        assert tool.metadata.name == "data_engineer"

    def test_metadata_has_required_fields(self):
        tool = DataEngineerSkill()
        assert tool.metadata.description
        assert tool.metadata.parameters
        assert tool.metadata.tags
        assert tool.metadata.category == ToolCategory.CODE_EXECUTION

    def test_has_task_parameter(self):
        tool = DataEngineerSkill()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "task" in param_names

    def test_task_is_required(self):
        tool = DataEngineerSkill()
        param = next(p for p in tool.metadata.parameters if p.name == "task")
        assert param.required is True

    def test_execute_without_agent_returns_failure(self):
        tool = DataEngineerSkill()
        result = tool.execute(task="create ETL pipeline")
        assert result is not None
        assert result.success is False
        assert result.error is not None
