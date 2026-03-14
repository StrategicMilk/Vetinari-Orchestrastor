"""Data Engineer Skill Tool Wrapper.

.. deprecated:: 1.1.0
   DEPRECATED: Superseded by ArchitectSkillTool (vetinari.skills.architect_skill).
   Will be removed in a future release.
"""

from __future__ import annotations

import logging

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import Tool, ToolCategory, ToolMetadata, ToolParameter, ToolResult
from vetinari.types import ExecutionMode

logger = logging.getLogger(__name__)


class DataEngineerSkill(Tool):
    """Tool wrapper for the Data Engineer agent."""

    def __init__(self):
        import warnings

        warnings.warn(
            "DataEngineerSkill is deprecated since v1.1.0. "
            "Use ArchitectSkillTool (vetinari.skills.architect_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="data_engineer",
            description="Design and implement data pipelines, schemas, ETL workflows, and database migrations",
            category=ToolCategory.CODE_EXECUTION,
            parameters=[
                ToolParameter("task", "str", "Data engineering task description", required=True),
                ToolParameter("data_source", "str", "Source data system or format", required=False),
                ToolParameter("target", "str", "Target data system or format", required=False),
                ToolParameter("schema", "dict", "Existing schema or requirements", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ, ToolPermission.FILE_WRITE],
            allowed_modes=[ExecutionMode.EXECUTION],
            tags=["data", "pipeline", "etl", "schema", "database"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.consolidated.researcher_agent import get_consolidated_researcher_agent

                self._agent = get_consolidated_researcher_agent()
            except Exception as e:
                logger.warning("DataEngineerAgent unavailable: %s", e)
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="DataEngineerAgent not available")
        try:
            from vetinari.agents.contracts import AgentTask
            from vetinari.types import AgentType

            task = AgentTask(
                task_id="data-eng",
                agent_type=AgentType.DATA_ENGINEER,
                description=kwargs.get("task", "Data engineering task"),
                context=kwargs,
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success, output=result.output, error="; ".join(result.errors) if result.errors else None
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
