"""Experimentation Manager Skill Tool Wrapper

.. deprecated:: 1.1.0
   DEPRECATED: Superseded by OperationsSkillTool (vetinari.skills.operations_skill).
   Will be removed in a future release.
"""

import logging
from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class ExperimentationManagerSkill(Tool):
    """Tool wrapper for the Experimentation Manager agent."""

    def __init__(self):
        import warnings
        warnings.warn(
            "ExperimentationManagerSkill is deprecated since v1.1.0. "
            "Use OperationsSkillTool (vetinari.skills.operations_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="experimentation_manager",
            description="Track experiments, manage A/B tests, version models, and analyze experiment results",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("experiment_name", "str", "Name of the experiment", required=True),
                ToolParameter("hypothesis", "str", "What is being tested", required=False),
                ToolParameter("variants", "list", "List of variant descriptions", required=False),
                ToolParameter("metric", "str", "Success metric to track", required=False),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["experiment", "ab_test", "versioning", "metrics"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.experimentation_manager_agent import get_experimentation_manager_agent
                self._agent = get_experimentation_manager_agent()
            except Exception as e:
                logger.warning("ExperimentationManagerAgent unavailable: %s", e)
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="ExperimentationManagerAgent not available")
        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="exp-mgr",
                agent_type=AgentType.EXPERIMENTATION_MANAGER,
                description=f"Manage experiment: {kwargs.get('experiment_name', 'experiment')}",
                context=kwargs,
            )
            result = agent.execute(task)
            return ToolResult(success=result.success, output=result.output,
                              error="; ".join(result.errors) if result.errors else None)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
