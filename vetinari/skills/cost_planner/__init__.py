"""Cost Planner Skill Tool Wrapper

.. deprecated:: 1.1.0
   DEPRECATED: Superseded by OperationsSkillTool (vetinari.skills.operations_skill).
   Will be removed in a future release.
"""

import logging
from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class CostPlannerSkill(Tool):
    """Tool wrapper for the Cost Planner agent."""

    def __init__(self):
        import warnings
        warnings.warn(
            "CostPlannerSkill is deprecated since v1.1.0. "
            "Use OperationsSkillTool (vetinari.skills.operations_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="cost_planner",
            description="Analyze and optimize model selection costs, generate cost forecasts, and track budget",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("task_description", "str", "Task or project to cost-plan", required=True),
                ToolParameter("budget_usd", "float", "Budget limit in USD", required=False),
                ToolParameter("task_types", "list", "List of task types to plan for", required=False),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["cost", "budget", "optimization", "planning"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.cost_planner_agent import get_cost_planner_agent
                self._agent = get_cost_planner_agent()
            except Exception as e:
                logger.warning("CostPlannerAgent unavailable: %s", e)
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        agent = self._get_agent()
        if agent is None:
            # Fallback: use the cost optimizer directly
            try:
                from vetinari.learning.cost_optimizer import get_cost_optimizer
                optimizer = get_cost_optimizer()
                task_types = kwargs.get("task_types", ["coding", "research", "analysis"])
                forecast = optimizer.get_budget_forecast(
                    planned_tasks=len(task_types),
                    task_types=task_types,
                    models=["default"],
                )
                return ToolResult(success=True, output=forecast)
            except Exception as e:
                return ToolResult(success=False, output=None, error=str(e))

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="cost-plan",
                agent_type=AgentType.COST_PLANNER,
                description=kwargs.get("task_description", "Cost planning"),
                context=kwargs,
            )
            result = agent.execute(task)
            return ToolResult(success=result.success, output=result.output,
                              error="; ".join(result.errors) if result.errors else None)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
