"""Improvement Skill Tool Wrapper

Provides meta-analysis, optimization recommendations, and system performance
improvement suggestions as a standardized Vetinari tool.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class ImprovementSkillTool(Tool):
    """Tool wrapper for the Improvement / Meta agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="improvement",
            description="Meta-analysis, optimization recommendations, and system performance improvement",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("target", str, "System or code to analyse for improvements", required=True),
                ToolParameter("focus", str, "Focus area: performance|quality|process|all", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["improvement", "optimization", "meta"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.improvement_agent import ImprovementAgent
                self._agent = ImprovementAgent()
            except Exception as e:
                logger.warning(f"ImprovementAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        target = kwargs.get("target", "")
        focus = kwargs.get("focus", "all")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="ImprovementAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="improvement-analysis",
                agent_type=AgentType.IMPROVEMENT,
                description=f"Improvement analysis with focus: {focus}",
                prompt=f"Suggest improvements for: {target}",
                context={"target": target, "focus": focus, "mode": "suggest"},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
