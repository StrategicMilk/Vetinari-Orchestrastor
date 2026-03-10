"""Error Recovery Skill Tool Wrapper

Provides failure analysis, retry strategies, circuit breaking, and
fallback planning as a standardized Vetinari tool.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class ErrorRecoverySkillTool(Tool):
    """Tool wrapper for the Error Recovery / Resilience agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="error_recovery",
            description="Failure analysis, retry strategies, circuit breaking, and fallback planning",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("error_info", str, "Error message or failure description", required=True),
                ToolParameter("context", str, "Additional context about the failure", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["error", "recovery", "resilience", "debugging"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.resilience_agent import ResilienceAgent
                self._agent = ResilienceAgent()
            except Exception as e:
                logger.warning(f"ResilienceAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        error_info = kwargs.get("error_info", "")
        context = kwargs.get("context", "")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="ResilienceAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="error-recovery",
                agent_type=AgentType.ERROR_RECOVERY,
                description="Analyse error and propose recovery strategy",
                prompt=f"Error: {error_info}",
                context={"error_info": error_info, "context": context},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
