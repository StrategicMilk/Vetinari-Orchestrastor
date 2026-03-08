"""Context Manager Skill Tool Wrapper

Provides long-term context management, memory consolidation, and session
summarisation as a standardized Vetinari tool.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class ContextManagerSkillTool(Tool):
    """Tool wrapper for the Context Manager agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="context_manager",
            description="Long-term context management, memory consolidation, and session summarisation",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("scope", str, "Scope of context to manage", required=True),
                ToolParameter("action", str, "Action: summarize|consolidate|prune", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["context", "memory", "consolidation"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.context_manager_agent import ContextManagerAgent
                self._agent = ContextManagerAgent()
            except Exception as e:
                logger.warning(f"ContextManagerAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        scope = kwargs.get("scope", "")
        action = kwargs.get("action", "summarize")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="ContextManagerAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="context-management",
                agent_type=AgentType.CONTEXT_MANAGER,
                description=f"Context management action: {action}",
                prompt=f"Perform {action} on scope: {scope}",
                context={"scope": scope, "action": action, "mode": "consolidate"},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
