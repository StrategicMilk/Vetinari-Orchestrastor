"""User Interaction Skill Tool Wrapper

Provides ambiguity detection, clarifying question generation, and context
gathering as a standardized Vetinari tool.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class UserInteractionSkillTool(Tool):
    """Tool wrapper for the User Interaction agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="user_interaction",
            description="Ambiguity detection, clarifying question generation, and context gathering",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("question", str, "Question or prompt to process", required=True),
                ToolParameter("context", str, "Optional context for the question", required=False),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["user", "interaction", "clarification", "qa"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.user_interaction_agent import UserInteractionAgent
                self._agent = UserInteractionAgent()
            except Exception as e:
                logger.warning(f"UserInteractionAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        context = kwargs.get("context", "")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="UserInteractionAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="user-interaction",
                agent_type=AgentType.USER_INTERACTION,
                description="Process user question and gather context",
                prompt=question,
                context={"question": question, "context": context, "mode": "interact"},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
