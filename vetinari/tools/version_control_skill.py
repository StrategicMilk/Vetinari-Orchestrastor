"""Version Control Skill Tool Wrapper

Provides git operations, branch strategy, PR creation, and code review
coordination as a standardized Vetinari tool.
"""

import logging

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class VersionControlSkillTool(Tool):
    """Tool wrapper for the Version Control agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="version_control",
            description="Git operations, branch strategy, PR creation, and code review coordination",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("target", str, "Repository or branch to operate on", required=True),
                ToolParameter("action", str, "Action: review|history|branch_strategy", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["git", "version-control", "commits", "pr"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.version_control_agent import VersionControlAgent
                self._agent = VersionControlAgent()
            except Exception as e:
                logger.warning(f"VersionControlAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        target = kwargs.get("target", "")
        action = kwargs.get("action", "review")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="VersionControlAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="version-control",
                agent_type=AgentType.VERSION_CONTROL,
                description=f"Version control action: {action}",
                prompt=f"Perform {action} on: {target}",
                context={"target": target, "action": action, "mode": "git"},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
