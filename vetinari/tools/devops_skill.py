"""DevOps Skill Tool Wrapper

Provides CI/CD pipeline design, containerisation, IaC, deployment,
and monitoring capabilities as a standardized Vetinari tool.
"""

import logging
from typing import Any, Dict, Optional

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class DevOpsSkillTool(Tool):
    """Tool wrapper for the DevOps agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="devops",
            description="CI/CD pipeline design, containerisation, IaC, deployment, and monitoring",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("target", str, "System or project to analyse", required=True),
                ToolParameter("focus", str, "Focus area: ci|deployment|monitoring|all", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["devops", "ci", "deployment", "infrastructure"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.devops_agent import DevOpsAgent
                self._agent = DevOpsAgent()
            except Exception as e:
                logger.warning(f"DevOpsAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        target = kwargs.get("target", "")
        focus = kwargs.get("focus", "all")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="DevOpsAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="devops-analysis",
                agent_type=AgentType.DEVOPS,
                description=f"DevOps analysis with focus: {focus}",
                prompt=f"Analyse devops for: {target}",
                context={"target": target, "focus": focus, "domain": "devops"},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
