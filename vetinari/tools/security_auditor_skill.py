"""
Security Auditor Skill Tool Wrapper

Provides policy compliance checks, vulnerability scanning, and security
assessment as a standardized Vetinari tool.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class SecurityAuditorSkill(Tool):
    """Tool wrapper for the Security Auditor agent."""

    def __init__(self):
        metadata = ToolMetadata(
            name="security_auditor",
            description="Perform security audits, vulnerability scanning, and policy compliance checks",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("target", "str", "Code or system to audit", required=True),
                ToolParameter("focus", "str", "Audit focus: vulnerabilities|compliance|access_control|all", required=False),
                ToolParameter("severity_threshold", "str", "Minimum severity: low|medium|high|critical", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["security", "audit", "compliance", "vulnerability"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.security_auditor_agent import get_security_auditor_agent
                self._agent = get_security_auditor_agent()
            except Exception as e:
                logger.warning(f"SecurityAuditorAgent unavailable: {e}")
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        target = kwargs.get("target", "")
        focus = kwargs.get("focus", "all")
        severity = kwargs.get("severity_threshold", "low")

        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="SecurityAuditorAgent not available")

        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="security-audit",
                agent_type=AgentType.SECURITY_AUDITOR,
                description=f"Security audit with focus: {focus}",
                context={"target": target, "focus": focus, "severity_threshold": severity},
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success,
                output=result.output,
                error="; ".join(result.errors) if result.errors else None,
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
