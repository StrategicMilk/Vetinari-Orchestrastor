"""Documentation Agent Skill Tool Wrapper.

.. deprecated:: 1.1.0
   DEPRECATED: Superseded by OperationsSkillTool (vetinari.skills.operations_skill).
   Will be removed in a future release.
"""

from __future__ import annotations

import logging

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import Tool, ToolCategory, ToolMetadata, ToolParameter, ToolResult
from vetinari.types import ExecutionMode

logger = logging.getLogger(__name__)


class DocumentationSkill(Tool):
    """Tool wrapper for the Documentation agent."""

    def __init__(self):
        import warnings

        warnings.warn(
            "DocumentationSkill is deprecated since v1.1.0. "
            "Use OperationsSkillTool (vetinari.skills.operations_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="documentation",
            description="Generate API documentation, user guides, README files, and technical specifications",
            category=ToolCategory.SEARCH_ANALYSIS,
            parameters=[
                ToolParameter("target", "str", "Code or system to document", required=True),
                ToolParameter("doc_type", "str", "Documentation type: api|user_guide|readme|spec|all", required=False),
                ToolParameter("format", "str", "Output format: markdown|rst|html", required=False),
            ],
            required_permissions=[ToolPermission.FILE_READ, ToolPermission.FILE_WRITE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["documentation", "api_docs", "readme", "guide"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.consolidated.operations_agent import get_operations_agent

                self._agent = get_operations_agent()
            except Exception as e:
                logger.warning("DocumentationAgent unavailable: %s", e)
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="DocumentationAgent not available")
        try:
            from vetinari.agents.contracts import AgentTask
            from vetinari.types import AgentType

            task = AgentTask(
                task_id="doc-gen",
                agent_type=AgentType.DOCUMENTATION_AGENT,
                description=f"Generate {kwargs.get('doc_type', 'documentation')} for: {str(kwargs.get('target', ''))[:100]}",
                context=kwargs,
            )
            result = agent.execute(task)
            return ToolResult(
                success=result.success, output=result.output, error="; ".join(result.errors) if result.errors else None
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
