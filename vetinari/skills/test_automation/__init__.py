"""Test Automation Skill Tool Wrapper

.. deprecated:: 1.1.0
   DEPRECATED: Superseded by QualitySkillTool (vetinari.skills.quality_skill).
   Will be removed in a future release.
"""

import logging
from vetinari.tool_interface import Tool, ToolMetadata, ToolResult, ToolParameter, ToolCategory
from vetinari.execution_context import ToolPermission, ExecutionMode

logger = logging.getLogger(__name__)


class TestAutomationSkill(Tool):
    """Tool wrapper for the Test Automation agent."""

    def __init__(self):
        import warnings
        warnings.warn(
            "TestAutomationSkill is deprecated since v1.1.0. "
            "Use QualitySkillTool (vetinari.skills.quality_skill) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="test_automation",
            description="Generate test suites, improve test coverage, validate code correctness, and run tests",
            category=ToolCategory.TESTING,
            parameters=[
                ToolParameter("target_code", "str", "Code to generate tests for", required=True),
                ToolParameter("test_type", "str", "Test type: unit|integration|e2e|all", required=False),
                ToolParameter("coverage_target", "float", "Target coverage percentage 0-100", required=False),
                ToolParameter("framework", "str", "Test framework: pytest|unittest|jest", required=False),
            ],
            required_permissions=[ToolPermission.CODE_EXECUTION],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.SANDBOX],
            tags=["testing", "coverage", "validation", "pytest"],
        )
        super().__init__(metadata)
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            try:
                from vetinari.agents.test_automation_agent import get_test_automation_agent
                self._agent = get_test_automation_agent()
            except Exception as e:
                logger.warning("TestAutomationAgent unavailable: %s", e)
        return self._agent

    def execute(self, **kwargs) -> ToolResult:
        agent = self._get_agent()
        if agent is None:
            return ToolResult(success=False, output=None, error="TestAutomationAgent not available")
        try:
            from vetinari.agents.contracts import AgentTask, AgentType
            task = AgentTask(
                task_id="test-auto",
                agent_type=AgentType.TEST_AUTOMATION,
                description=f"Generate {kwargs.get('test_type', 'unit')} tests",
                context=kwargs,
            )
            result = agent.execute(task)
            return ToolResult(success=result.success, output=result.output,
                              error="; ".join(result.errors) if result.errors else None)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
