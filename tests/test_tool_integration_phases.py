"""Tests for Phase 2 (execution mode wiring) and Phase 3 (concrete tool registration).

Phase 2: Verifies that agent execution switches to EXECUTION mode.
Phase 3: Verifies FileOperationsTool and GitOperationsTool are registered.
"""

from __future__ import annotations

import pytest

from vetinari.execution_context import get_context_manager
from vetinari.tool_interface import get_tool_registry
from vetinari.types import ExecutionMode

# ===================================================================
# Phase 3: Concrete tool registration
# ===================================================================

class TestConcreteToolRegistration:
    """Verify that FileOperationsTool and GitOperationsTool can be created and registered."""

    @pytest.fixture(autouse=True)
    def _ensure_registered(self):
        """Register the concrete tools via the factory functions."""
        from vetinari.tools.tool_registry_integration import _make_file_tool, _make_git_tool
        registry = get_tool_registry()
        file_tool = _make_file_tool()
        git_tool = _make_git_tool()
        if file_tool is not None:
            registry.register(file_tool)
        if git_tool is not None:
            registry.register(git_tool)

    def test_file_operations_tool_is_registered(self):
        """FileOperationsTool should be available via the registry."""
        registry = get_tool_registry()
        tool = registry.get("file_operations")
        assert tool is not None, "file_operations tool not registered"
        assert tool.metadata.name == "file_operations"

    def test_git_operations_tool_is_registered(self):
        """GitOperationsTool should be available via the registry."""
        registry = get_tool_registry()
        tool = registry.get("git_operations")
        assert tool is not None, "git_operations tool not registered"
        assert tool.metadata.name == "git_operations"

    def test_file_tool_factory_returns_tool(self):
        """_make_file_tool should return a Tool instance."""
        from vetinari.tools.tool_registry_integration import _make_file_tool
        tool = _make_file_tool()
        assert tool is not None
        assert tool.metadata.name == "file_operations"

    def test_git_tool_factory_returns_tool(self):
        """_make_git_tool should return a Tool instance."""
        from vetinari.tools.tool_registry_integration import _make_git_tool
        tool = _make_git_tool()
        assert tool is not None
        assert tool.metadata.name == "git_operations"


# ===================================================================
# Phase 2: Execution mode wiring
# ===================================================================

class TestExecutionModeWiring:
    """Verify that the context manager supports EXECUTION mode switching."""

    def test_temporary_mode_switches_and_restores(self):
        """temporary_mode should switch to EXECUTION and restore original mode."""
        ctx_mgr = get_context_manager()
        original_mode = ctx_mgr.current_mode

        with ctx_mgr.temporary_mode(ExecutionMode.EXECUTION, task_id="test-task"):
            assert ctx_mgr.current_mode == ExecutionMode.EXECUTION

        assert ctx_mgr.current_mode == original_mode

    def test_temporary_mode_restores_on_exception(self):
        """Mode should be restored even if an exception occurs inside the block."""
        ctx_mgr = get_context_manager()
        original_mode = ctx_mgr.current_mode

        with pytest.raises(ValueError), ctx_mgr.temporary_mode(ExecutionMode.EXECUTION, task_id="test-err"):
            assert ctx_mgr.current_mode == ExecutionMode.EXECUTION
            raise ValueError("intentional")

        assert ctx_mgr.current_mode == original_mode

    def test_execution_mode_allows_execution_tools(self):
        """Tools with allowed_modes=[EXECUTION] should pass permission checks in EXECUTION mode."""
        from vetinari.tool_interface import Tool, ToolCategory, ToolMetadata, ToolResult

        class _ExecOnlyTool(Tool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="exec_only",
                    description="Only runs in EXECUTION mode",
                    category=ToolCategory.SYSTEM_OPERATIONS,
                    allowed_modes=[ExecutionMode.EXECUTION],
                )
                super().__init__(metadata)

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ran")

        tool = _ExecOnlyTool()
        ctx_mgr = get_context_manager()

        with ctx_mgr.temporary_mode(ExecutionMode.EXECUTION):
            result = tool.run()
            assert result.success is True
            assert result.output == "ran"

    def test_planning_mode_blocks_execution_only_tools(self):
        """Tools restricted to EXECUTION mode should fail in PLANNING mode."""
        from vetinari.tool_interface import Tool, ToolCategory, ToolMetadata, ToolResult

        class _ExecOnlyTool(Tool):
            def __init__(self):
                metadata = ToolMetadata(
                    name="exec_only_2",
                    description="Only runs in EXECUTION mode",
                    category=ToolCategory.SYSTEM_OPERATIONS,
                    allowed_modes=[ExecutionMode.EXECUTION],
                )
                super().__init__(metadata)

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ran")

        tool = _ExecOnlyTool()
        ctx_mgr = get_context_manager()

        with ctx_mgr.temporary_mode(ExecutionMode.PLANNING):
            result = tool.run()
            assert result.success is False
            assert "not allowed" in result.error.lower()
