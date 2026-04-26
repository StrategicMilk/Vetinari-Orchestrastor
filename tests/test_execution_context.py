"""
Unit tests for Execution Context System (Phase 2)

Tests the ExecutionMode, ToolPermission, ContextManager, and ExecutionContext classes.
"""

import contextvars

import pytest

from vetinari.exceptions import SecurityError
from vetinari.execution_context import (
    DEFAULT_POLICIES,
    ContextManager,
    ExecutionContext,
    ExecutionMode,
    PermissionPolicy,
    ToolPermission,
    _context_stack_var,
    get_context_manager,
)


class TestExecutionMode:
    """Test ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test that execution modes have correct values."""
        assert ExecutionMode.PLANNING.value == "planning"
        assert ExecutionMode.EXECUTION.value == "execution"
        assert ExecutionMode.SANDBOX.value == "sandbox"

    def test_execution_mode_creation(self):
        """Test creating execution modes."""
        mode = ExecutionMode("planning")
        assert mode == ExecutionMode.PLANNING


class TestToolPermission:
    """Test ToolPermission enum."""

    def test_all_permissions_defined(self):
        """Test that all expected permissions are defined."""
        expected = [
            "FILE_READ",
            "FILE_WRITE",
            "FILE_DELETE",
            "BASH_EXECUTE",
            "PYTHON_EXECUTE",
            "MODEL_INFERENCE",
            "MODEL_DISCOVERY",
            "NETWORK_REQUEST",
            "DATABASE_WRITE",
            "MEMORY_WRITE",
            "GIT_COMMIT",
            "GIT_PUSH",
        ]
        for perm_name in expected:
            assert hasattr(ToolPermission, perm_name)


class TestPermissionPolicy:
    """Test PermissionPolicy class."""

    def test_has_permission(self):
        """Test permission checking."""
        policy = PermissionPolicy(
            mode=ExecutionMode.PLANNING,
            allowed_permissions={ToolPermission.FILE_READ},
        )
        assert policy.has_permission(ToolPermission.FILE_READ)
        assert not policy.has_permission(ToolPermission.FILE_WRITE)

    def test_requires_confirmation(self):
        """Test confirmation requirement checking."""
        policy = PermissionPolicy(
            mode=ExecutionMode.EXECUTION,
            allowed_permissions={ToolPermission.FILE_DELETE},
            require_confirmation={ToolPermission.FILE_DELETE},
        )
        assert policy.requires_confirmation(ToolPermission.FILE_DELETE)
        assert not policy.requires_confirmation(ToolPermission.FILE_READ)


class TestDefaultPolicies:
    """Test default permission policies."""

    def test_planning_mode_restrictions(self):
        """Test planning mode has read-only restrictions."""
        policy = DEFAULT_POLICIES[ExecutionMode.PLANNING]

        # Should allow reads
        assert policy.has_permission(ToolPermission.FILE_READ)
        assert policy.has_permission(ToolPermission.MODEL_INFERENCE)

        # Should deny writes
        assert not policy.has_permission(ToolPermission.FILE_WRITE)
        assert not policy.has_permission(ToolPermission.FILE_DELETE)
        assert not policy.has_permission(ToolPermission.GIT_PUSH)

    def test_execution_mode_full_access(self):
        """Test execution mode has full access."""
        policy = DEFAULT_POLICIES[ExecutionMode.EXECUTION]

        # Should allow all basic operations
        assert policy.has_permission(ToolPermission.FILE_READ)
        assert policy.has_permission(ToolPermission.FILE_WRITE)
        assert policy.has_permission(ToolPermission.BASH_EXECUTE)
        assert policy.has_permission(ToolPermission.MODEL_INFERENCE)

    def test_sandbox_mode_restrictions(self):
        """Test sandbox mode has restricted access."""
        policy = DEFAULT_POLICIES[ExecutionMode.SANDBOX]

        # Should allow limited operations
        assert policy.has_permission(ToolPermission.FILE_READ)
        assert policy.has_permission(ToolPermission.PYTHON_EXECUTE)

        # Should deny dangerous operations
        assert not policy.has_permission(ToolPermission.FILE_WRITE)
        assert not policy.has_permission(ToolPermission.BASH_EXECUTE)


class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_context_creation(self):
        """Test creating an execution context."""
        ctx = ExecutionContext(mode=ExecutionMode.EXECUTION)
        assert ctx.mode == ExecutionMode.EXECUTION
        assert ctx.policy is not None
        assert isinstance(ctx.policy, PermissionPolicy)

    def test_can_execute_permission(self):
        """Test permission checking."""
        ctx = ExecutionContext(mode=ExecutionMode.PLANNING)
        assert ctx.can_execute(ToolPermission.FILE_READ)
        assert not ctx.can_execute(ToolPermission.FILE_WRITE)

    def test_record_operation(self):
        """Test operation recording in audit trail."""
        ctx = ExecutionContext(mode=ExecutionMode.EXECUTION)
        assert len(ctx.executed_operations) == 0

        ctx.record_operation("test_op", {"param": "value"}, {"result": "success"})
        assert len(ctx.executed_operations) == 1

        trail = ctx.get_audit_trail()
        assert trail[0]["operation"] == "test_op"

    def test_hooks(self):
        """Test pre and post execution hooks."""
        ctx = ExecutionContext(mode=ExecutionMode.EXECUTION)

        pre_called = []
        post_called = []

        def pre_hook(op_name, params):
            pre_called.append((op_name, params))
            return True

        def post_hook(op_name, params, result):
            post_called.append((op_name, params, result))

        ctx.add_pre_execution_hook(pre_hook)
        ctx.add_post_execution_hook(post_hook)

        assert len(ctx.pre_execution_hooks) == 1
        assert len(ctx.post_execution_hooks) == 1


class TestContextManager:
    """Test ContextManager class."""

    def test_context_manager_creation(self):
        """Test creating context manager."""
        manager = ContextManager()
        assert manager.current_mode == ExecutionMode.PLANNING
        assert manager.current_context is not None
        assert isinstance(manager.current_context, ExecutionContext)

    def test_switch_mode(self):
        """Test switching execution modes."""
        manager = ContextManager()
        assert manager.current_mode == ExecutionMode.PLANNING

        manager.switch_mode(ExecutionMode.EXECUTION, task_id="t1")
        assert manager.current_mode == ExecutionMode.EXECUTION
        assert manager.current_context.active_task_id == "t1"

    def test_context_stacking(self):
        """Test context stack management."""
        manager = ContextManager()
        initial_mode = manager.current_mode

        manager.switch_mode(ExecutionMode.EXECUTION)
        manager.switch_mode(ExecutionMode.SANDBOX)
        assert manager.current_mode == ExecutionMode.SANDBOX

        manager.pop_context()
        assert manager.current_mode == ExecutionMode.EXECUTION

        manager.pop_context()
        assert manager.current_mode == initial_mode

    def test_check_permission(self):
        """Test permission checking through manager."""
        manager = ContextManager()
        manager.switch_mode(ExecutionMode.EXECUTION)

        # Should have execution permissions
        assert manager.check_permission(ToolPermission.FILE_WRITE)

        manager.pop_context()
        manager.switch_mode(ExecutionMode.PLANNING)

        # Should not have write permissions
        assert not manager.check_permission(ToolPermission.FILE_WRITE)

    def test_enforce_permission(self):
        """Test permission enforcement."""
        manager = ContextManager()
        manager.switch_mode(ExecutionMode.PLANNING)

        # Should raise SecurityError
        with pytest.raises(SecurityError):
            manager.enforce_permission(ToolPermission.FILE_WRITE, "test_operation")

        manager.pop_context()
        manager.switch_mode(ExecutionMode.EXECUTION)

        # Should not raise
        manager.enforce_permission(ToolPermission.FILE_WRITE, "test_operation")

    def test_temporary_mode(self):
        """Test temporary mode context manager."""
        manager = ContextManager()
        initial_mode = manager.current_mode

        with manager.temporary_mode(ExecutionMode.EXECUTION):
            assert manager.current_mode == ExecutionMode.EXECUTION

        assert manager.current_mode == initial_mode

    def test_get_status(self):
        """Test getting status information."""
        manager = ContextManager()
        manager.switch_mode(ExecutionMode.EXECUTION, task_id="t1")

        status = manager.get_status()
        assert status["mode"] == "execution"
        assert status["task_id"] == "t1"
        assert "permissions" in status


class TestGlobalContextManager:
    """Test global context manager singleton."""

    def test_get_context_manager(self):
        """Test getting global context manager."""
        manager1 = get_context_manager()
        manager2 = get_context_manager()

        # Should be same instance
        assert manager1 is manager2

    def test_global_context_persistence(self):
        """Test that global context persists across calls."""
        manager = get_context_manager()
        manager.switch_mode(ExecutionMode.EXECUTION)

        # Get another reference
        manager2 = get_context_manager()
        assert manager2.current_mode == ExecutionMode.EXECUTION


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_planning_to_execution_workflow(self):
        """Test switching from planning to execution mode."""
        manager = ContextManager()

        # Start in planning mode
        manager.switch_mode(ExecutionMode.PLANNING, task_id="t1")
        assert not manager.check_permission(ToolPermission.FILE_WRITE)

        # Switch to execution
        manager.pop_context()
        manager.switch_mode(ExecutionMode.EXECUTION, task_id="t1")
        assert manager.check_permission(ToolPermission.FILE_WRITE)

        # Record operations
        ctx = manager.current_context
        ctx.record_operation("task_t1", {}, {"status": "completed"})

        assert len(ctx.executed_operations) == 1

    def test_permission_matrix(self):
        """Test permission matrix across modes."""
        manager = ContextManager()

        test_permissions = [
            ToolPermission.FILE_READ,
            ToolPermission.FILE_WRITE,
            ToolPermission.BASH_EXECUTE,
            ToolPermission.GIT_PUSH,
        ]

        # Test planning mode
        manager.switch_mode(ExecutionMode.PLANNING)
        planning_perms = [manager.check_permission(p) for p in test_permissions]
        manager.pop_context()

        # Test execution mode
        manager.switch_mode(ExecutionMode.EXECUTION)
        execution_perms = [manager.check_permission(p) for p in test_permissions]

        # Execution should have more permissions than planning
        assert sum(execution_perms) > sum(planning_perms)


class TestContextManagerThreadSafety:
    """Tests for Bug 1: per-thread stack isolation in ContextManager."""

    def test_thread_mode_isolation(self):
        """Two threads switching modes simultaneously must not bleed state into each other.

        Thread A switches to EXECUTION; Thread B switches to SANDBOX.
        After each thread pops its context, neither thread must see the
        other's mode pushed onto its own stack.
        """
        import threading

        from vetinari.execution_context import ContextManager
        from vetinari.types import ExecutionMode

        manager = ContextManager()
        results: dict[str, ExecutionMode] = {}
        errors: list[Exception] = []

        def thread_a() -> None:
            try:
                manager.switch_mode(ExecutionMode.EXECUTION)
                # Small yield so threads overlap
                import time

                time.sleep(0.01)
                results["a_inner"] = manager.current_mode
                manager.pop_context()
                results["a_outer"] = manager.current_mode
            except Exception as exc:
                errors.append(exc)

        def thread_b() -> None:
            try:
                manager.switch_mode(ExecutionMode.SANDBOX)
                import time

                time.sleep(0.01)
                results["b_inner"] = manager.current_mode
                manager.pop_context()
                results["b_outer"] = manager.current_mode
            except Exception as exc:
                errors.append(exc)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert not errors, f"Thread(s) raised: {errors}"
        # Each thread must have seen its own mode, not the other's
        assert results["a_inner"] == ExecutionMode.EXECUTION
        assert results["b_inner"] == ExecutionMode.SANDBOX
        # After pop, each thread returns to its own default (PLANNING)
        assert results["a_outer"] == ExecutionMode.PLANNING
        assert results["b_outer"] == ExecutionMode.PLANNING

    def test_thread_stack_does_not_share_operations(self):
        """Operations recorded by thread A must not appear in thread B's context."""
        import threading

        from vetinari.execution_context import ContextManager
        from vetinari.types import ExecutionMode

        manager = ContextManager()
        op_counts: dict[str, int] = {}
        errors: list[Exception] = []

        def thread_a() -> None:
            try:
                manager.switch_mode(ExecutionMode.EXECUTION)
                ctx = manager.current_context
                ctx.record_operation("tool_a", {}, {"ok": True})
                ctx.record_operation("tool_a2", {}, {"ok": True})
                op_counts["a"] = len(ctx.executed_operations)
                manager.pop_context()
            except Exception as exc:
                errors.append(exc)

        def thread_b() -> None:
            try:
                manager.switch_mode(ExecutionMode.EXECUTION)
                ctx = manager.current_context
                # Thread B records nothing — its count must be 0
                op_counts["b"] = len(ctx.executed_operations)
                manager.pop_context()
            except Exception as exc:
                errors.append(exc)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert not errors, f"Thread(s) raised: {errors}"
        assert op_counts["a"] == 2
        assert op_counts["b"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
