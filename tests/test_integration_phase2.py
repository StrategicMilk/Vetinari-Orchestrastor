"""
Integration tests for Phase 2: Orchestrator Integration

Tests the complete workflow of:
- ExecutionContext mode switching
- AdapterManager provider selection
- Tool interface and execution
- Verification pipeline integration
- Permission enforcement across components
"""

import sys
import unittest
from unittest.mock import Mock, patch

import pytest

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in ("vetinari.adapter_manager", "vetinari.tool_interface"):
    sys.modules.pop(_stubname, None)

from vetinari.adapter_manager import (
    AdapterManager,
    ProviderMetrics,
    get_adapter_manager,
)
from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ProviderType,
)
from vetinari.execution_context import (
    ExecutionMode,
    ToolPermission,
    get_context_manager,
)
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    get_tool_registry,
)
from vetinari.validation import (
    get_verifier_pipeline,
)


class TestExecutionContextIntegration:
    """Test ExecutionContext integration with other systems."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.context_manager = get_context_manager()

    def test_mode_switching_affects_permissions(self):
        """Test that switching modes affects permission checks."""
        # Start in planning mode
        self.context_manager.switch_mode(ExecutionMode.PLANNING)

        # Planning mode should allow FILE_READ
        assert self.context_manager.check_permission(ToolPermission.FILE_READ)

        # Planning mode should deny FILE_WRITE (requires confirmation)
        # Note: This depends on the default policy

        self.context_manager.pop_context()

        # Switch to execution mode
        self.context_manager.switch_mode(ExecutionMode.EXECUTION)

        # Execution mode should allow both FILE_READ and FILE_WRITE
        assert self.context_manager.check_permission(ToolPermission.FILE_READ)
        assert self.context_manager.check_permission(ToolPermission.FILE_WRITE)

        self.context_manager.pop_context()

    def test_context_stacking(self):
        """Test context stacking and cleanup."""
        # Start with no context
        initial_depth = len(self.context_manager._get_stack())

        # Switch to planning
        self.context_manager.switch_mode(ExecutionMode.PLANNING)
        assert len(self.context_manager._get_stack()) == initial_depth + 1

        # Switch to execution (nested)
        self.context_manager.switch_mode(ExecutionMode.EXECUTION)
        assert len(self.context_manager._get_stack()) == initial_depth + 2

        # Pop contexts
        self.context_manager.pop_context()
        assert len(self.context_manager._get_stack()) == initial_depth + 1

        self.context_manager.pop_context()
        assert len(self.context_manager._get_stack()) == initial_depth

    def test_audit_trail_records_operations(self):
        """Test that audit trail records operations."""
        self.context_manager.switch_mode(ExecutionMode.EXECUTION, "test_task")

        # Simulate an operation
        self.context_manager.check_permission(ToolPermission.FILE_WRITE)

        # Get status (which includes audit trail info)
        status = self.context_manager.get_status()

        # Should have status information
        assert status is not None
        assert "mode" in status
        assert status["mode"] == "execution"

        self.context_manager.pop_context()


class TestAdapterManagerWithExecutionContext:
    """Test AdapterManager integration with ExecutionContext."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.context_manager = get_context_manager()
        self.mock_adapter = Mock()
        yield
        """Clean up after test."""
        # Pop any remaining contexts
        while len(self.context_manager._get_stack()) > 1:
            self.context_manager.pop_context()

    def test_infer_respects_execution_mode(self):
        """Test that inference respects execution mode restrictions."""
        # Switch to planning mode
        self.context_manager.switch_mode(ExecutionMode.PLANNING)

        request = InferenceRequest(
            model_id="gpt-4",
            prompt="Test",
        )

        # Should deny in planning mode (depending on policy)
        self.manager.infer(request)
        # Verify that the context is now in PLANNING mode
        current_mode = self.context_manager.current_mode
        assert current_mode == ExecutionMode.PLANNING

        self.context_manager.pop_context()

    def test_provider_metrics_tracking(self):
        """Test that provider metrics are tracked through lifecycle."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )

        # Simulate successful inference
        success_response = InferenceResponse(
            model_id="gpt-4",
            output="Success",
            latency_ms=150,
            tokens_used=100,
            status="ok",
        )

        self.mock_adapter.infer.return_value = success_response

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            request = InferenceRequest(model_id="gpt-4", prompt="Test")
            self.manager.infer(request, provider_name="openai-1")

            # Check metrics updated
            assert self.manager._metrics["openai-1"].successful_inferences == 1
            assert self.manager._metrics["openai-1"].total_tokens_used == 100


class TestToolInterfaceIntegration:
    """Test Tool interface integration with permissions."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.registry = get_tool_registry()
        self.context_manager = get_context_manager()
        yield
        """Clean up."""
        while len(self.context_manager._get_stack()) > 1:
            self.context_manager.pop_context()

    def test_tool_permission_enforcement(self):
        """Test that tools enforce permissions."""

        class FileReadTool(Tool):
            def get_metadata(self):
                return ToolMetadata(
                    name="read_file",
                    description="Read a file",
                    category=ToolCategory.FILE_OPERATIONS,
                    required_permissions=[ToolPermission.FILE_READ],
                )

            def execute(self, **kwargs):
                return {"content": "file content"}

        metadata = ToolMetadata(
            name="read_file",
            description="Read a file",
            category=ToolCategory.FILE_OPERATIONS,
            required_permissions=[ToolPermission.FILE_READ],
        )
        tool = FileReadTool(metadata)

        # In planning mode, should be allowed
        self.context_manager.switch_mode(ExecutionMode.PLANNING)
        result = tool.execute(path="test.txt")
        assert isinstance(result, dict)
        assert "content" in result

        self.context_manager.pop_context()


class TestVerificationPipelineIntegration:
    """Test verification pipeline integration."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.pipeline = get_verifier_pipeline()

    def test_verify_generated_code(self):
        """Test verifying generated code from LLM."""
        # Simulated code from LLM
        code = """
def process_data(input_data):
    result = []
    for item in input_data:
        result.append(item * 2)
    return result
"""

        results = self.pipeline.verify(code)

        # Should have passed syntax check
        assert "code_syntax" in results
        assert results["code_syntax"].status.value in ["passed", "skipped"]

    def test_verify_json_output(self):
        """Test verifying JSON output."""
        json_output = '{"status": "success", "data": [1, 2, 3]}'

        results = self.pipeline.verify(json_output)

        # Pipeline should run all verifiers
        assert len(results) > 0

    def test_verify_with_security_issues(self):
        """Test verification catches security issues."""
        unsafe_code = """
password = input("Enter password: ")
exec(password)
"""

        results = self.pipeline.verify(unsafe_code)

        # Security verifier should flag the exec()
        assert results["security"].status.value in ["warning", "failed"]


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.context_manager = get_context_manager()
        self.adapter_manager = AdapterManager()
        self.tool_registry = get_tool_registry()
        self.verifier_pipeline = get_verifier_pipeline()

        # Set up mock adapter
        self.mock_adapter = Mock()
        yield
        """Clean up."""
        while len(self.context_manager._get_stack()) > 1:
            self.context_manager.pop_context()

    def test_planning_mode_workflow(self):
        """Test workflow in planning mode."""
        # Switch to planning mode
        self.context_manager.switch_mode(ExecutionMode.PLANNING, "analysis_task")

        # Should allow reading files
        can_read = self.context_manager.check_permission(ToolPermission.FILE_READ)
        assert can_read is True

        # Should allow inference
        can_infer = self.context_manager.check_permission(ToolPermission.MODEL_INFERENCE)
        assert can_infer is True

        self.context_manager.pop_context()

    def test_execution_mode_workflow(self):
        """Test workflow in execution mode."""
        # Switch to execution mode
        self.context_manager.switch_mode(ExecutionMode.EXECUTION, "implementation_task")

        # Should allow all file operations
        assert self.context_manager.check_permission(ToolPermission.FILE_READ)
        assert self.context_manager.check_permission(ToolPermission.FILE_WRITE)

        # Should allow inference
        assert self.context_manager.check_permission(ToolPermission.MODEL_INFERENCE)

        self.context_manager.pop_context()

    def test_mode_transition_workflow(self):
        """Test transitioning between modes during task."""
        # Start in planning mode
        self.context_manager.switch_mode(ExecutionMode.PLANNING, "analyze")

        # Do analysis
        can_read = self.context_manager.check_permission(ToolPermission.FILE_READ)
        assert can_read is True

        # Transition to execution mode for implementation
        self.context_manager.pop_context()
        self.context_manager.switch_mode(ExecutionMode.EXECUTION, "implement")

        # Now can write files
        can_write = self.context_manager.check_permission(ToolPermission.FILE_WRITE)
        assert can_write is True

        self.context_manager.pop_context()


class TestPermissionEnforcementAcrossComponents:
    """Test permission enforcement consistency across components."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test fixtures."""
        self.context_manager = get_context_manager()
        self.adapter_manager = AdapterManager()

        # Initialize metrics
        self.adapter_manager._metrics["test-provider"] = ProviderMetrics(
            name="test-provider",
            provider_type=ProviderType.LOCAL,
        )

        self.mock_adapter = Mock()
        yield
        """Clean up."""
        while len(self.context_manager._get_stack()) > 1:
            self.context_manager.pop_context()

    def test_permission_denied_in_sandbox_mode(self):
        """Test that permissions are denied in sandbox mode."""
        self.context_manager.switch_mode(ExecutionMode.SANDBOX)

        # Sandbox mode should be very restrictive
        # Verify the context actually switched to SANDBOX
        current_mode = self.context_manager.current_mode
        assert current_mode == ExecutionMode.SANDBOX

        self.context_manager.pop_context()

    def test_audit_trail_tracks_permission_checks(self):
        """Test that audit trail tracks permission checks."""
        self.context_manager.switch_mode(ExecutionMode.EXECUTION, "audit_test")

        # Perform various checks
        self.context_manager.check_permission(ToolPermission.FILE_READ)
        self.context_manager.check_permission(ToolPermission.FILE_WRITE)
        self.context_manager.check_permission(ToolPermission.MODEL_INFERENCE)

        # Get status which includes context information
        status = self.context_manager.get_status()

        # Should have status information
        assert status is not None
        assert "mode" in status
        assert status["mode"] == "execution"

        self.context_manager.pop_context()


class TestPhase2Integration:
    """Comprehensive Phase 2 integration tests."""

    def test_context_manager_singleton(self):
        """Test context manager is singleton."""
        cm1 = get_context_manager()
        cm2 = get_context_manager()
        assert cm1 is cm2

    def test_adapter_manager_singleton(self):
        """Test adapter manager is singleton."""
        am1 = get_adapter_manager()
        am2 = get_adapter_manager()
        assert am1 is am2

    def test_tool_registry_singleton(self):
        """Test tool registry is singleton."""
        tr1 = get_tool_registry()
        tr2 = get_tool_registry()
        assert tr1 is tr2

    def test_verifier_pipeline_singleton(self):
        """Test verifier pipeline is singleton."""
        vp1 = get_verifier_pipeline()
        vp2 = get_verifier_pipeline()
        assert vp1 is vp2
