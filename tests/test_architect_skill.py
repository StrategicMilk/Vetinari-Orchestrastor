"""
Unit tests for the unified Architect Skill Tool.

Tests cover:
- Tool initialization and metadata
- All 6 architect modes (ui_design, database, devops, git_workflow, system_design, api_design)
- Parameter validation
- Thinking mode depth variation
- Dataclass serialization
- Error cases
"""

from unittest.mock import Mock

import pytest

from vetinari.execution_context import (
    ExecutionContext,
    ExecutionMode,
    ToolPermission,
)
from vetinari.skills.architect_skill import (
    ArchitectComponent,
    ArchitectMode,
    ArchitectRequest,
    ArchitectResult,
    ArchitectSkillTool,
    ThinkingMode,
)


class TestArchitectSkillToolMetadata:
    """Tests for architect skill metadata and initialization."""

    def test_initialization(self):
        """Test ArchitectSkillTool initializes with correct metadata."""
        tool = ArchitectSkillTool()

        assert tool.metadata.name == "architect"
        assert "architecture" in tool.metadata.tags
        assert "design" in tool.metadata.tags
        assert "api" in tool.metadata.tags
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
        assert ToolPermission.MODEL_INFERENCE in tool.metadata.required_permissions

    def test_allowed_execution_modes(self):
        """Test allowed execution modes include EXECUTION and PLANNING."""
        tool = ArchitectSkillTool()

        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes

    def test_tool_parameters(self):
        """Test tool parameters are properly defined."""
        tool = ArchitectSkillTool()

        param_names = {p.name for p in tool.metadata.parameters}
        assert "mode" in param_names
        assert "design_request" in param_names
        assert "domain" in param_names
        assert "context" in param_names
        assert "thinking_mode" in param_names
        assert "constraints" in param_names

    def test_mode_parameter_validation(self):
        """Test mode parameter has correct allowed values."""
        tool = ArchitectSkillTool()

        mode_param = next(p for p in tool.metadata.parameters if p.name == "mode")

        assert mode_param.required is True
        assert all(m.value in mode_param.allowed_values for m in ArchitectMode)

    def test_thinking_mode_parameter_defaults(self):
        """Test thinking_mode parameter defaults to medium."""
        tool = ArchitectSkillTool()

        thinking_param = next(p for p in tool.metadata.parameters if p.name == "thinking_mode")

        assert thinking_param.required is False
        assert thinking_param.default == "medium"
        assert all(m.value in thinking_param.allowed_values for m in ThinkingMode)

    def test_version(self):
        """Test tool version is set correctly."""
        tool = ArchitectSkillTool()
        assert tool.metadata.version == "1.1.0"


class TestArchitectSkillToolExecution:
    """Tests for architect skill execution logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ArchitectSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_ui_design(self):
        """Test UI design mode."""
        result = self.tool.execute(
            mode="ui_design",
            design_request="Design a dashboard layout",
        )

        assert result.success is True
        output = result.output
        assert output["design"]["architecture_pattern"] is not None
        assert output["design"]["rationale"] is not None
        assert len(output["components"]) > 0
        assert len(output["design"]["alternatives_considered"]) > 0

    def test_database_design(self):
        """Test database schema design mode."""
        result = self.tool.execute(
            mode="database",
            design_request="Design user management schema",
        )

        assert result.success is True
        output = result.output
        assert "Database Design" in output["design"]["summary"]
        assert output["design"]["architecture_pattern"] is not None
        assert any("index" in c["name"].lower() or "table" in c["name"].lower() for c in output["components"])

    def test_database_design_warnings(self):
        """Test database design includes data-loss warnings."""
        result = self.tool.execute(
            mode="database",
            design_request="Migrate user table schema",
        )

        assert result.success is True
        assert len(result.output["warnings"]) > 0

    def test_devops_design(self):
        """Test DevOps pipeline design mode."""
        result = self.tool.execute(
            mode="devops",
            design_request="CI/CD pipeline for microservice",
        )

        assert result.success is True
        output = result.output
        assert "DevOps Pipeline" in output["design"]["summary"]
        assert len(output["migration_plan"]["steps"]) > 0
        # STD-ARC-005: rollback and health checks
        component_names = [c["name"].lower() for c in output["components"]]
        assert any("ci" in n or "cd" in n or "monitor" in n for n in component_names)

    def test_git_workflow_design(self):
        """Test git workflow design mode."""
        result = self.tool.execute(
            mode="git_workflow",
            design_request="Branch strategy for team of 5",
        )

        assert result.success is True
        output = result.output
        assert "Git Workflow" in output["design"]["summary"]
        assert output["design"]["architecture_pattern"] is not None
        assert len(output["design"]["alternatives_considered"]) > 0

    def test_system_design(self):
        """Test system architecture design mode."""
        result = self.tool.execute(
            mode="system_design",
            design_request="Design modular backend",
        )

        assert result.success is True
        output = result.output
        assert "System Design" in output["design"]["summary"]
        assert len(output["components"]) >= 3
        # Check components have dependencies
        has_deps = any(len(c.get("dependencies", [])) > 0 for c in output["components"])
        assert has_deps

    def test_api_design(self):
        """Test API architecture design mode."""
        result = self.tool.execute(
            mode="api_design",
            design_request="RESTful API for user service",
        )

        assert result.success is True
        output = result.output
        assert "API Design" in output["design"]["summary"]
        # STD-ARC-004: auth, authz, rate limiting
        component_names = [c["name"].lower() for c in output["components"]]
        assert any("auth" in n for n in component_names)
        assert any("rate" in n for n in component_names)

    def test_invalid_mode(self):
        """Test execution with invalid mode returns error."""
        result = self.tool.execute(
            mode="invalid_mode",
            design_request="Something",
        )

        assert result.success is False
        assert result.output is None
        assert "Invalid mode" in result.error

    def test_missing_design_request(self):
        """Test execution without design_request returns error."""
        result = self.tool.execute(
            mode="system_design",
        )

        assert result.success is False
        assert "design_request" in result.error

    def test_invalid_thinking_mode(self):
        """Test execution with invalid thinking mode returns error."""
        result = self.tool.execute(
            mode="system_design",
            design_request="Design something",
            thinking_mode="invalid",
        )

        assert result.success is False
        assert "thinking_mode" in result.error

    def test_optional_domain_parameter(self):
        """Test domain parameter is passed correctly."""
        result = self.tool.execute(
            mode="system_design",
            design_request="Design backend",
            domain="web",
        )

        assert result.success is True

    def test_optional_constraints_parameter(self):
        """Test constraints parameter is accepted."""
        result = self.tool.execute(
            mode="api_design",
            design_request="Design API",
            constraints=["no GraphQL", "REST only"],
        )

        assert result.success is True

    def test_metadata_in_result(self):
        """Test that metadata is included in ToolResult."""
        result = self.tool.execute(
            mode="devops",
            design_request="CI pipeline",
            thinking_mode="high",
        )

        assert result.metadata["mode"] == "devops"
        assert result.metadata["thinking_mode"] == "high"
        assert "components_count" in result.metadata


class TestArchitectModeEnum:
    """Tests for ArchitectMode enum."""

    def test_all_modes_have_values(self):
        """Test all modes have valid string values."""
        for mode in ArchitectMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_expected_modes_exist(self):
        """Test expected modes are defined."""
        expected = {"ui_design", "database", "devops", "git_workflow", "system_design", "api_design"}
        actual = {m.value for m in ArchitectMode}
        assert actual == expected

    def test_mode_count(self):
        """Test correct number of modes."""
        assert len(ArchitectMode) == 6


class TestArchitectDataclasses:
    """Tests for architect dataclasses."""

    def test_request_defaults(self):
        """Test ArchitectRequest default values."""
        req = ArchitectRequest(
            mode=ArchitectMode.SYSTEM_DESIGN,
            design_request="Design something",
        )

        assert req.domain is None
        assert req.context is None
        assert req.thinking_mode == ThinkingMode.MEDIUM
        assert req.constraints == []

    def test_request_full(self):
        """Test ArchitectRequest with all fields."""
        req = ArchitectRequest(
            mode=ArchitectMode.API_DESIGN,
            design_request="Design REST API",
            domain="backend",
            context="existing Flask app",
            thinking_mode=ThinkingMode.HIGH,
            constraints=["no breaking changes"],
        )

        assert req.mode == ArchitectMode.API_DESIGN
        assert req.domain == "backend"
        assert req.constraints == ["no breaking changes"]

    def test_request_to_dict(self):
        """Test ArchitectRequest serialization."""
        req = ArchitectRequest(
            mode=ArchitectMode.DATABASE,
            design_request="Schema design",
            thinking_mode=ThinkingMode.XHIGH,
        )

        d = req.to_dict()
        assert d["mode"] == "database"
        assert d["design_request"] == "Schema design"
        assert d["thinking_mode"] == "xhigh"
        assert d["constraints"] == []

    def test_component_to_dict(self):
        """Test ArchitectComponent serialization."""
        comp = ArchitectComponent(
            name="API Gateway",
            responsibility="Route requests",
            interfaces=["REST", "gRPC"],
            dependencies=["Auth Service"],
        )

        d = comp.to_dict()
        assert d["name"] == "API Gateway"
        assert d["responsibility"] == "Route requests"
        assert d["interfaces"] == ["REST", "gRPC"]
        assert d["dependencies"] == ["Auth Service"]

    def test_component_defaults(self):
        """Test ArchitectComponent default values."""
        comp = ArchitectComponent(name="Cache", responsibility="Cache data")

        assert comp.interfaces == []
        assert comp.dependencies == []

    def test_result_success(self):
        """Test ArchitectResult success case."""
        result = ArchitectResult(
            success=True,
            summary="Design complete",
            architecture_pattern="Microservices",
            rationale="Scalability",
            components=[
                ArchitectComponent(name="Service A", responsibility="Handle users"),
            ],
            alternatives_considered=["Monolith"],
        )

        assert result.success is True
        assert len(result.components) == 1
        assert result.warnings == []
        assert result.migration_steps == []

    def test_result_to_dict(self):
        """Test ArchitectResult serialization."""
        result = ArchitectResult(
            success=True,
            summary="Done",
            architecture_pattern="Layered",
            rationale="Simplicity",
            components=[],
            alternatives_considered=["Event-driven"],
            warnings=["Review needed"],
            migration_steps=["Step 1", "Step 2"],
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["design"]["summary"] == "Done"
        assert d["design"]["architecture_pattern"] == "Layered"
        assert d["design"]["rationale"] == "Simplicity"
        assert d["design"]["alternatives_considered"] == ["Event-driven"]
        assert d["warnings"] == ["Review needed"]
        assert d["migration_plan"]["steps"] == ["Step 1", "Step 2"]


class TestArchitectSkillToolParameterValidation:
    """Tests for input parameter validation."""

    def test_valid_parameters(self):
        """Test valid parameters pass validation."""
        tool = ArchitectSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "system_design",
            "design_request": "Design a system",
        })

        assert is_valid is True
        assert error is None

    def test_missing_required_mode(self):
        """Test missing mode is caught."""
        tool = ArchitectSkillTool()

        is_valid, error = tool.validate_inputs({
            "design_request": "Design something",
        })

        assert is_valid is False
        assert "mode" in error.lower()

    def test_missing_required_design_request(self):
        """Test missing design_request is caught."""
        tool = ArchitectSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "system_design",
        })

        assert is_valid is False
        assert "design_request" in error.lower()

    def test_invalid_mode_value(self):
        """Test invalid mode value is caught."""
        tool = ArchitectSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "nonexistent_mode",
            "design_request": "test",
        })

        assert is_valid is False
        assert "mode" in error.lower()

    def test_optional_params_omitted(self):
        """Test optional parameters can be omitted."""
        tool = ArchitectSkillTool()

        is_valid, error = tool.validate_inputs({
            "mode": "api_design",
            "design_request": "Design API",
        })

        assert is_valid is True
        assert error is None


class TestArchitectSkillToolEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ArchitectSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_context.execution_mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        self.tool._context_manager = self.mock_ctx_manager

    def test_empty_design_request(self):
        """Test with empty design_request returns error."""
        result = self.tool.execute(
            mode="system_design",
            design_request="",
        )

        # Empty string should fail the required check
        assert result.success is False

    def test_very_long_design_request(self):
        """Test with very long design request."""
        result = self.tool.execute(
            mode="system_design",
            design_request="A" * 10000,
        )

        assert result.success is True

    def test_special_characters(self):
        """Test with special characters in design_request."""
        result = self.tool.execute(
            mode="api_design",
            design_request='Design <API> with "quotes" & special chars',
        )

        assert result.success is True

    def test_all_modes_execute_successfully(self):
        """Test that all modes execute without errors."""
        for mode in ArchitectMode:
            result = self.tool.execute(
                mode=mode.value,
                design_request=f"Test design for {mode.value}",
            )
            assert result.success is True, f"Mode {mode.value} failed"

    def test_all_thinking_modes_accepted(self):
        """Test all thinking modes are accepted."""
        for tm in ThinkingMode:
            result = self.tool.execute(
                mode="system_design",
                design_request="Test",
                thinking_mode=tm.value,
            )
            assert result.success is True, f"ThinkingMode {tm.value} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
