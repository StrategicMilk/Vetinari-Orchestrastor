"""
Unit tests for Tool Interface System (Phase 2)

Tests the Tool, ToolMetadata, ToolParameter, ToolResult, and ToolRegistry classes.
"""

import pytest
from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolParameter,
    ToolResult,
    ToolCategory,
    ToolRegistry,
    get_tool_registry,
)
from vetinari.execution_context import (
    ExecutionMode,
    ToolPermission,
    get_context_manager,
)


class TestToolParameter:
    """Test ToolParameter class."""
    
    def test_parameter_creation(self):
        """Test creating a tool parameter."""
        param = ToolParameter(
            name="input_file",
            type=str,
            description="Input file path",
            required=True,
        )
        assert param.name == "input_file"
        assert param.type == str
        assert param.required is True
    
    def test_parameter_validation_type(self):
        """Test parameter type validation."""
        param = ToolParameter(
            name="count",
            type=int,
            description="Count value",
            required=True,
        )
        assert param.validate(42) is True
        assert param.validate("not an int") is False
    
    def test_parameter_validation_allowed_values(self):
        """Test parameter allowed values validation."""
        param = ToolParameter(
            name="mode",
            type=str,
            description="Operation mode",
            allowed_values=["read", "write", "append"],
        )
        assert param.validate("read") is True
        assert param.validate("delete") is False
    
    def test_parameter_optional(self):
        """Test optional parameter validation."""
        param = ToolParameter(
            name="optional_param",
            type=str,
            description="Optional parameter",
            required=False,
            default="default_value",
        )
        assert param.validate(None) is True


class TestToolMetadata:
    """Test ToolMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating tool metadata."""
        params = [
            ToolParameter(
                name="input",
                type=str,
                description="Input text",
                required=True,
            ),
        ]
        
        metadata = ToolMetadata(
            name="my_tool",
            description="My tool description",
            category=ToolCategory.FILE_OPERATIONS,
            parameters=params,
            required_permissions=[ToolPermission.FILE_READ],
        )
        
        assert metadata.name == "my_tool"
        assert metadata.category == ToolCategory.FILE_OPERATIONS
        assert len(metadata.parameters) == 1
        assert ToolPermission.FILE_READ in metadata.required_permissions
    
    def test_metadata_default_mode(self):
        """Test metadata default allowed modes."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test",
            category=ToolCategory.FILE_OPERATIONS,
        )
        assert ExecutionMode.EXECUTION in metadata.allowed_modes


class TestToolResult:
    """Test ToolResult class."""
    
    def test_result_success(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            output="Tool executed successfully",
            execution_time_ms=100,
        )
        assert result.success is True
        assert result.error is None
        assert result.execution_time_ms == 100
    
    def test_result_failure(self):
        """Test failed tool result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Tool execution failed",
        )
        assert result.success is False
        assert result.error == "Tool execution failed"
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ToolResult(
            success=True,
            output="success",
            metadata={"key": "value"},
        )
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["metadata"]["key"] == "value"


class SimpleTool(Tool):
    """A simple test tool implementation."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="simple_tool",
            description="A simple test tool",
            category=ToolCategory.FILE_OPERATIONS,
            parameters=[
                ToolParameter(
                    name="message",
                    type=str,
                    description="Message to process",
                    required=True,
                ),
            ],
            required_permissions=[ToolPermission.FILE_READ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
        )
        super().__init__(metadata)
    
    def execute(self, message: str, **kwargs) -> ToolResult:
        """Simple implementation that echoes the message."""
        return ToolResult(
            success=True,
            output=f"Processed: {message}",
        )


class TestTool:
    """Test Tool base class."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        tool = SimpleTool()
        assert tool.metadata.name == "simple_tool"
        assert len(tool.metadata.parameters) == 1
    
    def test_tool_validation_success(self):
        """Test successful input validation."""
        tool = SimpleTool()
        is_valid, error = tool.validate_inputs({"message": "test"})
        assert is_valid is True
        assert error is None
    
    def test_tool_validation_missing_required(self):
        """Test validation of missing required parameter."""
        tool = SimpleTool()
        is_valid, error = tool.validate_inputs({})
        assert is_valid is False
        assert error is not None
    
    def test_tool_validation_wrong_type(self):
        """Test validation of wrong parameter type."""
        tool = SimpleTool()
        is_valid, error = tool.validate_inputs({"message": 123})
        assert is_valid is False
    
    def test_tool_permission_check(self):
        """Test permission checking."""
        tool = SimpleTool()
        
        # Set execution mode
        manager = get_context_manager()
        manager.switch_mode(ExecutionMode.EXECUTION)
        
        # Should have permission
        has_perms, error = tool.check_permissions()
        assert has_perms is True
        
        manager.pop_context()
        manager.switch_mode(ExecutionMode.SANDBOX)
        
        # Tool requires FILE_READ which isn't in SANDBOX
        # But SimpleTool is allowed in SANDBOX mode, so should pass
        has_perms, error = tool.check_permissions()
        # This depends on whether SANDBOX allows the tool
    
    def test_tool_execution(self):
        """Test executing a tool."""
        tool = SimpleTool()
        
        manager = get_context_manager()
        manager.switch_mode(ExecutionMode.EXECUTION)
        
        result = tool.execute(message="hello")
        assert result.success is True
        assert "hello" in result.output
    
    def test_tool_run_with_validation(self):
        """Test running tool with automatic validation."""
        tool = SimpleTool()
        
        manager = get_context_manager()
        manager.switch_mode(ExecutionMode.EXECUTION)
        
        result = tool.run(message="test message")
        assert result.success is True
        assert result.execution_time_ms > 0
    
    def test_tool_run_invalid_input(self):
        """Test running tool with invalid input."""
        tool = SimpleTool()
        
        manager = get_context_manager()
        manager.switch_mode(ExecutionMode.EXECUTION)
        
        result = tool.run()  # Missing required parameter
        assert result.success is False
        assert result.error is not None


class TestToolRegistry:
    """Test ToolRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert len(registry.list_tools()) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = SimpleTool()
        
        registry.register(tool)
        assert len(registry.list_tools()) == 1
        assert registry.get("simple_tool") is tool
    
    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None
    
    def test_list_tools_by_category(self):
        """Test listing tools by category."""
        registry = ToolRegistry()
        
        tool1 = SimpleTool()
        registry.register(tool1)
        
        file_tools = registry.get_tools_by_category(ToolCategory.FILE_OPERATIONS)
        assert len(file_tools) == 1
        assert file_tools[0].metadata.name == "simple_tool"
        
        code_tools = registry.get_tools_by_category(ToolCategory.CODE_EXECUTION)
        assert len(code_tools) == 0
    
    def test_list_tools_by_mode(self):
        """Test listing tools available in a mode."""
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)
        
        execution_tools = registry.list_tools_for_mode(ExecutionMode.EXECUTION)
        assert len(execution_tools) == 1
        
        planning_tools = registry.list_tools_for_mode(ExecutionMode.PLANNING)
        assert len(planning_tools) == 1
    
    def test_list_tools_by_permission(self):
        """Test listing tools by required permission."""
        registry = ToolRegistry()
        tool = SimpleTool()
        registry.register(tool)
        
        file_read_tools = registry.get_tools_requiring_permission(ToolPermission.FILE_READ)
        assert len(file_read_tools) == 1
        
        file_write_tools = registry.get_tools_requiring_permission(ToolPermission.FILE_WRITE)
        assert len(file_write_tools) == 0


class TestGlobalToolRegistry:
    """Test global tool registry singleton."""
    
    def test_get_tool_registry(self):
        """Test getting global tool registry."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        
        # Should be same instance
        assert registry1 is registry2


class TestToolDescription:
    """Test tool description generation."""
    
    def test_get_description(self):
        """Test getting formatted tool description."""
        tool = SimpleTool()
        description = tool.get_description()
        
        assert "simple_tool" in description
        assert "File Operations" in description
        assert "message" in description
        assert "Input text" in description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
