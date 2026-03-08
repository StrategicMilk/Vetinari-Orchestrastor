"""
Tool Interface Module for Vetinari

Defines a standardized interface for all tools/skills, inspired by OpenCode's
tool execution system.

Features:
- Abstract Tool base class with clear contracts
- Tool metadata (name, description, permissions required)
- Input/output validation
- Safety checks and pre-execution hooks
- Permission enforcement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from enum import Enum
import logging
from datetime import datetime

from vetinari.execution_context import (
    ToolPermission,
    ExecutionMode,
    get_context_manager,
)

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Category of tool for organizational purposes."""
    FILE_OPERATIONS = "file_operations"
    CODE_EXECUTION = "code_execution"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_OPERATIONS = "system_operations"
    GIT_OPERATIONS = "git_operations"
    SEARCH_ANALYSIS = "search_analysis"
    DATA_PROCESSING = "data_processing"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this parameter definition."""
        if value is None:
            return not self.required
        
        if not isinstance(value, self.type):
            return False
        
        if self.allowed_values is not None:
            return value in self.allowed_values
        
        return True


@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "Vetinari"
    parameters: List[ToolParameter] = field(default_factory=list)
    required_permissions: List[ToolPermission] = field(default_factory=list)
    allowed_modes: List[ExecutionMode] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default allowed modes if not provided."""
        if not self.allowed_modes:
            self.allowed_modes = [ExecutionMode.EXECUTION]


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """
    Abstract base class for all Vetinari tools.
    
    Tools are the building blocks of skills and represent individual
    executable operations that can be performed by LLM agents.
    """
    
    def __init__(self, metadata: ToolMetadata):
        """Initialize the tool with metadata."""
        self.metadata = metadata
        self._context_manager = get_context_manager()
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Parameters matching the tool's metadata
            
        Returns:
            ToolResult with success status, output, and metadata
        """
        pass
    
    def validate_inputs(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate input parameters against tool metadata.
        
        Args:
            params: Parameters to validate
            
        Returns:
            (is_valid, error_message)
        """
        for tool_param in self.metadata.parameters:
            if tool_param.required and tool_param.name not in params:
                return False, f"Missing required parameter: {tool_param.name}"
            
            if tool_param.name in params:
                if not tool_param.validate(params[tool_param.name]):
                    return False, f"Invalid value for parameter {tool_param.name}"
        
        return True, None
    
    def check_permissions(self) -> tuple[bool, Optional[str]]:
        """
        Check if current execution context allows this tool.
        
        Returns:
            (is_allowed, error_message)
        """
        ctx_manager = get_context_manager()
        current_mode = ctx_manager.current_mode
        
        # Check if tool is allowed in current mode
        if current_mode not in self.metadata.allowed_modes:
            return False, (
                f"Tool '{self.metadata.name}' is not allowed in "
                f"{current_mode.value} mode"
            )
        
        # Check all required permissions
        for permission in self.metadata.required_permissions:
            if not ctx_manager.check_permission(permission):
                return False, (
                    f"Tool '{self.metadata.name}' requires {permission.value} "
                    f"permission, which is not available in {current_mode.value} mode"
                )
        
        return True, None
    
    def run(self, **kwargs) -> ToolResult:
        """
        Run the tool with safety checks and validation.
        
        This is the main entry point that enforces permissions and validates inputs.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            ToolResult with execution status and output
        """
        import time
        start_time = time.time()
        
        # Validate inputs
        is_valid, error_msg = self.validate_inputs(kwargs)
        if not is_valid:
            logger.warning("Input validation failed for %s: %s", self.metadata.name, error_msg)
            return ToolResult(
                success=False,
                output=None,
                error=error_msg,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Check permissions
        has_perms, error_msg = self.check_permissions()
        if not has_perms:
            logger.warning("Permission check failed for %s: %s", self.metadata.name, error_msg)
            return ToolResult(
                success=False,
                output=None,
                error=error_msg,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Check if confirmation is required
        ctx = self._context_manager.current_context
        for permission in self.metadata.required_permissions:
            if ctx.requires_confirmation(permission):
                logger.info("Tool '%s' requires confirmation for %s", self.metadata.name, permission.value)
                # In a real CLI, this would prompt the user
                # For now, we just log it
        
        # Execute pre-execution hooks
        for hook in ctx.pre_execution_hooks:
            should_proceed = hook(self.metadata.name, kwargs)
            if not should_proceed:
                logger.info("Pre-execution hook blocked %s", self.metadata.name)
                return ToolResult(
                    success=False,
                    output=None,
                    error="Pre-execution hook blocked execution",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )
        
        # Execute the tool
        try:
            logger.info("Executing tool: %s", self.metadata.name)
            result = self.execute(**kwargs)
            result.execution_time_ms = max(1, int((time.time() - start_time) * 1000))
            
            # Execute post-execution hooks
            for hook in ctx.post_execution_hooks:
                hook(self.metadata.name, kwargs, result)
            
            # Record in audit trail
            ctx.record_operation(
                self.metadata.name,
                kwargs,
                result.to_dict(),
            )
            
            return result
        except Exception as e:
            logger.error("Error executing tool %s: %s", self.metadata.name, e)
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
    
    def get_description(self) -> str:
        """Get a formatted description of the tool."""
        lines = [
            f"Tool: {self.metadata.name}",
            f"Description: {self.metadata.description}",
            f"Category: {self.metadata.category.value.replace('_', ' ').title()}",
            f"Version: {self.metadata.version}",
            f"Allowed Modes: {', '.join(m.value for m in self.metadata.allowed_modes)}",
            f"Required Permissions: {', '.join(p.value for p in self.metadata.required_permissions)}",
            "",
            "Parameters:",
        ]
        
        for param in self.metadata.parameters:
            req = "required" if param.required else "optional"
            lines.append(
                f"  - {param.name} ({param.type.__name__}, {req}): {param.description}"
            )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Tool({self.metadata.name} v{self.metadata.version})"


class ToolRegistry:
    """
    Registry for managing available tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.metadata.name] = tool
        logger.info("Registered tool: %s", tool.metadata.name)
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def list_tools_for_mode(self, mode: ExecutionMode) -> List[Tool]:
        """List tools available in a specific execution mode."""
        return [
            tool for tool in self._tools.values()
            if mode in tool.metadata.allowed_modes
        ]
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools by category."""
        return [
            tool for tool in self._tools.values()
            if tool.metadata.category == category
        ]
    
    def get_tools_requiring_permission(self, permission: ToolPermission) -> List[Tool]:
        """Get tools that require a specific permission."""
        return [
            tool for tool in self._tools.values()
            if permission in tool.metadata.required_permissions
        ]


# Global tool registry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
