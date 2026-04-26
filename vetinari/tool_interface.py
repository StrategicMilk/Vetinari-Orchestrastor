"""Tool Interface Module for Vetinari.

Defines a standardized interface for all tools/skills, inspired by OpenCode's
tool execution system.

Features:
- Abstract Tool base class with clear contracts
- Tool metadata (name, description, permissions required)
- Input/output validation
- Safety checks and pre-execution hooks
- Permission enforcement
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_origin

from vetinari.execution_context import (
    AGENT_PERMISSION_MAP,
    ExecutionMode,
    ToolPermission,
    check_permission_unified,
    get_context_manager,
)
from vetinari.types import AgentType
from vetinari.utils.registry import BaseRegistry
from vetinari.utils.serialization import dataclass_to_dict

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
    type: type
    description: str
    required: bool = True
    default: Any | None = None
    allowed_values: list[Any] | None = None

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ToolParameter(name={self.name!r}, required={self.required!r})"

    def validate(self, value: Any) -> bool:
        """Validate a value against this parameter definition.

        Handles both plain types and parameterized generic aliases (e.g.
        ``list[str]``, ``dict[str, int]``) by extracting the origin type
        via :func:`typing.get_origin` before calling ``isinstance``.  A bare
        ``isinstance(value, list[str])`` raises ``TypeError`` on Python 3.9+.

        Returns:
            True if the value is valid for this parameter, False otherwise.
        """
        if value is None:
            return not self.required

        # get_origin returns the un-parameterised base (e.g. list for list[str]).
        # Falls back to self.type when it is already a plain type (get_origin → None).
        origin = get_origin(self.type)
        check_type = origin if origin is not None else self.type
        if not isinstance(value, check_type):
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
    parameters: list[ToolParameter] = field(default_factory=list)
    required_permissions: list[ToolPermission] = field(default_factory=list)
    allowed_modes: list[ExecutionMode] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ToolMetadata(name={self.name!r}, category={self.category!r}, version={self.version!r})"

    def __post_init__(self):
        """Set default allowed modes if not provided."""
        if not self.allowed_modes:
            self.allowed_modes = [ExecutionMode.EXECUTION]


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ToolResult(success={self.success!r}, execution_time_ms={self.execution_time_ms!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


class Tool(ABC):
    """Abstract base class for all Vetinari tools.

    Tools are the building blocks of skills and represent individual
    executable operations that can be performed by LLM agents.
    """

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self._context_manager = get_context_manager()

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Parameters matching the tool's metadata

        Returns:
            ToolResult with success status, output, and metadata
        """

    def validate_inputs(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate input parameters against tool metadata.

        Args:
            params: Parameters to validate

        Returns:
            (is_valid, error_message)
        """
        for tool_param in self.metadata.parameters:
            if tool_param.required and tool_param.name not in params:
                return False, f"Missing required parameter: {tool_param.name}"

            if tool_param.name in params and not tool_param.validate(params[tool_param.name]):
                return False, f"Invalid value for parameter {tool_param.name}"

        return True, None

    def check_permissions(self) -> tuple[bool, str | None]:
        """Check if current execution context allows this tool.

        Returns:
            (is_allowed, error_message)
        """
        ctx_manager = get_context_manager()
        current_mode = ctx_manager.current_mode

        # Check if tool is allowed in current mode
        if current_mode not in self.metadata.allowed_modes:
            return False, (f"Tool '{self.metadata.name}' is not allowed in {current_mode.value} mode")

        # Check all required permissions
        for permission in self.metadata.required_permissions:
            if not ctx_manager.check_permission(permission):
                return False, (
                    f"Tool '{self.metadata.name}' requires {permission.value} "
                    f"permission, which is not available in {current_mode.value} mode"
                )

        return True, None

    def run(self, agent_type: AgentType | None = None, **kwargs) -> ToolResult:
        """Run the tool with safety checks and validation.

        This is the main entry point that enforces mode permissions, optional
        agent-level permissions, confirmation requirements, and input validation
        before delegating to :meth:`execute`.

        Args:
            agent_type: Optional agent type for agent-level permission enforcement.
                When provided, the tool is checked via ``check_permission_unified``
                against both the mode policy and the agent permission map —
                the most-restrictive policy wins.
            **kwargs: Parameters forwarded to ``execute()``.

        Returns:
            ToolResult with execution status and output. Exceptions from
            hooks or the audit trail are caught and logged; they never
            surface as raw exceptions to the caller.
        """
        import time

        start_time = time.time()

        # -- Bug 4: inject defaults for missing optional parameters before validation --
        for tool_param in self.metadata.parameters:
            if not tool_param.required and tool_param.name not in kwargs and tool_param.default is not None:
                kwargs[tool_param.name] = tool_param.default

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

        # Check mode-level permissions
        has_perms, error_msg = self.check_permissions()
        if not has_perms:
            logger.warning("Permission check failed for %s: %s", self.metadata.name, error_msg)
            return ToolResult(
                success=False,
                output=None,
                error=error_msg,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        # -- Bug 2: unified agent-level permission check (most-restrictive-wins) --
        # check_permission_unified covers mode policy, AGENT_PERMISSION_MAP, and
        # PolicyEnforcer in one call — replacing the raw enforce_agent_permissions loop.
        if agent_type is not None:
            for permission in self.metadata.required_permissions:
                if not check_permission_unified(agent_type, permission):
                    logger.warning(
                        "Agent permission denied for tool '%s' (agent=%s, permission=%s)",
                        self.metadata.name,
                        agent_type.value,
                        permission.value,
                    )
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Permission {permission.value} denied for agent {agent_type.value}",
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )

        # Check if confirmation is required — fail closed: block execution when
        # confirmation is needed rather than silently proceeding.
        ctx = self._context_manager.current_context
        for permission in self.metadata.required_permissions:
            if ctx.requires_confirmation(permission):
                logger.warning(
                    "Tool '%s' requires confirmation for %s — blocking execution (fail-closed)",
                    self.metadata.name,
                    permission.value,
                )
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Operation requires confirmation for {permission.value} — not auto-approved",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

        # -- Bug 5: filter kwargs to only declared parameter names before execute() --
        # Undeclared keys are dropped so execute() never receives unexpected arguments.
        # If no parameters are declared, pass all kwargs through (tool accepts **kwargs).
        declared_names = {p.name for p in self.metadata.parameters}
        if declared_names:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in declared_names}
        else:
            filtered_kwargs = kwargs

        # -- Bug 6: wrap pre-execution hooks — a failing hook returns a bounded result --
        for hook in ctx.pre_execution_hooks:
            try:
                should_proceed = hook(self.metadata.name, filtered_kwargs)
            except Exception as hook_exc:
                logger.warning(
                    "Pre-execution hook raised an error for tool '%s' — blocking as fail-closed: %s",
                    self.metadata.name,
                    hook_exc,
                )
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Pre-execution hook error: {hook_exc!s}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )
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
            result = self.execute(**filtered_kwargs)
            result.execution_time_ms = max(1, int((time.time() - start_time) * 1000))
        except Exception as e:
            logger.error("Error executing tool %s: %s", self.metadata.name, e)
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {e!s}",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        # -- Bug 7: post-execution hooks and audit trail are best-effort —
        # failures are logged but the primary result is always returned.
        for hook in ctx.post_execution_hooks:
            try:
                hook(self.metadata.name, filtered_kwargs, result)
            except Exception as hook_exc:
                logger.warning(
                    "Post-execution hook raised an error for tool '%s' — result preserved, hook skipped: %s",
                    self.metadata.name,
                    hook_exc,
                )

        try:
            ctx.record_operation(
                self.metadata.name,
                filtered_kwargs,
                result.to_dict(),
            )
        except Exception as audit_exc:
            logger.warning(
                "Audit trail recording failed for tool '%s' — result preserved, audit skipped: %s",
                self.metadata.name,
                audit_exc,
            )

        return result

    def get_description(self) -> str:
        """Build a formatted multi-line description of the tool for display.

        Returns:
            Human-readable string listing the tool name, description, category,
            version, allowed execution modes, required permissions, and each
            parameter with its type and required/optional status.
        """
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
            lines.append(f"  - {param.name} ({param.type.__name__}, {req}): {param.description}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Tool({self.metadata.name} v{self.metadata.version})"


class ToolRegistry(BaseRegistry[str, Tool]):
    """Registry for managing available tools."""

    def register(self, tool: Tool) -> None:  # type: ignore[override]
        """Register a tool, keying it on its metadata name.

        Args:
            tool: The Tool instance to register.
        """
        super().register(tool.metadata.name, tool)
        logger.info("Registered tool: %s", tool.metadata.name)

    def list_tools(self) -> list[Tool]:
        """Return a snapshot of all registered tools.

        Returns:
            List of all Tool instances currently in the registry.
        """
        return self.list_all()

    def list_tools_for_mode(self, mode: ExecutionMode, agent_type: AgentType | None = None) -> list[Tool]:
        """List tools available in a specific execution mode, optionally filtered by agent type.

        When ``agent_type`` is provided, tools are also filtered by the
        agent-level permission map so that only tools whose required permissions
        are all granted to that agent type are returned.  This applies the
        most-restrictive-wins policy at the listing stage.

        Args:
            mode: The ExecutionMode to filter by.
            agent_type: Optional agent type for additional permission filtering.

        Returns:
            Tools whose allowed_modes include the given mode, further filtered
            by agent-level permissions when agent_type is provided.
        """
        tools = [tool for tool in self.list_all() if mode in tool.metadata.allowed_modes]
        if agent_type is not None:
            agent_perms = AGENT_PERMISSION_MAP.get(agent_type, frozenset())
            tools = [t for t in tools if all(p in agent_perms for p in t.metadata.required_permissions)]
        return tools

    def get_tools_by_category(self, category: ToolCategory) -> list[Tool]:
        """Get tools belonging to a specific category.

        Args:
            category: The ToolCategory to filter by.

        Returns:
            Tools whose category matches the given value.
        """
        return [tool for tool in self.list_all() if tool.metadata.category == category]

    def get_tools_requiring_permission(self, permission: ToolPermission) -> list[Tool]:
        """Get tools that require a specific permission.

        Args:
            permission: The ToolPermission to filter by.

        Returns:
            Tools that list the given permission as required.
        """
        return [tool for tool in self.list_all() if permission in tool.metadata.required_permissions]


# Global tool registry
_tool_registry: ToolRegistry | None = None
_tool_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry.

    Returns:
        The singleton ToolRegistry used by all subsystems to register and
        look up tools.
    """
    global _tool_registry
    if _tool_registry is None:
        with _tool_registry_lock:
            if _tool_registry is None:
                _tool_registry = ToolRegistry()
    return _tool_registry
