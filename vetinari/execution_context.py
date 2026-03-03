"""
Execution Context Module for Vetinari

Inspired by OpenCode's agent-based approach, this module implements:
- ExecutionMode (Planning/Read-only vs Execution/Write)
- Context-aware safety checks
- Tool permission enforcement
- Pre/post-execution hooks

This allows Vetinari to operate in different modes with varying levels of access,
similar to OpenCode's 'plan' vs 'build' agents.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """
    Execution modes available in Vetinari.
    
    PLANNING: Read-only mode for analysis and planning
    EXECUTION: Full read/write mode for implementation
    SANDBOX: Restricted mode for untrusted code
    """
    PLANNING = "planning"
    EXECUTION = "execution"
    SANDBOX = "sandbox"


class ToolPermission(Enum):
    """
    Tool permissions that can be restricted based on execution mode.
    """
    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    
    # Execution operations
    BASH_EXECUTE = "bash_execute"
    PYTHON_EXECUTE = "python_execute"
    
    # Model operations
    MODEL_INFERENCE = "model_inference"
    MODEL_DISCOVERY = "model_discovery"
    
    # System operations
    NETWORK_REQUEST = "network_request"
    DATABASE_WRITE = "database_write"
    MEMORY_WRITE = "memory_write"
    
    # Git operations
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"


@dataclass
class PermissionPolicy:
    """Defines which permissions are allowed in a given execution mode."""
    mode: ExecutionMode
    allowed_permissions: Set[ToolPermission]
    require_confirmation: Set[ToolPermission] = field(default_factory=set)
    deny_patterns: List[str] = field(default_factory=list)  # Regex patterns to deny
    
    def has_permission(self, permission: ToolPermission) -> bool:
        """Check if a permission is granted."""
        return permission in self.allowed_permissions
    
    def requires_confirmation(self, permission: ToolPermission) -> bool:
        """Check if a permission requires user confirmation."""
        return permission in self.require_confirmation


# Default permission policies for each mode
DEFAULT_POLICIES = {
    ExecutionMode.PLANNING: PermissionPolicy(
        mode=ExecutionMode.PLANNING,
        allowed_permissions={
            ToolPermission.FILE_READ,
            ToolPermission.MODEL_INFERENCE,
            ToolPermission.MODEL_DISCOVERY,
            ToolPermission.NETWORK_REQUEST,  # Read-only requests
        },
        require_confirmation={
            ToolPermission.BASH_EXECUTE,
            ToolPermission.PYTHON_EXECUTE,
        },
        deny_patterns=[r"^rm\s", r"^mv\s", r"^del\s"],  # Deny destructive commands
    ),
    ExecutionMode.EXECUTION: PermissionPolicy(
        mode=ExecutionMode.EXECUTION,
        allowed_permissions={
            ToolPermission.FILE_READ,
            ToolPermission.FILE_WRITE,
            ToolPermission.FILE_DELETE,
            ToolPermission.BASH_EXECUTE,
            ToolPermission.PYTHON_EXECUTE,
            ToolPermission.MODEL_INFERENCE,
            ToolPermission.MODEL_DISCOVERY,
            ToolPermission.NETWORK_REQUEST,
            ToolPermission.DATABASE_WRITE,
            ToolPermission.MEMORY_WRITE,
            ToolPermission.GIT_COMMIT,
        },
        require_confirmation={
            ToolPermission.GIT_PUSH,
            ToolPermission.FILE_DELETE,
        },
    ),
    ExecutionMode.SANDBOX: PermissionPolicy(
        mode=ExecutionMode.SANDBOX,
        allowed_permissions={
            ToolPermission.FILE_READ,
            ToolPermission.PYTHON_EXECUTE,
            ToolPermission.MODEL_INFERENCE,
        },
        require_confirmation={
            ToolPermission.BASH_EXECUTE,
        },
        deny_patterns=[r"^import\s+os", r"^import\s+subprocess"],
    ),
}


@dataclass
class ExecutionContext:
    """
    Represents the current execution context.
    
    Tracks mode, permissions, active tasks, and enables safety checks.
    """
    mode: ExecutionMode = ExecutionMode.PLANNING
    policy: Optional[PermissionPolicy] = None
    active_task_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    
    # Hooks
    pre_execution_hooks: List[Callable[[str, Dict[str, Any]], bool]] = field(default_factory=list)
    post_execution_hooks: List[Callable[[str, Dict[str, Any], Any], None]] = field(default_factory=list)
    
    # Audit trail
    executed_operations: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize policy if not provided."""
        if self.policy is None:
            self.policy = DEFAULT_POLICIES.get(self.mode)
    
    def can_execute(self, permission: ToolPermission) -> bool:
        """
        Check if the current context allows execution of a tool.
        
        Args:
            permission: The ToolPermission to check
            
        Returns:
            True if execution is allowed, False otherwise
        """
        if not self.policy:
            logger.warning(f"No policy defined for mode {self.mode}")
            return False
        return self.policy.has_permission(permission)
    
    def requires_confirmation(self, permission: ToolPermission) -> bool:
        """Check if an operation requires user confirmation."""
        if not self.policy:
            return True
        return self.policy.requires_confirmation(permission)
    
    def add_pre_execution_hook(self, hook: Callable[[str, Dict[str, Any]], bool]):
        """
        Register a hook to run before execution.
        
        Hook signature: (operation_name: str, operation_params: Dict) -> bool
        Should return True to proceed, False to block.
        """
        self.pre_execution_hooks.append(hook)
    
    def add_post_execution_hook(self, hook: Callable[[str, Dict[str, Any], Any], None]):
        """
        Register a hook to run after execution.
        
        Hook signature: (operation_name: str, operation_params: Dict, result: Any) -> None
        """
        self.post_execution_hooks.append(hook)
    
    def record_operation(self, operation_name: str, params: Dict[str, Any], result: Any):
        """Record an executed operation for audit trail."""
        self.executed_operations.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "params": params,
            "result": result,
        })
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get the audit trail of all executed operations."""
        return self.executed_operations.copy()
    
    def __repr__(self) -> str:
        return f"ExecutionContext(mode={self.mode.value}, task_id={self.active_task_id})"


class ContextManager:
    """
    Manages execution contexts for Vetinari.
    
    Provides context-switching, safety checks, and enforcement of permissions.
    """
    
    def __init__(self):
        self._context_stack: List[ExecutionContext] = []
        self._default_context = ExecutionContext(mode=ExecutionMode.PLANNING)
        self._context_stack.append(self._default_context)
    
    @property
    def current_context(self) -> ExecutionContext:
        """Get the current execution context."""
        return self._context_stack[-1] if self._context_stack else self._default_context
    
    @property
    def current_mode(self) -> ExecutionMode:
        """Get the current execution mode."""
        return self.current_context.mode
    
    def switch_mode(self, mode: ExecutionMode, task_id: Optional[str] = None) -> ExecutionContext:
        """
        Switch to a different execution mode.
        
        Args:
            mode: The ExecutionMode to switch to
            task_id: Optional task ID associated with this context
            
        Returns:
            The new ExecutionContext
        """
        context = ExecutionContext(
            mode=mode,
            active_task_id=task_id,
            policy=DEFAULT_POLICIES.get(mode),
        )
        self._context_stack.append(context)
        logger.info(f"Switched to mode {mode.value}" + (f" for task {task_id}" if task_id else ""))
        return context
    
    def pop_context(self) -> Optional[ExecutionContext]:
        """
        Pop the current context and return to the previous one.
        
        Returns:
            The popped ExecutionContext, or None if stack is empty
        """
        if len(self._context_stack) > 1:
            context = self._context_stack.pop()
            logger.info(f"Popped context, returned to {self.current_mode.value}")
            return context
        return None
    
    def check_permission(self, permission: ToolPermission) -> bool:
        """
        Check if the current context allows a permission.
        
        Args:
            permission: The ToolPermission to check
            
        Returns:
            True if allowed, False otherwise
        """
        return self.current_context.can_execute(permission)
    
    def requires_confirmation(self, permission: ToolPermission) -> bool:
        """Check if an operation requires user confirmation."""
        return self.current_context.requires_confirmation(permission)
    
    def enforce_permission(self, permission: ToolPermission, operation_name: str = "operation") -> None:
        """
        Enforce a permission, raising an exception if not allowed.
        
        Args:
            permission: The ToolPermission to enforce
            operation_name: The name of the operation (for error messages)
            
        Raises:
            PermissionError: If the permission is not allowed in the current context
        """
        if not self.check_permission(permission):
            raise PermissionError(
                f"'{operation_name}' operation requires {permission.value} permission, "
                f"which is not allowed in {self.current_mode.value} mode"
            )
    
    @contextmanager
    def temporary_mode(self, mode: ExecutionMode, task_id: Optional[str] = None):
        """
        Context manager for temporarily switching to a different mode.
        
        Usage:
            with context_manager.temporary_mode(ExecutionMode.EXECUTION):
                # Execute operations with EXECUTION permissions
                ...
        """
        self.switch_mode(mode, task_id)
        try:
            yield self.current_context
        finally:
            self.pop_context()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the context manager."""
        ctx = self.current_context
        return {
            "mode": ctx.mode.value,
            "task_id": ctx.active_task_id,
            "started_at": ctx.started_at.isoformat(),
            "operations_count": len(ctx.executed_operations),
            "permissions": [p.value for p in ctx.policy.allowed_permissions] if ctx.policy else [],
        }


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get or create the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def current_mode() -> ExecutionMode:
    """Get the current execution mode."""
    return get_context_manager().current_mode


def current_context() -> ExecutionContext:
    """Get the current execution context."""
    return get_context_manager().current_context
