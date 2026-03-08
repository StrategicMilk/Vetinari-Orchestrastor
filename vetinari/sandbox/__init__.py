"""Sandbox package - code execution isolation.

Submodules:
  process.py -- InProcessSandbox, ExternalPluginSandbox, SandboxManager
  docker.py  -- CodeSandbox, CodeExecutor (subprocess-based)
"""

from vetinari.sandbox.process import (  # noqa: F401
    SandboxType, SandboxStatus, SandboxResult, AuditEntry,
    is_path_blocked, ALLOWED_BUILTINS, BLOCKED_BUILTINS,
    InProcessSandbox, ExternalPluginSandbox, SandboxManager,
    sandbox_manager, get_code_executor,
)
from vetinari.sandbox.docker import (  # noqa: F401
    ExecutionResult, CodeSandbox, CodeExecutor,
    get_subprocess_executor, init_code_executor,
)

__all__ = [
    "SandboxType", "SandboxStatus", "SandboxResult", "AuditEntry",
    "is_path_blocked", "ALLOWED_BUILTINS", "BLOCKED_BUILTINS",
    "InProcessSandbox", "ExternalPluginSandbox", "SandboxManager",
    "sandbox_manager", "get_code_executor",
    "ExecutionResult", "CodeSandbox", "CodeExecutor",
    "get_subprocess_executor", "init_code_executor",
]
