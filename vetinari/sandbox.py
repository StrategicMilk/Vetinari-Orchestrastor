"""Backward-compatibility shim for vetinari.sandbox.

The sandbox implementation has been split into a package:
  vetinari/sandbox/process.py -- InProcessSandbox, ExternalPluginSandbox, SandboxManager
  vetinari/sandbox/docker.py  -- CodeSandbox, CodeExecutor

Existing code that imports from vetinari.sandbox continues to work unchanged.
"""

from vetinari.sandbox import *  # noqa: F401,F403
from vetinari.sandbox import (  # noqa: F401
    SandboxType, SandboxStatus, SandboxResult, AuditEntry,
    is_path_blocked, ALLOWED_BUILTINS, BLOCKED_BUILTINS,
    InProcessSandbox, ExternalPluginSandbox, SandboxManager,
    sandbox_manager, get_code_executor,
    ExecutionResult, CodeSandbox, CodeExecutor,
    get_subprocess_executor, init_code_executor,
)
