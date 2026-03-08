import ast
import uuid
import time
import json
import os
import re
import shlex
import sys
import subprocess
import tempfile
import shutil
import threading
import traceback
import tracemalloc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

# Import structured logging
try:
    from vetinari.structured_logging import get_logger, log_sandbox_execution
    STRUCTURED_LOGGING = True
except ImportError:
    STRUCTURED_LOGGING = False
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = logging.getLogger(__name__)


class SandboxType(Enum):
    IN_PROCESS = "in_process"
    EXTERNAL = "external"


class SandboxStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class SandboxResult:
    execution_id: str
    success: bool
    result: Any = None
    error: str = ""
    execution_time_ms: int = 0
    memory_used_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            'execution_id': self.execution_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'memory_used_mb': self.memory_used_mb
        }


@dataclass
class AuditEntry:
    timestamp: str
    execution_id: str
    operation: str
    sandbox_type: str
    status: str
    duration_ms: int
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'execution_id': self.execution_id,
            'operation': self.operation,
            'sandbox_type': self.sandbox_type,
            'status': self.status,
            'duration_ms': self.duration_ms,
            'details': self.details
        }


def is_path_blocked(file_path: str, blocked_paths: List[str] = None) -> bool:
    """Check if a file path is blocked by the sandbox policy.

    Uses pathlib.Path.resolve() to canonicalize the path, preventing
    symlink traversal attacks that could bypass string-based blocking.

    Args:
        file_path: The path to check.
        blocked_paths: List of blocked path patterns. If None, loads from
            the sandbox_policy.yaml config file.

    Returns:
        True if the path is blocked.
    """
    if blocked_paths is None:
        try:
            import yaml
            config_path = Path(__file__).resolve().parents[1] / "config" / "sandbox_policy.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    policy = yaml.safe_load(f) or {}
                blocked_paths = policy.get("rules", {}).get("blocked_paths", [])
            else:
                blocked_paths = []
        except Exception:
            blocked_paths = []

    if not blocked_paths:
        return False

    # Resolve the candidate path to its canonical absolute form (follows symlinks)
    try:
        resolved = Path(file_path).expanduser().resolve()
    except (OSError, ValueError):
        # If we can't resolve the path, block it defensively
        return True

    resolved_str = str(resolved)

    for pattern in blocked_paths:
        pattern = pattern.strip()
        if not pattern:
            continue

        # Glob-style extension patterns like "*.pem"
        if pattern.startswith("*."):
            ext = pattern[1:]  # e.g. ".pem"
            if resolved_str.endswith(ext):
                return True
            continue

        # Directory / path prefix patterns -- resolve them too
        try:
            blocked_resolved = Path(pattern).expanduser().resolve()
        except (OSError, ValueError):
            continue

        blocked_str = str(blocked_resolved)

        # Check if the resolved path starts with (is inside) the blocked directory
        if resolved_str == blocked_str or resolved_str.startswith(blocked_str + os.sep):
            return True

    return False


ALLOWED_BUILTINS = {
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'type', 'range', 'enumerate', 'zip', 'map', 'filter',
    'len', 'sum', 'min', 'max', 'sorted', 'reversed', 'any', 'all',
    'abs', 'round', 'pow',
    'print', 'isinstance', 'hasattr',
    'isinstance', 'issubclass', 'callable', 'id', 'hash',
    'strip', 'split', 'join', 'replace', 'upper', 'lower', 'title',
    'startswith', 'endswith', 'find', 'count', 'format',
    'append', 'extend', 'pop', 'get', 'keys', 'values', 'items', 'copy',
}

# Shell metacharacters that indicate command injection attempts
_SHELL_METACHARACTERS = re.compile(r'[;|&`]|\$\(')


class _DangerousNodeVisitor(ast.NodeVisitor):
    """AST visitor that detects dangerous constructs in sandbox code."""

    DANGEROUS_NAMES: frozenset = frozenset({
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        'getattr', 'setattr', 'delattr',
    })
    DANGEROUS_ATTRS: frozenset = frozenset({
        '__builtins__', '__globals__', '__locals__', '__subclasses__',
        '__bases__', '__mro__', '__code__', '__func__',
    })

    def __init__(self):
        self.violations: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_NAMES:
            self.violations.append(f"Call to dangerous builtin '{node.func.id}'")
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.DANGEROUS_NAMES:
            self.violations.append(f"Call to dangerous attribute '{node.func.attr}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.DANGEROUS_ATTRS:
            self.violations.append(f"Access to dangerous attribute '{node.attr}'")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.violations.append(f"Import statement not allowed: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.violations.append(f"Import statement not allowed: 'from {node.module}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ('__builtins__', '__globals__', '__locals__'):
            self.violations.append(f"Access to dangerous name '{node.id}'")
        self.generic_visit(node)


def _check_code_safety(code: str) -> Optional[str]:
    """Parse code into AST and check for dangerous constructs.

    Returns an error string if dangerous patterns are found, else ``None``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    visitor = _DangerousNodeVisitor()
    visitor.visit(tree)
    if visitor.violations:
        return f"Dangerous pattern detected: {'; '.join(visitor.violations)}"
    return None


BLOCKED_BUILTINS = {
    'open', 'eval', 'exec', 'compile', '__import__',
    'exit', 'quit', 'input', 'raw_input',
    'vars', 'dir', 'globals', 'locals', 'memoryview',
    '__builtins__', '__globals__', '__locals__',
}


class InProcessSandbox:
    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        allowed_builtins: set = None
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_builtins = allowed_builtins or ALLOWED_BUILTINS
        self._timeout_lock = threading.Lock()
        self._execution_done = threading.Event()

    def _get_safe_builtins(self) -> Dict:
        """Get safe builtins for restricted execution."""
        import builtins
        return {k: getattr(builtins, k) for k in self.allowed_builtins if hasattr(builtins, k)}

    def execute(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # AST-based safety check replaces naive string pattern matching
        safety_error = _check_code_safety(code)
        if safety_error:
            return SandboxResult(
                success=False,
                error=f"Safety check failed: {safety_error}",
                execution_time_ms=int((time.time() - start_time) * 1000),
                execution_id=execution_id
            )

        # Use a subprocess instead of an unkillable daemon thread so the
        # process can actually be terminated on timeout via proc.kill().
        import base64 as _b64

        context_json = json.dumps(context or {})
        code_b64 = _b64.b64encode(code.encode("utf-8")).decode("ascii")
        context_b64 = _b64.b64encode(context_json.encode("utf-8")).decode("ascii")
        allowed_b64 = _b64.b64encode(
            json.dumps(list(self.allowed_builtins)).encode("utf-8")
        ).decode("ascii")

        # Runner script executed in a child process with restricted builtins.
        # Uses compile() to pre-compile code before eval/exec (no raw
        # eval/exec on string input).
        runner_code = (
            'import sys, json, traceback, tracemalloc, base64, builtins\n'
            f'allowed_names = json.loads(base64.b64decode("{allowed_b64}").decode("utf-8"))\n'
            'safe_builtins = {k: getattr(builtins, k) for k in allowed_names if hasattr(builtins, k)}\n'
            f'code = base64.b64decode("{code_b64}").decode("utf-8")\n'
            f'context = json.loads(base64.b64decode("{context_b64}").decode("utf-8"))\n'
            'tracemalloc.start()\n'
            'try:\n'
            '    restricted_globals = {"__builtins__": safe_builtins}\n'
            '    restricted_globals.update(context)\n'
            '    try:\n'
            '        compiled = compile(code, "<sandbox>", "eval")\n'
            '        result = eval(compiled, restricted_globals, {})\n'
            '    except SyntaxError:\n'
            '        compiled = compile(code, "<sandbox>", "exec")\n'
            '        exec_globals = restricted_globals.copy()\n'
            '        exec(compiled, exec_globals)\n'
            '        lines = code.strip().split("\\n")\n'
            '        result = None\n'
            '        if lines:\n'
            '            last_line = lines[-1].strip()\n'
            '            try:\n'
            '                last_compiled = compile(last_line, "<sandbox>", "eval")\n'
            '                result = eval(last_compiled, exec_globals, {})\n'
            '            except (SyntaxError, TypeError):\n'
            '                pass\n'
            '    current, peak = tracemalloc.get_traced_memory()\n'
            '    peak_mb = peak / (1024 * 1024)\n'
            '    tracemalloc.stop()\n'
            '    output = {"success": True, "result": repr(result), "peak_mb": peak_mb}\n'
            '    sys.stdout.write(json.dumps(output))\n'
            'except Exception as e:\n'
            '    current, peak = tracemalloc.get_traced_memory()\n'
            '    peak_mb = peak / (1024 * 1024)\n'
            '    tracemalloc.stop()\n'
            '    output = {"success": False, "error": f"{type(e).__name__}: {e}", "peak_mb": peak_mb}\n'
            '    sys.stdout.write(json.dumps(output))\n'
        )

        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                result = SandboxResult(
                    success=False,
                    error=f"Execution timeout after {self.timeout}s",
                    execution_time_ms=self.timeout * 1000,
                    execution_id=execution_id
                )
                if STRUCTURED_LOGGING:
                    log_sandbox_execution(execution_id, False, self.timeout * 1000, 0.0)
                return result

            elapsed_ms = int((time.time() - start_time) * 1000)

            if proc.returncode != 0:
                result = SandboxResult(
                    success=False,
                    error=stderr or f"Process exited with code {proc.returncode}",
                    execution_time_ms=elapsed_ms,
                    execution_id=execution_id
                )
                if STRUCTURED_LOGGING:
                    log_sandbox_execution(execution_id, False, elapsed_ms, 0.0, error=stderr)
                return result

            try:
                output = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                output = {"success": False, "error": f"Unexpected output: {stdout[:200]}"}

            if output.get("success"):
                result = SandboxResult(
                    success=True,
                    result=output.get("result"),
                    execution_time_ms=elapsed_ms,
                    memory_used_mb=output.get("peak_mb", 0.0),
                    execution_id=execution_id
                )
                if STRUCTURED_LOGGING:
                    log_sandbox_execution(execution_id, True, elapsed_ms, output.get("peak_mb", 0.0))
                return result
            else:
                result = SandboxResult(
                    success=False,
                    error=output.get("error", "Unknown error"),
                    execution_time_ms=elapsed_ms,
                    execution_id=execution_id
                )
                if STRUCTURED_LOGGING:
                    log_sandbox_execution(execution_id, False, elapsed_ms, 0.0, error=output.get("error"))
                return result

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            result = SandboxResult(
                success=False,
                error=f"Sandbox execution failed: {type(e).__name__}: {e}",
                execution_time_ms=elapsed_ms,
                execution_id=execution_id
            )
            if STRUCTURED_LOGGING:
                log_sandbox_execution(execution_id, False, elapsed_ms, 0.0, error=str(e))
            return result


class ExternalPluginSandbox:
    ALLOWED_HOOKS = ['read_file', 'write_file', 'search_code']

    def __init__(
        self,
        plugin_dir: str = "./plugins",
        timeout: int = 300,
        max_memory_mb: int = 2048
    ):
        self.plugin_dir = Path(plugin_dir)
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.loaded_plugins: Dict[str, Any] = {}
        self.audit_log: List[AuditEntry] = []

    def discover_plugins(self) -> List[Dict]:
        manifests = []
        if not self.plugin_dir.exists():
            return manifests

        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir():
                manifest_file = plugin_path / "manifest.yaml"
                if manifest_file.exists():
                    try:
                        import yaml
                        with open(manifest_file) as f:
                            data = yaml.safe_load(f)
                            manifests.append(data)
                    except Exception:
                        pass
        return manifests

    def execute_hook(
        self,
        plugin_name: str,
        hook_name: str,
        params: Dict
    ) -> Any:
        if hook_name not in self.ALLOWED_HOOKS:
            return {"error": f"Hook {hook_name} not allowed"}

        execution_id = f"plugin_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        self._log_audit(AuditEntry(
            timestamp=datetime.now().isoformat(),
            execution_id=execution_id,
            operation=hook_name,
            sandbox_type="external",
            status="executing",
            duration_ms=0,
            details={'plugin': plugin_name, 'params': params}
        ))

        try:
            # Look for hook file in .vetinari/hooks/{hook_name}.py
            hook_file = Path(".vetinari") / "hooks" / f"{hook_name}.py"
            if not hook_file.exists():
                result = {"status": "not_found", "hook": hook_name}
                self._log_audit(AuditEntry(
                    timestamp=datetime.now().isoformat(),
                    execution_id=execution_id,
                    operation=hook_name,
                    sandbox_type="external",
                    status="not_found",
                    duration_ms=int((time.time() - start_time) * 1000),
                    details={'plugin': plugin_name, 'hook_file': str(hook_file)}
                ))
                return result

            # Execute hook via subprocess
            hook_result = subprocess.run(
                [sys.executable, str(hook_file)],
                input=json.dumps(params),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if hook_result.returncode == 0:
                try:
                    result = json.loads(hook_result.stdout)
                except (json.JSONDecodeError, ValueError):
                    result = {"status": "success", "hook": hook_name, "output": hook_result.stdout}
            else:
                result = {"status": "error", "hook": hook_name, "error": hook_result.stderr}

            self._log_audit(AuditEntry(
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id,
                operation=hook_name,
                sandbox_type="external",
                status="success" if hook_result.returncode == 0 else "error",
                duration_ms=int((time.time() - start_time) * 1000),
                details={'plugin': plugin_name, 'returncode': hook_result.returncode}
            ))

            return result

        except subprocess.TimeoutExpired:
            self._log_audit(AuditEntry(
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id,
                operation=hook_name,
                sandbox_type="external",
                status="timeout",
                duration_ms=int((time.time() - start_time) * 1000),
                details={'plugin': plugin_name, 'timeout': self.timeout}
            ))
            return {"status": "timeout", "hook": hook_name}

        except Exception as e:
            self._log_audit(AuditEntry(
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id,
                operation=hook_name,
                sandbox_type="external",
                status="error",
                duration_ms=int((time.time() - start_time) * 1000),
                details={'plugin': plugin_name, 'error': str(e)}
            ))
            return {"error": str(e)}

    def _log_audit(self, entry: AuditEntry):
        self.audit_log.append(entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return [e.to_dict() for e in self.audit_log[-limit:]]


class SandboxManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.in_process = InProcessSandbox()
        self.external = ExternalPluginSandbox()
        self.current_load = 0.0
        self.max_concurrent = 5

    def execute(
        self,
        code: str,
        sandbox_type: str = "in_process",
        timeout: int = 30,
        context: Dict = None
    ) -> SandboxResult:
        if sandbox_type == "in_process":
            return self.in_process.execute(code, context)
        elif sandbox_type == "subprocess":
            # External subprocess sandbox using CodeSandbox (defined below)
            csb = CodeSandbox(
                max_execution_time=timeout,
                allow_network=False,
            )
            result = csb.execute_python(code, input_data=context or {})
            # Map ExecutionResult fields -> SandboxResult fields
            combined_output = result.output or ""
            if result.stdout:
                combined_output = result.stdout + ("\n" + combined_output if combined_output else "")
            execution_id = str(uuid.uuid4())
            return SandboxResult(
                execution_id=execution_id,
                success=result.success,
                result=combined_output,
                error=result.error or (result.stderr or ""),
                execution_time_ms=result.execution_time_ms,
            )
        else:
            # Unknown type -- fall back to in-process
            return self.in_process.execute(code, context)

    def get_status(self) -> Dict:
        return {
            'in_process': {
                'available': True,
                'current_load': self.current_load,
                'max_concurrent': self.max_concurrent,
                'queue_length': 0
            },
            'external': {
                'available': True,
                'plugins_loaded': len(self.external.loaded_plugins),
                'isolation': 'process'
            }
        }

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self.external.get_audit_log(limit)


sandbox_manager = SandboxManager.get_instance()


def get_code_executor() -> SandboxManager:
    """Get the singleton sandbox manager as a code executor."""
    return sandbox_manager


# ---------------------------------------------------------------------------
# Subprocess-based sandbox (consolidated from code_sandbox.py)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str = ""
    execution_time_ms: int = 0
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    files_created: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "files_created": self.files_created,
            "metadata": self.metadata,
        }


class CodeSandbox:
    """
    Sandboxed code execution environment.

    Provides:
    - Isolated execution
    - Resource limits
    - Timeout handling
    - Output capture
    - Multiple language support
    """

    def __init__(self,
                 working_dir: str = None,
                 max_execution_time: int = 60,
                 max_memory_mb: int = 512,
                 allow_network: bool = False,
                 allowed_modules: List[str] = None,
                 blocked_modules: List[str] = None):
        """
        Initialize the code sandbox.

        Args:
            working_dir: Working directory for execution
            max_execution_time: Maximum execution time in seconds
            max_memory_mb: Maximum memory in MB
            allow_network: Allow network access
            allowed_modules: Whitelist of allowed modules
            blocked_modules: Blacklist of blocked modules
        """
        self.working_dir = Path(working_dir or tempfile.mkdtemp(prefix="vetinari_sandbox_"))
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.allow_network = allow_network

        # Module restrictions
        self.allowed_modules = allowed_modules or []
        self.blocked_modules = blocked_modules or [
            "os",  # Will be partially allowed
            "sys",
            "subprocess",
            "socket",
            "requests",
            "urllib",
        ]

        # Execution tracking
        self._execution_count = 0
        self._lock = threading.Lock()

        logger.info(f"CodeSandbox initialized (working_dir={self.working_dir})")

    def execute_python(self,
                     code: str,
                     input_data: Dict[str, Any] = None,
                     env_vars: Dict[str, str] = None,
                     timeout: int = None) -> ExecutionResult:
        """
        Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            input_data: Input data to pass to the code
            env_vars: Environment variables
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with output and status
        """
        timeout = timeout or self.max_execution_time

        # Generate unique file
        execution_id = str(uuid.uuid4())[:8]
        script_file = self.working_dir / f"script_{execution_id}.py"

        # Wrap code to capture output
        wrapped_code = self._wrap_python_code(code, input_data)

        # Write code to file
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(wrapped_code)

        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Add sandbox to path
        env["PYTHONPATH"] = str(self.working_dir)

        # Execute
        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(self.working_dir)
            )

            execution_time = int((time.time() - start_time) * 1000)

            # Parse output
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time_ms=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                metadata={"execution_id": execution_id, "script": str(script_file)}
            )

        except subprocess.TimeoutExpired:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout} seconds",
                execution_time_ms=execution_time,
                return_code=-1,
                metadata={"execution_id": execution_id, "timeout": True}
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {str(e)}\n{traceback.format_exc()}",
                execution_time_ms=execution_time,
                return_code=-1,
                metadata={"execution_id": execution_id, "exception": str(e)}
            )

        finally:
            # Cleanup
            if script_file.exists():
                try:
                    script_file.unlink()
                except Exception:
                    pass

    def _wrap_python_code(self, code: str, input_data: Dict[str, Any] = None) -> str:
        """Wrap code with input/output handling and proper indentation inside try block."""
        import base64 as _b64
        input_json = json.dumps(input_data or {})
        # Encode user code as base64 to avoid any escaping / indentation issues
        code_b64 = _b64.b64encode(code.encode("utf-8")).decode("ascii")

        wrapped = f'''
import sys
import json
import traceback
import base64

# Input data
INPUT_DATA = {input_json}

# Capture output
_output = []
_errors = []

class OutputCapture:
    def write(self, text):
        if text.strip():
            _output.append(text)
    def flush(self):
        pass

sys.stdout = OutputCapture()
sys.stderr = OutputCapture()

# User code (base64-encoded to preserve indentation and avoid injection)
_user_code = base64.b64decode("{code_b64}").decode("utf-8")
try:
    exec(compile(_user_code, "<vetinari_sandbox>", "exec"), {{}})
except Exception as e:
    _errors.append(traceback.format_exc())

# Output results
result = {{
    "success": not _errors,
    "output": "".join(_output),
    "errors": "".join(_errors),
    "input_received": INPUT_DATA
}}

print("===VETINARI_OUTPUT_START===")
print(json.dumps(result))
print("===VETINARI_OUTPUT_END===")
'''
        return wrapped

    def execute_shell(self,
                    command: str,
                    timeout: int = None) -> ExecutionResult:
        """Execute a shell command.

        Security: uses shlex.split() with shell=False to prevent command
        injection.  Commands containing shell metacharacters are rejected.
        """
        timeout = timeout or self.max_execution_time

        # Reject commands containing shell metacharacters
        if _SHELL_METACHARACTERS.search(command):
            return ExecutionResult(
                success=False,
                output="",
                error="Command rejected: shell metacharacters (;|&`$()) are not allowed",
                return_code=-1,
            )

        start_time = time.time()

        try:
            result = subprocess.run(
                shlex.split(command),
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir)
            )

            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time_ms=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except subprocess.TimeoutExpired:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                execution_time_ms=execution_time,
                return_code=-1,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=execution_time,
                return_code=-1,
            )

    def test_code(self,
                 code: str,
                 test_code: str,
                 timeout: int = 60) -> ExecutionResult:
        """
        Execute code with tests.

        Args:
            code: Code to test
            test_code: Test code
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with test results
        """
        # Combine code and tests
        combined_code = f'''
# Code under test
{code}

# Tests
{test_code}
'''

        return self.execute_python(combined_code, timeout=timeout)

    def run_tests(self,
                 test_dir: str = None,
                 test_pattern: str = "test_*.py",
                 verbose: bool = True) -> ExecutionResult:
        """Run pytest tests in a directory."""
        test_dir = Path(test_dir) if test_dir else self.working_dir

        args = ["-m", "pytest"]
        if verbose:
            args.append("-v")
        args.append(str(test_dir))

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable] + args,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=str(test_dir)
            )

            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time_ms=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=execution_time,
                return_code=-1,
            )

    def lint_code(self,
                 code: str,
                 linter: str = "ruff") -> ExecutionResult:
        """Lint code with specified linter."""
        # Write code to temp file
        execution_id = str(uuid.uuid4())[:8]
        script_file = self.working_dir / f"lint_{execution_id}.py"

        with open(script_file, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [linter, "check", str(script_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.working_dir)
            )

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Linter '{linter}' not found",
                return_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
            )
        finally:
            if script_file.exists():
                script_file.unlink()

    def cleanup(self):
        """Clean up the sandbox."""
        if self.working_dir.exists():
            try:
                shutil.rmtree(self.working_dir)
                logger.info(f"Cleaned up sandbox: {self.working_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        return {
            "working_dir": str(self.working_dir),
            "execution_count": self._execution_count,
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "allow_network": self.allow_network,
        }


class CodeExecutor:
    """
    Higher-level code execution interface.

    Provides:
    - Simplified API
    - Result parsing
    - Automatic cleanup
    - Error handling
    """

    def __init__(self, sandbox: CodeSandbox = None):
        self.sandbox = sandbox or CodeSandbox()

    def run(self,
           code: str,
           language: str = "python",
           **kwargs) -> Dict[str, Any]:
        """
        Run code and return results.

        Args:
            code: Code to run
            language: Language (python, shell)
            **kwargs: Additional arguments

        Returns:
            Dictionary with results
        """
        if language.lower() == "python":
            result = self.sandbox.execute_python(code, **kwargs)
        elif language.lower() in ("bash", "shell", "sh"):
            result = self.sandbox.execute_shell(code, **kwargs)
        else:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "output": "",
            }

        return result.to_dict()

    def run_and_validate(self,
                        code: str,
                        expected_output: str = None,
                        error_pattern: str = None) -> Dict[str, Any]:
        """
        Run code and validate output.

        Args:
            code: Code to run
            expected_output: Expected output substring
            error_pattern: Pattern that should NOT appear in output

        Returns:
            Validation results
        """
        result = self.run(code)

        validation = {
            "success": result["success"],
            "output": result["output"],
            "error": result["error"],
            "validations": {}
        }

        # Check expected output
        if expected_output:
            validation["validations"]["expected_output"] = expected_output in result["output"]

        # Check error pattern
        if error_pattern:
            validation["validations"]["no_errors"] = error_pattern not in result["output"] and error_pattern not in result["error"]

        # Overall validation
        validation["all_validations_passed"] = all(
            v for v in validation["validations"].values()
        )

        return validation

    def test_with_input(self,
                       code: str,
                       inputs: List[Any]) -> List[Dict[str, Any]]:
        """
        Run code with multiple inputs.

        Args:
            code: Code to run
            inputs: List of inputs

        Returns:
            List of results
        """
        results = []

        for input_data in inputs:
            result = self.run(code, input_data={"value": input_data})
            results.append({
                "input": input_data,
                "result": result
            })

        return results


# Global subprocess executor
_subprocess_executor: Optional[CodeExecutor] = None


def get_subprocess_executor() -> CodeExecutor:
    """Get or create the global subprocess-based code executor."""
    global _subprocess_executor
    if _subprocess_executor is None:
        _subprocess_executor = CodeExecutor()
    return _subprocess_executor


def init_code_executor(**kwargs) -> CodeExecutor:
    """Initialize a new code executor."""
    global _subprocess_executor
    sandbox = CodeSandbox(**kwargs)
    _subprocess_executor = CodeExecutor(sandbox)
    return _subprocess_executor
