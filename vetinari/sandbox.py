import uuid
import time
import json
import sys
import threading
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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


ALLOWED_BUILTINS = {
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'type', 'range', 'enumerate', 'zip', 'map', 'filter',
    'len', 'sum', 'min', 'max', 'sorted', 'reversed', 'any', 'all',
    'abs', 'round', 'pow',
    'print', 'isinstance', 'hasattr', 'getattr', 'setattr',
    'isinstance', 'issubclass', 'callable', 'id', 'hash',
    'strip', 'split', 'join', 'replace', 'upper', 'lower', 'title',
    'startswith', 'endswith', 'find', 'count', 'format',
    'append', 'extend', 'pop', 'get', 'keys', 'values', 'items', 'copy',
}


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

        # Check for dangerous patterns before execution (check compile first)
        dangerous_patterns = ['compile', 'eval', 'exec', '__import__', 'open', 'input']
        for pattern in dangerous_patterns:
            if pattern in code:
                return SandboxResult(
                    success=False,
                    error=f"Dangerous pattern '{pattern}' not allowed in sandbox",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    execution_id=execution_id
                )

        # Use threading timeout instead of signal (works on Windows)
        result_holder = [None]
        error_holder = [None]
        peak_memory = [0.0]
        
        def run_code():
            # Start tracemalloc inside the thread to track memory
            import tracemalloc
            tracemalloc.start()
            try:
                restricted_globals = {
                    '__builtins__': self._get_safe_builtins()
                }
                if context:
                    restricted_globals.update(context)
                
                # Use exec() to support both expressions and statements (like function definitions)
                # but capture the last expression value if it's an expression
                is_expression = True
                try:
                    compile(code, '<string>', 'eval')
                except SyntaxError:
                    is_expression = False
                
                if is_expression:
                    result_holder[0] = eval(code, restricted_globals, {})
                else:
                    # For statements (like function defs + calls), capture the last expression result
                    # by evaluating the last line if it's an expression statement
                    lines = code.strip().split('\n')
                    exec_globals = restricted_globals.copy()
                    exec(code, exec_globals)
                    # Try to get result from last line if it's a function call
                    if lines:
                        last_line = lines[-1].strip()
                        # Check if last line is an expression (not a statement)
                        try:
                            compile(last_line, '<string>', 'eval')
                            result_holder[0] = eval(last_line, exec_globals, {})
                        except (SyntaxError, TypeError):
                            result_holder[0] = None
            except Exception as e:
                error_holder[0] = e
            finally:
                # Get peak memory in this thread
                current, peak = tracemalloc.get_traced_memory()
                peak_memory[0] = peak / (1024 * 1024)
                tracemalloc.stop()

        execution_thread = threading.Thread(target=run_code, daemon=True)
        execution_thread.start()
        execution_thread.join(timeout=self.timeout)

        if execution_thread.is_alive():
            # Timeout occurred
            result = SandboxResult(
                success=False,
                error=f"Execution timeout after {self.timeout}s",
                execution_time_ms=self.timeout * 1000,
                execution_id=execution_id
            )
            if STRUCTURED_LOGGING:
                log_sandbox_execution(execution_id, False, self.timeout * 1000, 0.0)
            return result

        if error_holder[0]:
            result = SandboxResult(
                success=False,
                error=f"{type(error_holder[0]).__name__}: {str(error_holder[0])}",
                execution_time_ms=int((time.time() - start_time) * 1000),
                execution_id=execution_id
            )
            if STRUCTURED_LOGGING:
                log_sandbox_execution(execution_id, False, int((time.time() - start_time) * 1000), 0.0, error=str(error_holder[0]))
            return result

        result = SandboxResult(
            success=True,
            result=result_holder[0],
            execution_time_ms=int((time.time() - start_time) * 1000),
            memory_used_mb=peak_memory[0],
            execution_id=execution_id
        )
        if STRUCTURED_LOGGING:
            log_sandbox_execution(execution_id, True, int((time.time() - start_time) * 1000), peak_memory[0])
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
                    except:
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
            result = {"status": "simulated", "hook": hook_name}

            self._log_audit(AuditEntry(
                timestamp=datetime.now().isoformat(),
                execution_id=execution_id,
                operation=hook_name,
                sandbox_type="external",
                status="success",
                duration_ms=int((time.time() - start_time) * 1000),
                details={'plugin': plugin_name}
            ))

            return result

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
        else:
            return SandboxResult(
                execution_id="",
                success=False,
                error="External sandbox not implemented in Phase 1"
            )

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
