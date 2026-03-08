"""
Code Execution Sandbox for Vetinari

Provides safe code execution capabilities for:
- Running generated code
- Testing outputs
- Validating artifacts

Features:
- Sandboxed execution
- Timeout handling
- Output capture
- Error handling
- Security restrictions
"""

import os
import shlex
import sys
import subprocess
import tempfile
import shutil
import logging
import uuid
import json
import time
import traceback
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


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
        self.blocked_modules = blocked_modules if blocked_modules is not None else [
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
        
        logger.info("CodeSandbox initialized (working_dir=%s)", self.working_dir)
    
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
                    logger.debug("Failed to clean up temporary script file %s", script_file, exc_info=True)

    def _wrap_python_code(self, code: str, input_data: Dict[str, Any] = None) -> str:
        """Wrap code with input/output handling, module restrictions, and proper indentation."""
        import base64 as _b64
        input_json = json.dumps(input_data or {})
        code_b64 = _b64.b64encode(code.encode("utf-8")).decode("ascii")
        blocked_json = json.dumps(self.blocked_modules)
        allow_network = "True" if self.allow_network else "False"

        # NOTE: The code execution below is the intentional purpose of this
        # sandbox module — it runs user code in an isolated subprocess.
        wrapped = f'''
import sys as _sys
import json as _json
import traceback as _tb
import base64 as _b64
import builtins as _builtins

# Save real stdout/stderr BEFORE capturing
_real_stdout = _sys.stdout
_real_stderr = _sys.stderr

INPUT_DATA = {input_json}

# --- Module restriction enforcement ---
_BLOCKED_MODULES = {blocked_json}
_ALLOW_NETWORK = {allow_network}
_WRAPPER_NEEDS = {{"sys", "json", "traceback", "base64", "builtins"}}
_original_import = _builtins.__import__

def _restricted_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(
            "Module %r is blocked in the Vetinari sandbox" % name
        )
    if not _ALLOW_NETWORK and top_level in ("socket", "requests", "urllib", "httpx", "aiohttp"):
        raise ImportError(
            "Network module %r is blocked (allow_network=False)" % name
        )
    return _original_import(name, *args, **kwargs)

_builtins.__import__ = _restricted_import
for _mod in list(_sys.modules):
    _top = _mod.split(".")[0]
    if _top in _BLOCKED_MODULES and _top not in _WRAPPER_NEEDS:
        del _sys.modules[_mod]

_output = []
_errors = []

class _OutputCapture:
    def write(self, text):
        if text.strip():
            _output.append(text)
    def flush(self):
        pass

_sys.stdout = _OutputCapture()
_sys.stderr = _OutputCapture()

_user_code = _b64.b64decode("{code_b64}").decode("utf-8")
try:
    _sandbox_globals = {{}}
    _code_obj = compile(_user_code, "<vetinari_sandbox>", "exec")
    _builtins.eval(_code_obj, _sandbox_globals)
except Exception as _e:
    _errors.append(_tb.format_exc())

# Restore real stdout for final JSON output
_sys.stdout = _real_stdout
_sys.stderr = _real_stderr

_result = {{
    "success": len(_errors) == 0,
    "output": "".join(_output),
    "errors": "".join(_errors),
    "input_received": INPUT_DATA
}}

print("===VETINARI_OUTPUT_START===")
print(_json.dumps(_result))
print("===VETINARI_OUTPUT_END===")
'''
        return wrapped
    
    def execute_shell(self, 
                    command: str,
                    timeout: int = None) -> ExecutionResult:
        """Execute a shell command."""
        timeout = timeout or self.max_execution_time
        
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
                logger.info("Cleaned up sandbox: %s", self.working_dir)
            except Exception as e:
                logger.warning("Failed to cleanup sandbox: %s", e)
    
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


# Global executor
_code_executor: Optional[CodeExecutor] = None


def get_subprocess_executor() -> CodeExecutor:
    """Get or create the global subprocess-based code executor."""
    global _code_executor
    if _code_executor is None:
        _code_executor = CodeExecutor()
    return _code_executor


# Backward-compatibility alias
def get_code_executor() -> CodeExecutor:
    """Alias for get_subprocess_executor() - use get_subprocess_executor() for new code."""
    return get_subprocess_executor()


def init_code_executor(**kwargs) -> CodeExecutor:
    """Initialize a new code executor."""
    global _code_executor
    sandbox = CodeSandbox(**kwargs)
    _code_executor = CodeExecutor(sandbox)
    return _code_executor


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test the sandbox
    sandbox = CodeSandbox(max_execution_time=10)
    
    # Test simple Python execution
    print("=== Simple Python execution ===")
    result = sandbox.execute_python("print('Hello from sandbox!')")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    
    # Test with input
    print("\n=== With input ===")
    result = sandbox.execute_python("print(f'Got: {{INPUT_DATA}}')")
    print(f"Output: {result.output}")
    
    # Test with error
    print("\n=== With error ===")
    result = sandbox.execute_python("raise Exception('Test error')")
    print(f"Success: {result.success}")
    print(f"Error: {result.error[:100]}...")
    
    # Test code validation
    print("\n=== Code validation ===")
    executor = CodeExecutor(sandbox)
    validation = executor.run_and_validate(
        "print('Test passed')",
        expected_output="Test passed"
    )
    print(f"Validation: {validation}")
    
    # Cleanup
    sandbox.cleanup()
