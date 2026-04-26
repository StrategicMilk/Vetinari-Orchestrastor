"""
Comprehensive security tests for the sandbox module.
Tests dangerous code injection attempts and execution constraints.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.code_sandbox import CodeSandbox, SandboxResult
from vetinari.sandbox_manager import SandboxManager


class TestSandboxDangerousPatterns:
    """Test that dangerous patterns are blocked before subprocess execution."""

    def setup_method(self):
        """Set up SandboxManager for each test."""
        self.manager = SandboxManager()

    def test_eval_pattern_blocked(self):
        """Verify eval() calls are blocked by pre-execution scan."""
        code = "eval('1+1')"
        result = self.manager.execute(code, client_id="test_eval")

        assert not result.success
        assert "eval" in result.error
        assert result.execution_id.startswith("exec_")

    def test_exec_pattern_blocked(self):
        """Verify exec() calls are blocked by pre-execution scan."""
        code = "exec('print(1)')"
        result = self.manager.execute(code, client_id="test_exec")

        assert not result.success
        assert "exec" in result.error

    def test_compile_pattern_blocked(self):
        """Verify compile() calls are blocked by pre-execution scan."""
        code = "compile('print(1)', '<string>', 'exec')"
        result = self.manager.execute(code, client_id="test_compile")

        assert not result.success
        assert "compile" in result.error

    def test_import_pattern_blocked(self):
        """Verify __import__ calls are blocked by pre-execution scan."""
        code = "__import__('os')"
        result = self.manager.execute(code, client_id="test_import")

        assert not result.success
        assert "__import__" in result.error

    def test_open_pattern_blocked(self):
        """Verify file open attempts are blocked by pre-execution scan."""
        code = "open('/etc/passwd', 'r')"
        result = self.manager.execute(code, client_id="test_open")

        assert not result.success
        assert "open" in result.error

    def test_input_pattern_blocked(self):
        """Verify input() calls are blocked by pre-execution scan."""
        code = "input('Enter: ')"
        result = self.manager.execute(code, client_id="test_input")

        assert not result.success
        assert "input" in result.error


class TestSandboxSafeExecution:
    """Test that safe code executes correctly via subprocess CodeSandbox."""

    def setup_method(self):
        """Set up CodeSandbox for each test."""
        self.sandbox = CodeSandbox(max_execution_time=10)

    def test_simple_arithmetic(self):
        """Verify basic arithmetic execution."""
        result = self.sandbox.execute_python("print(1 + 2)")

        assert result.success
        assert "3" in result.output
        assert result.execution_time_ms >= 0

    def test_string_operations(self):
        """Verify string operations work."""
        result = self.sandbox.execute_python("print('hello ' + 'world')")

        assert result.success
        assert "hello world" in result.output

    def test_list_comprehension(self):
        """Verify list comprehensions work."""
        result = self.sandbox.execute_python("print([x * 2 for x in range(5)])")

        assert result.success
        assert "0, 2, 4, 6, 8" in result.output

    def test_function_definition_allowed(self):
        """Verify function definitions are allowed (safe)."""
        code = """
def add(a, b):
    return a + b

print(add(5, 3))
"""
        result = self.sandbox.execute_python(code)

        assert result.success
        assert "8" in result.output

    def test_dict_operations(self):
        """Verify dictionary operations work."""
        result = self.sandbox.execute_python("print({'a': 1, 'b': 2}.get('a'))")

        assert result.success
        assert "1" in result.output


class TestSandboxTimeout:
    """Test timeout enforcement."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Subprocess timeout on Windows may behave differently; tested via CodeSandbox separately.",
    )
    def test_timeout_enforcement(self):
        """Verify code execution times out correctly."""
        sandbox = CodeSandbox(max_execution_time=1)

        code = """
while True:
    pass
"""
        result = sandbox.execute_python(code)

        assert not result.success
        assert result.execution_time_ms >= 1000  # At least the timeout duration

    def test_timeout_with_valid_code(self):
        """Verify timeout doesn't affect fast code."""
        sandbox = CodeSandbox(max_execution_time=10)

        result = sandbox.execute_python("print([x**2 for x in range(100)])")

        assert result.success
        assert result.execution_time_ms < 5000  # Should be fast


class TestSandboxManager:
    """Test the SandboxManager singleton."""

    def test_manager_singleton(self):
        """Verify SandboxManager is a singleton."""
        manager1 = SandboxManager.get_instance()
        manager2 = SandboxManager.get_instance()

        assert manager1 is manager2

    def test_manager_execute_in_process(self):
        """Verify manager delegates to subprocess-based sandbox for in_process type."""
        manager = SandboxManager.get_instance()

        result = manager.execute("print(2 + 2)", sandbox_type="in_process")

        assert result.success
        assert "4" in str(result.result)

    def test_manager_get_status(self):
        """Verify manager provides status information."""
        manager = SandboxManager.get_instance()
        status = manager.get_status()

        assert "subprocess" in status
        assert status["subprocess"]["available"] is True
        assert status["subprocess"]["max_concurrent"] > 0


class TestSandboxErrorHandling:
    """Test error handling in sandbox subprocess execution."""

    def setup_method(self):
        """Set up CodeSandbox for each test."""
        self.sandbox = CodeSandbox(max_execution_time=10)

    def test_runtime_error_handling(self):
        """Verify runtime errors are caught."""
        result = self.sandbox.execute_python("print(1 / 0)")

        assert not result.success
        assert "ZeroDivisionError" in result.error

    def test_name_error_handling(self):
        """Verify undefined name errors are caught."""
        result = self.sandbox.execute_python("print(undefined_variable + 1)")

        assert not result.success
        assert "NameError" in result.error

    def test_syntax_error_handling(self):
        """Verify syntax errors are caught."""
        result = self.sandbox.execute_python("if True")  # Missing colon

        assert not result.success
        assert ("SyntaxError" in result.error) or ("error" in result.error.lower())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
