"""
Comprehensive tests for vetinari/code_sandbox.py

Covers:
- ExecutionResult dataclass and to_dict()
- CodeSandbox: init, execute_python, _wrap_python_code, execute_shell,
  test_code, run_tests, lint_code, cleanup, get_stats
- CodeExecutor: run, run_and_validate, test_with_input
- Singleton helpers: get_subprocess_executor, init_code_executor
"""

import subprocess
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

import vetinari.code_sandbox as cs_module
import vetinari.sandbox_manager as sm_module
from tests.factories import make_proc_result as _make_proc_result
from vetinari.code_sandbox import (
    CodeSandbox,
    ExecutionResult,
    get_subprocess_executor,
    init_code_executor,
)
from vetinari.sandbox_manager import CodeExecutor

# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult(success=True, output="hello")
        assert r.success is True
        assert r.output == "hello"
        assert r.error == ""
        assert r.execution_time_ms == 0
        assert r.return_code == 0
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.files_created == []
        assert r.metadata == {}

    def test_all_fields(self):
        r = ExecutionResult(
            success=False,
            output="out",
            error="err",
            execution_time_ms=123,
            return_code=1,
            stdout="stdout_val",
            stderr="stderr_val",
            files_created=["a.py"],
            metadata={"key": "val"},
        )
        assert r.success is False
        assert r.execution_time_ms == 123
        assert r.return_code == 1
        assert r.files_created == ["a.py"]
        assert r.metadata == {"key": "val"}

    def test_to_dict_keys(self):
        r = ExecutionResult(success=True, output="o")
        d = r.to_dict()
        expected_keys = {
            "success",
            "output",
            "error",
            "execution_time_ms",
            "return_code",
            "stdout",
            "stderr",
            "files_created",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        r = ExecutionResult(
            success=True,
            output="o",
            error="e",
            execution_time_ms=50,
            return_code=0,
            stdout="so",
            stderr="se",
            files_created=["x.py"],
            metadata={"a": 1},
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["output"] == "o"
        assert d["error"] == "e"
        assert d["execution_time_ms"] == 50
        assert d["stdout"] == "so"
        assert d["files_created"] == ["x.py"]
        assert d["metadata"] == {"a": 1}

    def test_files_created_independent_per_instance(self):
        r1 = ExecutionResult(success=True, output="")
        r2 = ExecutionResult(success=True, output="")
        r1.files_created.append("f.py")
        assert r2.files_created == []

    def test_metadata_independent_per_instance(self):
        r1 = ExecutionResult(success=True, output="")
        r2 = ExecutionResult(success=True, output="")
        r1.metadata["x"] = 1
        assert r2.metadata == {}


# ---------------------------------------------------------------------------
# CodeSandbox — __init__
# ---------------------------------------------------------------------------


class TestCodeSandboxInit:
    def test_default_working_dir_created(self, tmp_path):
        with patch("tempfile.mkdtemp", return_value=str(tmp_path / "sandbox")):
            (tmp_path / "sandbox").mkdir()
            sb = CodeSandbox()
        assert sb.working_dir == tmp_path / "sandbox"

    def test_custom_working_dir(self, tmp_path):
        custom = tmp_path / "my_sandbox"
        custom.mkdir()
        sb = CodeSandbox(working_dir=str(custom))
        assert sb.working_dir == custom

    def test_default_params(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        assert sb.max_execution_time == 60
        assert sb.max_memory_mb == 512
        assert sb.allow_network is False

    def test_custom_params(self, tmp_path):
        sb = CodeSandbox(
            working_dir=str(tmp_path),
            max_execution_time=10,
            max_memory_mb=256,
            allow_network=True,
        )
        assert sb.max_execution_time == 10
        assert sb.max_memory_mb == 256
        assert sb.allow_network is True

    def test_default_blocked_modules(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        assert "subprocess" in sb.blocked_modules
        assert "socket" in sb.blocked_modules

    def test_custom_blocked_modules(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), blocked_modules=["requests"])
        assert sb.blocked_modules == ["requests"]

    def test_empty_blocked_modules(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), blocked_modules=[])
        assert sb.blocked_modules == []

    def test_allowed_modules_default_empty(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        assert sb.allowed_modules == []

    def test_custom_allowed_modules(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), allowed_modules=["math", "json"])
        assert sb.allowed_modules == ["math", "json"]

    def test_lock_created(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        assert isinstance(sb._lock, type(threading.Lock()))

    def test_execution_count_starts_zero(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        assert sb._execution_count == 0


# ---------------------------------------------------------------------------
# CodeSandbox — execute_python
# ---------------------------------------------------------------------------


class TestCodeSandboxExecutePython:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path), max_execution_time=30)

    def test_successful_execution(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="hello\n", returncode=0)
            result = sandbox.execute_python("print('hello')")
        assert result.success is True
        assert result.return_code == 0
        assert result.stdout == "hello\n"

    def test_failed_execution_nonzero_return(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stderr="SyntaxError", returncode=1)
            result = sandbox.execute_python("bad code !!!")
        assert result.success is False
        assert result.return_code == 1
        assert result.stderr == "SyntaxError"

    def test_timeout_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = sandbox.execute_python("import time; time.sleep(999)", timeout=1)
        assert result.success is False
        assert result.return_code == -1
        assert "timed out" in result.error

    def test_timeout_metadata_flag(self, sandbox):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = sandbox.execute_python("x = 1", timeout=1)
        assert result.metadata.get("timeout") is True

    def test_exception_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=OSError("no such file")):
            result = sandbox.execute_python("x = 1")
        assert result.success is False
        assert "no such file" in result.error

    def test_execution_time_populated(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result()
            result = sandbox.execute_python("x = 1")
        assert isinstance(result.execution_time_ms, int)
        assert result.execution_time_ms >= 0

    def test_metadata_contains_execution_id(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result()
            result = sandbox.execute_python("x = 1")
        assert "execution_id" in result.metadata

    def test_custom_timeout_overrides_default(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_python("x = 1", timeout=5)
        assert captured["timeout"] == 5

    def test_default_timeout_from_sandbox(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_python("x = 1")
        assert captured["timeout"] == sandbox.max_execution_time

    def test_env_vars_passed_to_subprocess(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_python("x = 1", env_vars={"MY_VAR": "42"})
        assert captured["env"].get("MY_VAR") == "42"

    def test_pythonpath_set_in_env(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_python("x = 1")
        assert "PYTHONPATH" in captured["env"]

    def test_script_file_cleaned_up_on_success(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result()
            sandbox.execute_python("x = 1")
        # After execution the script file should be deleted
        script_files = list(sandbox.working_dir.glob("script_*.py"))
        assert script_files == []

    def test_script_file_cleaned_up_on_error(self, sandbox):
        with patch("subprocess.run", side_effect=OSError("fail")):
            sandbox.execute_python("x = 1")
        script_files = list(sandbox.working_dir.glob("script_*.py"))
        assert script_files == []

    def test_output_and_error_filled(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="my output", stderr="my error", returncode=0)
            result = sandbox.execute_python("x = 1")
        assert result.output == "my output"
        assert result.error == "my error"

    def test_input_data_passed_to_wrap(self, sandbox):
        """_wrap_python_code is called with input_data."""
        with patch.object(sandbox, "_wrap_python_code", wraps=sandbox._wrap_python_code) as mock_wrap:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = _make_proc_result()
                sandbox.execute_python("x = 1", input_data={"key": "value"})
        mock_wrap.assert_called_once_with("x = 1", {"key": "value"})


# ---------------------------------------------------------------------------
# CodeSandbox — _wrap_python_code
# ---------------------------------------------------------------------------


class TestWrapPythonCode:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path))

    def test_returns_string(self, sandbox):
        result = sandbox._wrap_python_code("x = 1")
        assert isinstance(result, str)

    def test_contains_base64_encoded_user_code(self, sandbox):
        import base64

        code = "print('hello world')"
        wrapped = sandbox._wrap_python_code(code)
        encoded = base64.b64encode(code.encode()).decode("ascii")
        assert encoded in wrapped

    def test_contains_output_markers(self, sandbox):
        wrapped = sandbox._wrap_python_code("x = 1")
        assert "VETINARI_OUTPUT_START" in wrapped
        assert "VETINARI_OUTPUT_END" in wrapped

    def test_contains_blocked_modules_json(self, sandbox):
        import json

        wrapped = sandbox._wrap_python_code("x = 1")
        blocked_json = json.dumps(sandbox.blocked_modules)
        assert blocked_json in wrapped

    def test_allow_network_false_in_wrapper(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), allow_network=False)
        wrapped = sb._wrap_python_code("x = 1")
        assert "False" in wrapped

    def test_allow_network_true_in_wrapper(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), allow_network=True)
        wrapped = sb._wrap_python_code("x = 1")
        assert "True" in wrapped

    def test_input_data_serialized(self, sandbox):
        import json

        data = {"my_key": "my_value", "num": 42}
        wrapped = sandbox._wrap_python_code("x = 1", input_data=data)
        assert json.dumps(data) in wrapped

    def test_empty_input_data_default(self, sandbox):
        wrapped = sandbox._wrap_python_code("x = 1")
        assert "{}" in wrapped

    def test_restricted_import_function_defined(self, sandbox):
        wrapped = sandbox._wrap_python_code("x = 1")
        assert "_restricted_import" in wrapped

    def test_output_capture_class_defined(self, sandbox):
        wrapped = sandbox._wrap_python_code("x = 1")
        assert "_OutputCapture" in wrapped


# ---------------------------------------------------------------------------
# CodeSandbox — execute_shell
# ---------------------------------------------------------------------------


class TestCodeSandboxExecuteShell:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path))

    def test_successful_command(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="ok\n", returncode=0)
            result = sandbox.execute_shell("echo ok")
        assert result.success is True
        assert result.stdout == "ok\n"

    def test_failed_command(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(returncode=127)
            result = sandbox.execute_shell("nosuchcmd")
        assert result.success is False

    def test_timeout_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            result = sandbox.execute_shell("python -c 'import time; time.sleep(100)'", timeout=1)
        assert result.success is False
        assert "timed out" in result.error

    def test_exception_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=ValueError("bad")):
            result = sandbox.execute_shell("echo x")
        assert result.success is False
        assert "bad" in result.error

    def test_shell_false_used(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["shell"] = kwargs.get("shell")
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_shell("echo x")
        assert captured["shell"] is False

    def test_command_split_with_shlex(self, sandbox):
        """Verify shlex.split is used (args passed as list, not string)."""
        captured = {}

        def capture_run(*args, **kwargs):
            captured["args"] = args[0] if args else None
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_shell("echo hello world")
        # shlex.split("echo hello world") -> ["echo", "hello", "world"]
        assert isinstance(captured["args"], list)
        assert captured["args"] == ["echo", "hello", "world"]

    def test_working_dir_passed(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["cwd"] = kwargs.get("cwd")
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.execute_shell("echo x")
        assert captured["cwd"] == str(sandbox.working_dir)

    def test_stdout_and_stderr_populated(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="out", stderr="err", returncode=0)
            result = sandbox.execute_shell("echo x")
        assert result.stdout == "out"
        assert result.stderr == "err"


# ---------------------------------------------------------------------------
# CodeSandbox — test_code
# ---------------------------------------------------------------------------


class TestCodeSandboxTestCode:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path))

    def test_combines_code_and_tests(self, sandbox):
        captured = {}

        def fake_execute(code, **kwargs):
            captured["code"] = code
            return ExecutionResult(success=True, output="")

        with patch.object(sandbox, "execute_python", side_effect=fake_execute):
            sandbox.test_code("def foo(): pass", "assert foo() is None")
        assert "def foo(): pass" in captured["code"]
        assert "assert foo() is None" in captured["code"]

    def test_passes_timeout(self, sandbox):
        captured = {}

        def fake_execute(code, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return ExecutionResult(success=True, output="")

        with patch.object(sandbox, "execute_python", side_effect=fake_execute):
            sandbox.test_code("x = 1", "assert x == 1", timeout=15)
        assert captured["timeout"] == 15

    def test_returns_execution_result(self, sandbox):
        with patch.object(
            sandbox,
            "execute_python",
            return_value=ExecutionResult(success=True, output="passed"),
        ):
            result = sandbox.test_code("x = 1", "assert x == 1")
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    def test_default_timeout_is_60(self, sandbox):
        captured = {}

        def fake_execute(code, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return ExecutionResult(success=True, output="")

        with patch.object(sandbox, "execute_python", side_effect=fake_execute):
            sandbox.test_code("x = 1", "assert x == 1")
        assert captured["timeout"] == 60


# ---------------------------------------------------------------------------
# CodeSandbox — run_tests
# ---------------------------------------------------------------------------


class TestCodeSandboxRunTests:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path), max_execution_time=30)

    def test_successful_pytest_run(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="5 passed", returncode=0)
            result = sandbox.run_tests()
        assert result.success is True
        assert "5 passed" in result.stdout

    def test_failed_tests(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="2 failed", returncode=1)
            result = sandbox.run_tests()
        assert result.success is False

    def test_verbose_flag_included(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["args"] = args[0]
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.run_tests(verbose=True)
        assert "-v" in captured["args"]

    def test_verbose_false_excludes_flag(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["args"] = args[0]
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.run_tests(verbose=False)
        assert "-v" not in captured["args"]

    def test_custom_test_dir(self, sandbox, tmp_path):
        test_dir = tmp_path / "my_tests"
        test_dir.mkdir()
        captured = {}

        def capture_run(*args, **kwargs):
            captured["args"] = args[0]
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.run_tests(test_dir=str(test_dir))
        assert str(test_dir) in captured["args"]

    def test_exception_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=FileNotFoundError("pytest not found")):
            result = sandbox.run_tests()
        assert result.success is False

    def test_uses_sys_executable(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["cmd"] = args[0]
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.run_tests()
        assert captured["cmd"][0] == sys.executable


# ---------------------------------------------------------------------------
# CodeSandbox — lint_code
# ---------------------------------------------------------------------------


class TestCodeSandboxLintCode:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return CodeSandbox(working_dir=str(tmp_path))

    def test_successful_lint(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(returncode=0)
            result = sandbox.lint_code("x = 1\n")
        assert result.success is True

    def test_lint_violations_detected(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(stdout="E501 line too long", returncode=1)
            result = sandbox.lint_code("x = 1  # " + "a" * 100)
        assert result.success is False

    def test_linter_not_found(self, sandbox):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = sandbox.lint_code("x = 1", linter="nonexistent_linter")
        assert result.success is False
        assert "nonexistent_linter" in result.error

    def test_default_linter_is_ruff(self, sandbox):
        captured = {}

        def capture_run(*args, **kwargs):
            captured["cmd"] = args[0]
            return _make_proc_result()

        with patch("subprocess.run", side_effect=capture_run):
            sandbox.lint_code("x = 1\n")
        assert captured["cmd"][0] == "ruff"

    def test_custom_linter(self, sandbox):
        with patch("subprocess.run") as mock_run:
            result = sandbox.lint_code("x = 1\n", linter="mypy")
        assert result.success is False
        assert "allowlist" in result.error
        mock_run.assert_not_called()

    def test_unallowlisted_linter_is_rejected(self, sandbox):
        result = sandbox.lint_code("x = 1\n", linter="flake8")
        assert result.success is False
        assert "allowlist" in result.error

    def test_lint_file_cleaned_up_after_success(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(returncode=0)
            sandbox.lint_code("x = 1\n")
        lint_files = list(sandbox.working_dir.glob("lint_*.py"))
        assert lint_files == []

    def test_lint_file_cleaned_up_after_error(self, sandbox):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _make_proc_result(returncode=1)
            sandbox.lint_code("x = 1\n")
        lint_files = list(sandbox.working_dir.glob("lint_*.py"))
        assert lint_files == []

    def test_exception_returns_failure(self, sandbox):
        with patch("subprocess.run", side_effect=RuntimeError("unexpected")):
            result = sandbox.lint_code("x = 1")
        assert result.success is False


# ---------------------------------------------------------------------------
# CodeSandbox — cleanup & get_stats
# ---------------------------------------------------------------------------


class TestCodeSandboxCleanup:
    def test_cleanup_removes_working_dir(self, tmp_path):
        work_dir = tmp_path / "sandbox"
        work_dir.mkdir()
        sb = CodeSandbox(working_dir=str(work_dir))
        sb.cleanup()
        assert not work_dir.exists()

    def test_cleanup_nonexistent_dir_does_not_raise(self, tmp_path):
        work_dir = tmp_path / "gone"
        sb = CodeSandbox(working_dir=str(work_dir))
        sb.cleanup()
        assert not work_dir.exists(), "Non-existent dir should remain non-existent after cleanup"

    def test_cleanup_logs_warning_on_failure(self, tmp_path):
        work_dir = tmp_path / "sandbox"
        work_dir.mkdir()
        sb = CodeSandbox(working_dir=str(work_dir))
        with patch("shutil.rmtree", side_effect=PermissionError("denied")):
            sb.cleanup()
        # Directory should still exist since rmtree was mocked to fail
        assert work_dir.exists(), "Directory should survive cleanup failure"


class TestCodeSandboxGetStats:
    def test_get_stats_returns_dict(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        stats = sb.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_contains_expected_keys(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        stats = sb.get_stats()
        assert "working_dir" in stats
        assert "execution_count" in stats
        assert "max_execution_time" in stats
        assert "max_memory_mb" in stats
        assert "allow_network" in stats

    def test_get_stats_values(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path), max_execution_time=30, max_memory_mb=128)
        stats = sb.get_stats()
        assert stats["max_execution_time"] == 30
        assert stats["max_memory_mb"] == 128
        assert stats["working_dir"] == str(tmp_path)
        assert stats["execution_count"] == 0


# ---------------------------------------------------------------------------
# CodeExecutor — __init__ and run
# ---------------------------------------------------------------------------


class TestCodeExecutorInit:
    def test_creates_sandbox_if_none(self):
        executor = CodeExecutor()
        assert isinstance(executor.sandbox, CodeSandbox)
        assert executor.sandbox.working_dir.exists()

    def test_uses_provided_sandbox(self, tmp_path):
        sb = CodeSandbox(working_dir=str(tmp_path))
        executor = CodeExecutor(sandbox=sb)
        assert executor.sandbox is sb


class TestCodeExecutorRun:
    @pytest.fixture
    def sandbox(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="py_out")
        sb.execute_shell.return_value = ExecutionResult(success=True, output="sh_out")
        return sb

    def test_python_language_routes_to_execute_python(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        result = executor.run("x = 1", language="python")
        sandbox.execute_python.assert_called_once_with("x = 1")
        assert result["output"] == "py_out"

    def test_shell_language_routes_to_execute_shell(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        executor.run("echo x", language="shell")
        sandbox.execute_shell.assert_called_once_with("echo x")

    def test_bash_language_routes_to_execute_shell(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        executor.run("echo x", language="bash")
        sandbox.execute_shell.assert_called_once_with("echo x")

    def test_sh_language_routes_to_execute_shell(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        executor.run("echo x", language="sh")
        sandbox.execute_shell.assert_called_once_with("echo x")

    def test_unsupported_language_returns_failure(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        result = executor.run("code", language="ruby")
        assert result["success"] is False
        assert "Unsupported language" in result["error"]

    def test_result_is_dict(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        result = executor.run("x = 1", language="python")
        assert isinstance(result, dict)

    def test_language_case_insensitive(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        result = executor.run("x = 1", language="PYTHON")
        sandbox.execute_python.assert_called_once_with("x = 1")
        assert result["success"] is True
        assert result["output"] == "py_out"

    def test_kwargs_forwarded_to_execute_python(self, sandbox):
        executor = CodeExecutor(sandbox=sandbox)
        executor.run("x = 1", language="python", input_data={"a": 1})
        sandbox.execute_python.assert_called_once_with("x = 1", input_data={"a": 1})


# ---------------------------------------------------------------------------
# CodeExecutor — run_and_validate
# ---------------------------------------------------------------------------


class TestCodeExecutorRunAndValidate:
    @pytest.fixture
    def executor_with_output(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="expected output here")
        return CodeExecutor(sandbox=sb)

    def test_expected_output_found(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1", expected_output="expected output")
        assert result["validations"]["expected_output"] is True

    def test_expected_output_not_found(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1", expected_output="missing string")
        assert result["validations"]["expected_output"] is False

    def test_error_pattern_not_in_output(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="clean output", error="")
        executor = CodeExecutor(sandbox=sb)
        result = executor.run_and_validate("x = 1", error_pattern="ERROR")
        assert result["validations"]["no_errors"] is True

    def test_error_pattern_found_in_output(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=False, output="ERROR happened", error="")
        executor = CodeExecutor(sandbox=sb)
        result = executor.run_and_validate("x = 1", error_pattern="ERROR")
        assert result["validations"]["no_errors"] is False

    def test_all_validations_passed_true(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1", expected_output="expected output")
        assert result["all_validations_passed"] is True

    def test_all_validations_passed_false_when_mismatch(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1", expected_output="not here")
        assert result["all_validations_passed"] is False

    def test_no_validations_all_pass_vacuously(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1")
        assert result["all_validations_passed"] is False
        assert result["validations"]["validation_checks_configured"] is False

    def test_result_contains_output_and_error(self, executor_with_output):
        result = executor_with_output.run_and_validate("x = 1")
        assert "output" in result
        assert "error" in result


# ---------------------------------------------------------------------------
# CodeExecutor — test_with_input
# ---------------------------------------------------------------------------


class TestCodeExecutorTestWithInput:
    def test_runs_for_each_input(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="")
        executor = CodeExecutor(sandbox=sb)
        results = executor.test_with_input("x = 1", [1, 2, 3])
        assert len(results) == 3

    def test_input_value_wrapped_in_dict(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="")
        executor = CodeExecutor(sandbox=sb)
        executor.test_with_input("x = INPUT_DATA['value']", [42])
        sb.execute_python.assert_called_once_with("x = INPUT_DATA['value']", input_data={"value": 42})

    def test_result_contains_input_and_result_keys(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        sb.execute_python.return_value = ExecutionResult(success=True, output="out")
        executor = CodeExecutor(sandbox=sb)
        results = executor.test_with_input("x = 1", ["hello"])
        assert results[0]["input"] == "hello"
        assert "result" in results[0]

    def test_empty_inputs_returns_empty_list(self, tmp_path):
        sb = MagicMock(spec=CodeSandbox)
        executor = CodeExecutor(sandbox=sb)
        results = executor.test_with_input("x = 1", [])
        assert results == []


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


class TestSingletons:
    def setup_method(self):
        """Reset global singleton before each test."""
        cs_module._code_executor = None

    def teardown_method(self):
        cs_module._code_executor = None

    def test_get_subprocess_executor_returns_code_executor(self):
        executor = get_subprocess_executor()
        assert isinstance(executor, CodeExecutor)

    def test_get_subprocess_executor_same_instance(self):
        e1 = get_subprocess_executor()
        e2 = get_subprocess_executor()
        assert e1 is e2

    def test_init_code_executor_creates_new(self, tmp_path):
        old = get_subprocess_executor()
        new = init_code_executor(working_dir=str(tmp_path))
        assert new is not old
        assert isinstance(new, CodeExecutor)

    def test_init_code_executor_replaces_global(self, tmp_path):
        init_code_executor(working_dir=str(tmp_path))
        assert cs_module._code_executor is not None
        assert isinstance(cs_module._code_executor, CodeExecutor)

    def test_init_code_executor_kwargs_forwarded(self, tmp_path):
        executor = init_code_executor(working_dir=str(tmp_path), max_execution_time=5)
        assert executor.sandbox.max_execution_time == 5


# ---------------------------------------------------------------------------
# SandboxManager policy wiring
# ---------------------------------------------------------------------------


class TestSandboxManagerPolicyWiring:
    """Verify SandboxManager loads configuration from SandboxPolicyConfig."""

    def test_loads_policy_from_real_yaml(self, tmp_path):
        """SandboxManager reads timeout and memory values from a YAML file."""
        yaml_content = """\
version: "1.0"
sandbox:
  subprocess:
    enabled: true
    timeout_seconds: 30
    max_memory_mb: 512
  external:
    enabled: true
    plugin_dir: "./plugins"
    isolation: "process"
    timeout_seconds: 99
    max_memory_mb: 777
    allowed_hooks: []
    blocked_hooks: []
    allowed_domains: []
    require_signature: false
    allow_network: true
    allow_file_write: true
    audit_enabled: true
    audit_log_dir: "./logs/sandbox"
    audit_retention_days: 30
rules:
  allow_code_execution: true
  require_approval_for_external: true
  allow_network: true
  blocked_domains: []
  allow_file_read: true
  allow_file_write: true
  blocked_paths: []
approval:
  auto_approve_builtin: true
  require_approval_for: []
"""
        policy_file = tmp_path / "sandbox_policy.yaml"
        policy_file.write_text(yaml_content, encoding="utf-8")

        manager = sm_module.SandboxManager(policy_path=policy_file)

        assert manager.external.timeout == 99
        assert manager.external.max_memory_mb == 777
        assert manager.policy.sandbox.external.timeout_seconds == 99

    def test_falls_back_to_defaults_when_file_missing(self, tmp_path):
        """SandboxManager uses built-in defaults when the policy file is absent."""
        from vetinari.config.sandbox_schema import SandboxPolicyConfig

        missing = tmp_path / "does_not_exist.yaml"
        manager = sm_module.SandboxManager(policy_path=missing)

        default_policy = SandboxPolicyConfig()
        assert manager.external.timeout == default_policy.sandbox.external.timeout_seconds
        assert manager.external.max_memory_mb == default_policy.sandbox.external.max_memory_mb

    def test_policy_attribute_exposed(self, tmp_path):
        """The loaded SandboxPolicyConfig is accessible as manager.policy."""
        from vetinari.config.sandbox_schema import SandboxPolicyConfig

        manager = sm_module.SandboxManager(policy_path=tmp_path / "absent.yaml")
        assert isinstance(manager.policy, SandboxPolicyConfig)

    def test_singleton_get_instance_thread_safe(self):
        """get_instance() returns the same object from concurrent threads."""
        # Reset the singleton so this test controls the instance.
        original = sm_module.SandboxManager._instance
        sm_module.SandboxManager._instance = None
        try:
            results: list[sm_module.SandboxManager] = []

            def _grab():
                results.append(sm_module.SandboxManager.get_instance())

            threads = [threading.Thread(target=_grab) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 8
            assert all(r is results[0] for r in results)
        finally:
            # Restore original singleton state.
            sm_module.SandboxManager._instance = original
