"""
Security hardening tests for Batch 2 critical security fixes.

Tests cover:
- C2: Auth guard on /api/sandbox/* endpoints (execute, status, audit)
- C3: shell=True removed from code_sandbox.py execute_shell()
- C4: Path traversal protection on file read/write endpoints
- H6: Server binds to 127.0.0.1 only

References master plan Phase 2b.5 security findings.
"""

import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.code_sandbox import CodeSandbox, ExecutionResult

# ============ C3: shell=True Removal Tests ============


class TestCodeSandboxShellSecurity:
    """C3: Verify execute_shell() no longer uses shell=True."""

    def test_execute_shell_uses_shlex_split(self):
        """Verify execute_shell uses shlex.split for safe command parsing."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="output",
                stderr="",
            )

            sandbox.execute_shell("echo hello world")

            # Verify shell=False was used
            call_args = mock_run.call_args
            assert call_args.kwargs.get("shell", call_args[1].get("shell")) is False, (
                "execute_shell must use shell=False"
            )

            # Verify command was split into a list (not a raw string)
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("args")
            assert isinstance(cmd_arg, list), "Command must be passed as list, not string"
            assert cmd_arg == ["echo", "hello", "world"]

    def test_execute_shell_handles_quoted_args(self):
        """Verify shlex.split correctly handles quoted arguments."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            sandbox.execute_shell('echo "hello world" --flag')

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("args")
            assert isinstance(cmd_arg, list)
            assert cmd_arg == ["echo", "hello world", "--flag"]

    def test_execute_shell_prevents_injection(self):
        """Verify shell injection via semicolons is not possible."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            # This would execute 'rm -rf /' if shell=True
            sandbox.execute_shell("echo safe; rm -rf /")

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("args")
            # With shlex.split and shell=False, ";" is passed as a literal arg
            # to echo, NOT interpreted as a command separator
            assert isinstance(cmd_arg, list)
            assert cmd_arg[0] == "echo"
            assert "rm" not in cmd_arg[0]  # rm is NOT the first command

    def test_execute_shell_prevents_pipe_injection(self):
        """Verify pipe injection is not possible.

        The command ``echo data | cat /etc/passwd`` contains ``/etc/passwd`` as
        an absolute path outside the sandbox root.  The path-arg guard rejects
        it before subprocess.run is ever called, so the result is a failure
        rather than a successful (but unintended) execution.
        """
        sandbox = CodeSandbox()

        result = sandbox.execute_shell("echo data | cat /etc/passwd")

        # The path-arg guard must have blocked the command before it ran.
        assert result.success is False
        assert result.output == ""

    def test_execute_shell_prevents_backtick_injection(self):
        """Verify backtick command substitution is not possible."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            sandbox.execute_shell("echo `whoami`")

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("args")
            assert isinstance(cmd_arg, list)
            # Backticks become a literal arg, not expanded
            assert cmd_arg == ["echo", "`whoami`"]

    def test_execute_shell_timeout(self):
        """Verify timeout still works with shell=False."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["python", "-c", "import time; time.sleep(100)"], timeout=1
            )

            result = sandbox.execute_shell("python -c 'import time; time.sleep(100)'", timeout=1)

            assert result.success is False
            assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_execute_shell_returns_execution_result(self):
        """Verify execute_shell returns proper ExecutionResult."""
        import subprocess

        sandbox = CodeSandbox()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="hello\n",
                stderr="",
            )

            result = sandbox.execute_shell("echo hello")

            assert isinstance(result, ExecutionResult)
            assert result.success is True
            assert result.output == "hello\n"
            assert result.error == ""

    def test_shlex_import_exists(self):
        """Verify shlex is imported in code_sandbox module."""
        import vetinari.code_sandbox as cs

        assert hasattr(cs, "shlex"), "shlex must be imported in code_sandbox"
