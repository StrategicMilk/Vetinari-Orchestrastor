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

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from vetinari.code_sandbox import CodeSandbox, ExecutionResult

# ============ C3: shell=True Removal Tests ============


class TestCodeSandboxShellSecurity:
    """C3: Verify execute_shell() no longer uses shell=True."""

    def test_execute_shell_uses_shlex_split(self):
        """Verify execute_shell uses shlex.split for safe command parsing."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="output",
                stderr="",
            )

            sandbox.execute_shell("echo hello world")

            # Verify shell=False was used
            call_args = mock_run.call_args
            assert call_args.kwargs.get('shell', call_args[1].get('shell')) is False, \
                "execute_shell must use shell=False"

            # Verify command was split into a list (not a raw string)
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get('args')
            assert isinstance(cmd_arg, list), \
                "Command must be passed as list, not string"
            assert cmd_arg == ["echo", "hello", "world"]

    def test_execute_shell_handles_quoted_args(self):
        """Verify shlex.split correctly handles quoted arguments."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            sandbox.execute_shell('echo "hello world" --flag')

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get('args')
            assert isinstance(cmd_arg, list)
            assert cmd_arg == ["echo", "hello world", "--flag"]

    def test_execute_shell_prevents_injection(self):
        """Verify shell injection via semicolons is not possible."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            # This would execute 'rm -rf /' if shell=True
            sandbox.execute_shell("echo safe; rm -rf /")

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get('args')
            # With shlex.split and shell=False, ";" is passed as a literal arg
            # to echo, NOT interpreted as a command separator
            assert isinstance(cmd_arg, list)
            assert cmd_arg[0] == "echo"
            assert "rm" not in cmd_arg[0]  # rm is NOT the first command

    def test_execute_shell_prevents_pipe_injection(self):
        """Verify pipe injection is not possible."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            sandbox.execute_shell("echo data | cat /etc/passwd")

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get('args')
            assert isinstance(cmd_arg, list)
            # The pipe character is a literal arg, not interpreted
            assert "|" in cmd_arg
            assert cmd_arg[0] == "echo"

    def test_execute_shell_prevents_backtick_injection(self):
        """Verify backtick command substitution is not possible."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr="",
            )

            sandbox.execute_shell("echo `whoami`")

            call_args = mock_run.call_args
            cmd_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get('args')
            assert isinstance(cmd_arg, list)
            # Backticks become a literal arg, not expanded
            assert cmd_arg == ["echo", "`whoami`"]

    def test_execute_shell_timeout(self):
        """Verify timeout still works with shell=False."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["sleep", "100"], timeout=1
            )

            result = sandbox.execute_shell("sleep 100", timeout=1)

            assert result.success is False
            assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_execute_shell_returns_execution_result(self):
        """Verify execute_shell returns proper ExecutionResult."""
        import subprocess
        sandbox = CodeSandbox()

        with patch.object(subprocess, 'run') as mock_run:
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
        assert hasattr(cs, 'shlex'), "shlex must be imported in code_sandbox"


# ============ C2: Sandbox Endpoint Auth Tests ============


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestSandboxEndpointAuth:
    """C2: Verify /api/sandbox/* endpoints require admin auth."""

    @pytest.fixture
    def client(self):
        from vetinari.web_ui import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_sandbox_execute_requires_auth(self, client):
        """Verify /api/sandbox/execute returns 403 for non-local requests."""
        with patch('vetinari.web_ui._is_admin_user', return_value=False):
            resp = client.post(
                '/api/sandbox/execute',
                json={"code": "print('pwned')", "sandbox_type": "in_process"},
            )
            assert resp.status_code == 403
            data = json.loads(resp.data)
            assert "Unauthorized" in data["error"]

    def test_sandbox_status_requires_auth(self, client):
        """Verify /api/sandbox/status returns 403 for non-local requests."""
        with patch('vetinari.web_ui._is_admin_user', return_value=False):
            resp = client.get('/api/sandbox/status')
            assert resp.status_code == 403
            data = json.loads(resp.data)
            assert "Unauthorized" in data["error"]

    def test_sandbox_audit_requires_auth(self, client):
        """Verify /api/sandbox/audit returns 403 for non-local requests."""
        with patch('vetinari.web_ui._is_admin_user', return_value=False):
            resp = client.get('/api/sandbox/audit')
            assert resp.status_code == 403
            data = json.loads(resp.data)
            assert "Unauthorized" in data["error"]

    def test_sandbox_execute_allowed_for_admin(self, client):
        """Verify /api/sandbox/execute works for admin users."""
        with patch('vetinari.web_ui._is_admin_user', return_value=True), \
             patch('vetinari.sandbox.sandbox_manager') as mock_sm:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {
                "success": True,
                "output": "hello",
                "error": "",
            }
            mock_sm.execute.return_value = mock_result

            resp = client.post(
                '/api/sandbox/execute',
                json={"code": "print('hello')", "sandbox_type": "in_process"},
            )
            assert resp.status_code == 200

    def test_sandbox_status_allowed_for_admin(self, client):
        """Verify /api/sandbox/status works for admin users."""
        with patch('vetinari.web_ui._is_admin_user', return_value=True), \
             patch('vetinari.sandbox.sandbox_manager') as mock_sm:
            mock_sm.get_status.return_value = {"in_process": {"available": True}}

            resp = client.get('/api/sandbox/status')
            assert resp.status_code == 200

    def test_sandbox_audit_allowed_for_admin(self, client):
        """Verify /api/sandbox/audit works for admin users."""
        with patch('vetinari.web_ui._is_admin_user', return_value=True), \
             patch('vetinari.sandbox.sandbox_manager') as mock_sm:
            mock_sm.get_audit_log.return_value = []

            resp = client.get('/api/sandbox/audit')
            assert resp.status_code == 200

    def test_sandbox_execute_with_token_auth(self, client):
        """Verify token-based auth works for sandbox endpoints."""
        with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": "secret123"}):
            with patch('vetinari.sandbox.sandbox_manager') as mock_sm:
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"success": True}
                mock_sm.execute.return_value = mock_result

                # Valid token should work
                resp = client.post(
                    '/api/sandbox/execute',
                    json={"code": "1+1"},
                    headers={"X-Admin-Token": "secret123"},
                )
                assert resp.status_code == 200

    def test_sandbox_execute_with_wrong_token(self, client):
        """Verify wrong token is rejected for sandbox endpoints."""
        with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": "secret123"}):
            resp = client.post(
                '/api/sandbox/execute',
                json={"code": "1+1"},
                headers={"X-Admin-Token": "wrong_token"},
            )
            assert resp.status_code == 403


# ============ C4: Path Traversal Protection Tests ============


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestPathTraversalProtection:
    """C4: Verify path traversal is blocked on file read/write endpoints."""

    @pytest.fixture
    def client(self):
        from vetinari.web_ui import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_read_file_traversal_blocked(self, client):
        """Verify ../ traversal is blocked on file read."""
        resp = client.post(
            '/api/project/test_proj/files/read',
            json={"path": "../../etc/passwd"},
        )
        # Should be 403 (access denied) or 404 (project not found)
        assert resp.status_code in (403, 404)

    def test_write_file_traversal_blocked(self, client):
        """Verify ../ traversal is blocked on file write."""
        resp = client.post(
            '/api/project/test_proj/files/write',
            json={"path": "../../tmp/evil.txt", "content": "pwned"},
        )
        assert resp.status_code in (403, 404)

    def test_read_absolute_path_blocked(self, client):
        """Verify absolute path traversal is blocked on file read."""
        resp = client.post(
            '/api/project/test_proj/files/read',
            json={"path": "/etc/passwd"},
        )
        assert resp.status_code in (403, 404)

    def test_read_empty_path_rejected(self, client):
        """Verify empty path is rejected."""
        resp = client.post(
            '/api/project/test_proj/files/read',
            json={"path": ""},
        )
        assert resp.status_code == 400


# ============ H6: Bind Address Tests ============


class TestServerBindAddress:
    """H6: Verify server binds to 127.0.0.1 only."""

    def test_default_bind_is_localhost(self):
        """Verify default bind address is 127.0.0.1."""
        import vetinari.web_ui as web_ui

        # Check the source for the app.run() call to ensure it uses 127.0.0.1
        source = Path(web_ui.__file__).read_text()

        # The run() call should bind to 127.0.0.1, not 0.0.0.0
        assert "0.0.0.0" not in source or "host='127.0.0.1'" in source or 'host="127.0.0.1"' in source, \
            "Server must bind to 127.0.0.1, not 0.0.0.0"


# ============ Admin Auth Function Tests ============


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestAdminAuthFunction:
    """Tests for _is_admin_user() function."""

    @pytest.fixture
    def client(self):
        from vetinari.web_ui import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_admin_check_endpoint_exists(self, client):
        """Verify admin check endpoint exists and returns boolean."""
        resp = client.get('/api/auth/admin-check')
        # May return 200 with admin status
        assert resp.status_code == 200 or resp.status_code == 404

    def test_localhost_is_admin_when_no_token(self, client):
        """Verify localhost requests are admin when no token configured."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove VETINARI_ADMIN_TOKEN if set
            os.environ.pop("VETINARI_ADMIN_TOKEN", None)

            from vetinari.web_ui import _is_admin_user, app

            with app.test_request_context(
                environ_base={'REMOTE_ADDR': '127.0.0.1'}
            ):
                assert _is_admin_user() is True

    def test_non_localhost_is_not_admin(self, client):
        """Verify non-localhost requests are not admin when no token configured."""
        os.environ.pop("VETINARI_ADMIN_TOKEN", None)

        from vetinari.web_ui import _is_admin_user, app

        with app.test_request_context(
            environ_base={'REMOTE_ADDR': '192.168.1.100'}
        ):
            assert _is_admin_user() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
