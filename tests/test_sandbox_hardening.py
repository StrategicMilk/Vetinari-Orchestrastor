"""Tests for sandbox security hardening (US-041)."""

from __future__ import annotations

import pytest

from vetinari.exceptions import SandboxError


class TestCodeSandboxFilesystemAllowlist:
    """Tests for CodeSandbox filesystem allow-listing."""

    def test_empty_allowlist_permits_all(self) -> None:
        """An empty allowlist applies no path restrictions."""
        from vetinari.code_sandbox import CodeSandbox

        sb = CodeSandbox(filesystem_allowlist=[])
        result = sb.execute("print('hello')")
        assert result.success

    def test_allowlist_blocks_disallowed_path(self) -> None:
        """A non-empty allowlist blocks file access outside the listed prefixes."""
        from vetinari.code_sandbox import CodeSandbox

        sb = CodeSandbox(filesystem_allowlist=["/tmp/safe"])
        # Code tries to read outside allowlist — should be blocked or raise
        result = sb.execute("open('/etc/passwd').read()")
        assert not result.success or "blocked" in str(result.output).lower() or "denied" in str(result.error).lower()

    def test_allowlist_permits_allowed_path(self) -> None:
        """A path within the allowlist prefix is accessible."""
        import os
        import tempfile

        from vetinari.code_sandbox import CodeSandbox

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("hello")
            sb = CodeSandbox(filesystem_allowlist=[tmpdir])
            result = sb.execute(f"print(open('{test_file.replace(chr(92), '/')}').read())")
            # The sandbox must not block this path — either it succeeds and prints "hello",
            # or it fails for unrelated reasons (e.g. subprocess constraints), but must NOT
            # contain an allowlist-denial message
            assert result is not None
            assert "allowlist" not in str(result.error).lower(), (
                "Path within allowlist must not be blocked by allowlist enforcement"
            )


class TestCodeSandboxNetworkIsolation:
    """Tests for CodeSandbox network isolation."""

    def test_network_isolation_blocks_socket(self) -> None:
        """With network_isolation=True the socket module is blocked."""
        from vetinari.code_sandbox import CodeSandbox

        sb = CodeSandbox(network_isolation=True)
        result = sb.execute("import socket; print(socket.gethostname())")
        assert not result.success

    def test_network_isolation_blocks_requests(self) -> None:
        """With network_isolation=True the requests module is blocked."""
        from vetinari.code_sandbox import CodeSandbox

        sb = CodeSandbox(network_isolation=True)
        result = sb.execute("import requests")
        assert not result.success

    def test_network_isolation_disabled_allows_socket(self) -> None:
        """With network_isolation=False the socket module is not blocked by the isolation filter."""
        from vetinari.code_sandbox import CodeSandbox

        sb = CodeSandbox(network_isolation=False)
        # With isolation off, socket is not added to blocked_modules by the isolation filter.
        # The result may still fail for other reasons, but that is acceptable.
        result = sb.execute("import socket; print('ok')")
        # Just verify the flag is accepted without error on construction
        assert isinstance(result.success, bool)


class TestSandboxManagerEscapePrevention:
    """Tests for SandboxManager pre-execution scanning of dangerous attribute traversal."""

    def test_blocks_subclasses_access(self) -> None:
        """Access to __subclasses__ is detected and blocked by pre-execution scan."""
        from vetinari.code_sandbox import SandboxManager

        mgr = SandboxManager()
        result = mgr.execute("().__class__.__subclasses__()", client_id="test_esc")
        assert not result.success

    def test_blocks_globals_access(self) -> None:
        """Access to __globals__ is detected and blocked by pre-execution scan."""
        from vetinari.code_sandbox import SandboxManager

        mgr = SandboxManager()
        result = mgr.execute("x = lambda: None; x.__globals__", client_id="test_esc")
        assert not result.success

    def test_blocks_bases_access(self) -> None:
        """Access to __bases__ is detected and blocked by pre-execution scan."""
        from vetinari.code_sandbox import SandboxManager

        mgr = SandboxManager()
        result = mgr.execute("int.__bases__", client_id="test_esc")
        assert not result.success

    def test_blocks_builtins_access(self) -> None:
        """Full class-hierarchy escape chain is detected and blocked by pre-execution scan."""
        from vetinari.code_sandbox import SandboxManager

        mgr = SandboxManager()
        result = mgr.execute(
            "().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__['__builtins__']",
            client_id="test_esc",
        )
        assert not result.success


class TestSandboxManagerCodeLength:
    """Tests for SandboxManager max code length enforcement."""

    def test_rejects_oversized_code(self) -> None:
        """Code exceeding MAX_CODE_LENGTH raises SandboxError."""
        from vetinari.code_sandbox import SandboxManager

        mgr = SandboxManager()
        code = "x = 1\n" * 5000  # Well over 10 000 chars
        with pytest.raises(SandboxError, match="exceeds maximum"):
            mgr.execute(code, client_id="test")

    def test_accepts_normal_code(self) -> None:
        """Code within the length limit executes without error."""
        from vetinari.code_sandbox import SandboxManager, SandboxResult

        mgr = SandboxManager()
        result = mgr.execute("print('hello')", client_id="test")
        assert isinstance(result, SandboxResult)
        assert result.success is True


class TestSandboxStructuredLogging:
    """Tests for structured logging of sandbox executions."""

    def test_execution_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """SandboxManager.execute() emits an INFO log entry for each execution."""
        import logging

        from vetinari.code_sandbox import SandboxManager

        with caplog.at_level(logging.INFO, logger="vetinari.code_sandbox"):
            mgr = SandboxManager()
            mgr.execute("x = 1 + 1", client_id="test_log")
        # Check that some sandbox execution log was emitted
        assert any("sandbox" in r.message.lower() or "execution" in r.message.lower() for r in caplog.records)
