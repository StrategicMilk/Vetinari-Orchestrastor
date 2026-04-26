"""Tests for vetinari.verification.sandbox_verifier.

Verifies that the sandbox-based code verification layer correctly:
- Detects syntax errors before any subprocess is spawned
- Resolves or rejects imports via importlib
- Rejects code that raises at runtime when run_tests=True
- Returns bounded SandboxVerification objects (never raw exceptions)
- Cleans up orphaned script_*.py artifacts
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.verification.sandbox_verifier import (
    SandboxVerification,
    cleanup_sandbox_artifacts,
    verify_code,
    verify_code_safe,
)

# ── Syntax check tests ───────────────────────────────────────────────────────


class TestSyntaxCheck:
    """Tests for the ast.parse-based syntax validation step."""

    def test_valid_python_passes_syntax(self) -> None:
        """Simple valid Python should pass syntax check and have syntax_valid=True."""
        result = verify_code("x = 1\nprint(x)")
        assert result.syntax_valid is True
        assert "syntax" in result.checks_run

    def test_invalid_python_fails_syntax(self) -> None:
        """Incomplete function definition must fail with syntax_valid=False."""
        result = verify_code("def f(\n")
        assert result.passed is False
        assert result.syntax_valid is False
        assert result.error_message is not None
        assert "syntax" in result.error_message.lower() or "SyntaxError" in result.error_message

    def test_syntax_failure_short_circuits(self) -> None:
        """When syntax fails, imports check should NOT be attempted."""
        result = verify_code("def f(\n")
        # imports check is skipped — only 'syntax' should be in checks_run
        assert "imports" not in result.checks_run

    def test_empty_code_passes_syntax(self) -> None:
        """Empty string is syntactically valid Python."""
        result = verify_code("")
        assert result.syntax_valid is True

    def test_multiline_valid_passes(self) -> None:
        """Multi-line class definition should pass syntax check."""
        code = "class Foo:\n    def bar(self):\n        return 42\n"
        result = verify_code(code)
        assert result.syntax_valid is True


# ── Import check tests ───────────────────────────────────────────────────────


class TestImportCheck:
    """Tests for the importlib-based import resolution step."""

    def test_safe_stdlib_imports_valid(self) -> None:
        """Standard library imports allowed by sandbox policy must resolve."""
        code = "import json\nimport pathlib\nx = pathlib.Path('.')"
        result = verify_code(code)
        assert result.imports_valid is True
        assert result.passed is True
        assert "imports" in result.checks_run

    def test_sandbox_blocked_import_rejected(self) -> None:
        """Imports blocked by the runtime sandbox are rejected even if installed."""
        result = verify_code("import os\n")
        assert result.passed is False
        assert result.imports_valid is False
        assert result.error_message is not None
        assert "blocked by sandbox policy" in result.error_message

    def test_nonexistent_import_flagged(self) -> None:
        """An import of a package that does not exist must fail imports_valid."""
        code = "import nonexistent_fake_pkg_xyzzy\nprint('hello')"
        result = verify_code(code)
        assert result.passed is False
        assert result.imports_valid is False
        assert result.error_message is not None
        assert "nonexistent_fake_pkg_xyzzy" in result.error_message

    def test_multiple_imports_all_valid(self) -> None:
        """Multiple valid stdlib imports should all resolve."""
        code = "import json\nimport pathlib\nimport logging\n"
        result = verify_code(code)
        assert result.imports_valid is True

    def test_from_import_checked(self) -> None:
        """from-import style is also verified against importlib."""
        code = "from pathlib import Path\n"
        result = verify_code(code)
        assert result.imports_valid is True

    def test_from_nonexistent_import_flagged(self) -> None:
        """from nonexistent_pkg import X must fail imports_valid."""
        code = "from totally_fake_package_abc123 import SomeClass\n"
        result = verify_code(code)
        assert result.passed is False
        assert result.imports_valid is False


# ── Sandbox execution tests ──────────────────────────────────────────────────


class TestSandboxExecution:
    """Tests for the CodeSandbox subprocess execution step (run_tests=True)."""

    def test_inspector_rejects_sandbox_failure(self) -> None:
        """Code that raises at runtime must return passed=False with error info."""
        code = "raise ValueError('intentional test failure')"
        result = verify_code(code, run_tests=True, timeout=15)
        assert result.passed is False
        assert result.error_message is not None
        # The error message must contain information about the runtime failure
        assert len(result.error_message) > 0
        assert "execution" in result.checks_run

    def test_successful_execution_returns_output(self) -> None:
        """Code with a print statement should run and capture output."""
        code = "print('hello from sandbox')"
        result = verify_code(code, run_tests=True, timeout=15)
        assert result.passed is True
        assert result.execution_result is not None
        assert "hello from sandbox" in result.execution_result
        assert "execution" in result.checks_run

    def test_execution_skipped_when_run_tests_false(self) -> None:
        """When run_tests=False, execution check should not appear in checks_run."""
        code = "x = 1 + 1"
        result = verify_code(code, run_tests=False)
        assert "execution" not in result.checks_run
        assert result.execution_result is None

    def test_runtime_error_does_not_raise(self) -> None:
        """verify_code must return a result (not raise) even when code raises."""
        code = "raise RuntimeError('boom')"
        # Must not raise
        result = verify_code(code, run_tests=True, timeout=15)
        assert isinstance(result, SandboxVerification)
        assert result.passed is False


# ── Failure bounding tests ───────────────────────────────────────────────────


class TestFailureBounding:
    """Tests that OS-level exceptions are bounded to SandboxVerification objects."""

    def test_permission_error_returns_bounded_failure(self) -> None:
        """A PermissionError from the sandbox layer must be caught by verify_code_safe."""
        with patch(
            "vetinari.verification.sandbox_verifier.verify_code",
            side_effect=PermissionError("Access denied to temp directory"),
        ):
            result = verify_code_safe("x = 1")
        assert isinstance(result, SandboxVerification)
        assert result.passed is False
        assert result.error_message is not None
        assert "permission" in result.error_message.lower()

    def test_oserror_returns_bounded_failure(self) -> None:
        """An OSError from the sandbox layer must also be caught by verify_code_safe."""
        with patch(
            "vetinari.verification.sandbox_verifier.verify_code",
            side_effect=OSError("disk full"),
        ):
            result = verify_code_safe("x = 1")
        assert isinstance(result, SandboxVerification)
        assert result.passed is False

    def test_any_exception_returns_bounded_failure(self) -> None:
        """verify_code_safe must catch all exception types and return a failure result."""
        with patch(
            "vetinari.verification.sandbox_verifier.verify_code",
            side_effect=RuntimeError("unexpected internal error"),
        ):
            result = verify_code_safe("x = 1")
        assert isinstance(result, SandboxVerification)
        assert result.passed is False
        assert result.error_message is not None


# ── Cleanup tests ────────────────────────────────────────────────────────────


class TestCleanup:
    """Tests for cleanup_sandbox_artifacts."""

    def test_cleanup_removes_script_files(self) -> None:
        """Orphaned script_*.py files should be removed and the count returned."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create fake orphaned script files
            (tmp_path / "script_abc123.py").write_text("x = 1", encoding="utf-8")
            (tmp_path / "script_def456.py").write_text("y = 2", encoding="utf-8")
            # Create a non-script file that should NOT be removed
            (tmp_path / "other_file.py").write_text("z = 3", encoding="utf-8")

            removed = cleanup_sandbox_artifacts(tmp_path)

        assert removed == 2

    def test_cleanup_leaves_non_script_files(self) -> None:
        """Files not matching script_*.py must be left untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "script_aaa.py").write_text("x = 1", encoding="utf-8")
            keeper = tmp_path / "important.py"
            keeper.write_text("z = 3", encoding="utf-8")

            cleanup_sandbox_artifacts(tmp_path)

            assert keeper.exists()

    def test_cleanup_returns_zero_for_nonexistent_directory(self) -> None:
        """cleanup_sandbox_artifacts on a missing directory must return 0, not raise."""
        result = cleanup_sandbox_artifacts(Path("/nonexistent/path/that/does/not/exist"))
        assert result == 0

    def test_cleanup_returns_zero_when_no_scripts(self) -> None:
        """cleanup_sandbox_artifacts returns 0 when no script_*.py files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "other.txt").write_text("data", encoding="utf-8")
            removed = cleanup_sandbox_artifacts(tmp_path)
        assert removed == 0


# ── Safe wrapper tests ───────────────────────────────────────────────────────


class TestSafeWrapper:
    """Tests for verify_code_safe's never-raise contract."""

    def test_verify_code_safe_never_raises(self) -> None:
        """verify_code_safe must return passed=False even when verify_code raises."""
        with patch(
            "vetinari.verification.sandbox_verifier.verify_code",
            side_effect=Exception("something completely unexpected"),
        ):
            result = verify_code_safe("print('hello')")
        assert isinstance(result, SandboxVerification)
        assert result.passed is False

    def test_verify_code_safe_passes_through_on_success(self) -> None:
        """verify_code_safe should return the same result as verify_code on success."""
        code = "x = 1\n"
        safe_result = verify_code_safe(code)
        direct_result = verify_code(code)
        assert safe_result.passed == direct_result.passed
        assert safe_result.syntax_valid == direct_result.syntax_valid
        assert safe_result.imports_valid == direct_result.imports_valid

    def test_verify_code_safe_kwargs_forwarded(self) -> None:
        """verify_code_safe must forward filename and run_tests kwargs."""
        mock_result = SandboxVerification(
            passed=True,
            syntax_valid=True,
            imports_valid=True,
            execution_result=None,
            error_message=None,
            checks_run=("syntax", "imports"),
        )
        with patch(
            "vetinari.verification.sandbox_verifier.verify_code",
            return_value=mock_result,
        ) as mock_verify:
            verify_code_safe("x = 1", filename="test.py", run_tests=False)
        mock_verify.assert_called_once_with("x = 1", filename="test.py", run_tests=False)


# ── Wiring / import tests ────────────────────────────────────────────────────


class TestModuleWiring:
    """Tests that the public API is correctly wired and importable."""

    def test_imports_from_verification_package(self) -> None:
        """All public symbols must be importable from vetinari.verification."""
        from vetinari.verification import (
            SandboxFailure,
            SandboxVerification,
            cleanup_sandbox_artifacts,
            verify_code,
            verify_code_safe,
        )

        assert SandboxFailure is not None
        assert SandboxVerification is not None
        assert cleanup_sandbox_artifacts is not None
        assert verify_code is not None
        assert verify_code_safe is not None

    def test_sandbox_verification_is_frozen(self) -> None:
        """SandboxVerification must be a frozen dataclass (immutable)."""
        result = verify_code("x = 1")
        with pytest.raises((AttributeError, TypeError)):
            result.passed = False  # type: ignore[misc]

    def test_sandbox_failure_is_frozen(self) -> None:
        """SandboxFailure must be a frozen dataclass (immutable)."""
        from vetinari.verification.sandbox_verifier import SandboxFailure

        failure = SandboxFailure(
            error_type="permission_error",
            message="Access denied",
            recoverable=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            failure.recoverable = False  # type: ignore[misc]
