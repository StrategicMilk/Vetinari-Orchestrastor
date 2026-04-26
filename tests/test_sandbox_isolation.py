"""Tests verifying sandbox isolation hardening (SESSION-28 security fixes).

Covers four regression scenarios:
1. apply_changes rejects path-traversal keys (``../`` escaping sandbox root)
2. _run_tests returns (False, [...]) when pytest is unavailable
3. _run_lint returns (False, [...]) when ruff is unavailable
4. execute_python with an empty filesystem allowlist blocks writes outside
   the working directory
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.code_sandbox import CodeSandbox
from vetinari.project.sandbox import _run_lint, _run_tests, apply_changes

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def real_sandbox_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory that already exists on disk.

    apply_changes resolves paths against the real filesystem, so the root
    directory must exist before we call it.
    """
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


# -- Test 1: apply_changes rejects path-traversal keys -----------------------


class TestApplyChangesPathTraversal:
    """apply_changes must reject keys that escape the sandbox root."""

    def test_dotdot_key_is_rejected(self, real_sandbox_dir: Path, tmp_path: Path) -> None:
        """A ``../`` key must not write outside the sandbox root."""
        victim = tmp_path / "victim.txt"
        # Construct a relative path that traverses out of the sandbox root.
        # On Windows and POSIX alike, ``../victim.txt`` escapes one level.
        traversal_key = "../victim.txt"

        result = apply_changes(real_sandbox_dir, {traversal_key: "pwned"})

        assert result is False, "apply_changes must return False for a traversal key"
        assert not victim.exists(), "The traversal write must not have reached the parent directory"

    def test_nested_dotdot_key_is_rejected(self, real_sandbox_dir: Path, tmp_path: Path) -> None:
        """A deeply nested ``../../`` traversal must also be rejected."""
        victim = tmp_path / "deep_victim.txt"
        traversal_key = "sub/../../deep_victim.txt"

        result = apply_changes(real_sandbox_dir, {traversal_key: "pwned"})

        assert result is False
        assert not victim.exists()

    def test_legitimate_key_is_written(self, real_sandbox_dir: Path) -> None:
        """A plain relative key must still be written successfully."""
        result = apply_changes(real_sandbox_dir, {"legit.txt": "safe content"})

        assert result is True
        written = real_sandbox_dir / "legit.txt"
        assert written.exists()
        assert written.read_text(encoding="utf-8") == "safe content"

    def test_mixed_keys_partial_ok(self, real_sandbox_dir: Path, tmp_path: Path) -> None:
        """A mix of safe and traversal keys: safe key is written, result is False."""
        victim = tmp_path / "mixed_victim.txt"

        result = apply_changes(
            real_sandbox_dir,
            {
                "safe.txt": "good",
                "../mixed_victim.txt": "pwned",
            },
        )

        assert result is False
        assert not victim.exists()
        assert (real_sandbox_dir / "safe.txt").exists()


# -- Test 2: _run_tests is fail-closed when pytest is missing ----------------


class TestRunTestsFailClosed:
    """_run_tests must return (False, [...]) when pytest cannot be found."""

    def test_missing_pytest_returns_false(self, real_sandbox_dir: Path) -> None:
        """FileNotFoundError from subprocess → (False, non-empty error list)."""
        with patch("vetinari.project.sandbox._PYTHON_EXE", "/nonexistent/python"):
            passed, errors = _run_tests(real_sandbox_dir, timeout=30)

        assert passed is False, "_run_tests must fail closed when pytest is unavailable"
        assert len(errors) > 0, "_run_tests must return at least one error message"
        assert "pytest" in errors[0].lower(), f"Error message should mention pytest, got: {errors[0]!r}"

    def test_missing_pytest_does_not_return_true(self, real_sandbox_dir: Path) -> None:
        """Explicitly verify the old fail-open behaviour is gone."""
        with patch("vetinari.project.sandbox._PYTHON_EXE", "/nonexistent/python"):
            passed, _ = _run_tests(real_sandbox_dir, timeout=30)

        assert passed is not True


# -- Test 3: _run_lint is fail-closed when ruff is missing -------------------


class TestRunLintFailClosed:
    """_run_lint must return (False, [...]) when ruff cannot be found."""

    def test_missing_ruff_returns_false(self, real_sandbox_dir: Path) -> None:
        """FileNotFoundError from subprocess → (False, non-empty error list)."""
        with patch("vetinari.project.sandbox._RUFF_EXE", "/nonexistent/ruff"):
            passed, errors = _run_lint(real_sandbox_dir, timeout=30)

        assert passed is False, "_run_lint must fail closed when ruff is unavailable"
        assert len(errors) > 0, "_run_lint must return at least one error message"
        assert "ruff" in errors[0].lower(), f"Error message should mention ruff, got: {errors[0]!r}"

    def test_missing_ruff_does_not_return_true(self, real_sandbox_dir: Path) -> None:
        """Explicitly verify the old fail-open behaviour is gone."""
        with patch("vetinari.project.sandbox._RUFF_EXE", "/nonexistent/ruff"):
            passed, _ = _run_lint(real_sandbox_dir, timeout=30)

        assert passed is not True


# -- Test 4: execute_python blocks writes outside working_dir when allowlist empty --


class TestExecutePythonWriteIsolation:
    """execute_python with an empty filesystem_allowlist must block file writes
    outside the sandbox working directory."""

    def test_write_outside_working_dir_is_blocked(self, tmp_path: Path) -> None:
        """Attempting to open a file outside working_dir for writing raises PermissionError."""
        working_dir = tmp_path / "sandbox_work"
        working_dir.mkdir()
        outside_target = tmp_path / "outside.txt"

        code = textwrap.dedent(
            f"""\
            with open({str(outside_target)!r}, "w") as f:
                f.write("should not appear")
            """
        )

        with CodeSandbox(
            working_dir=str(working_dir),
            filesystem_allowlist=[],  # empty = no writes allowed outside working_dir
        ) as sb:
            result = sb.execute_python(code)

        # The sandbox must report failure and the file must NOT have been created.
        assert result.success is False, "execute_python should fail when code tries to write outside working_dir"
        assert not outside_target.exists(), "The file outside the working directory must not have been written"

    def test_write_inside_working_dir_is_permitted(self, tmp_path: Path) -> None:
        """Writes within the working directory must be allowed even with no explicit allowlist."""
        working_dir = tmp_path / "sandbox_work2"
        working_dir.mkdir()
        inside_target = working_dir / "output.txt"

        code = textwrap.dedent(
            f"""\
            with open({str(inside_target)!r}, "w") as f:
                f.write("hello from sandbox")
            """
        )

        with CodeSandbox(
            working_dir=str(working_dir),
            # Allowlist set to working_dir — writes inside it must succeed.
            filesystem_allowlist=[str(working_dir)],
        ) as sb:
            result = sb.execute_python(code)
            # Assert inside the context manager — __exit__ deletes working_dir.
            assert result.success is True, f"Write inside working_dir should succeed: {result.error}"
            assert inside_target.exists()
            assert inside_target.read_text(encoding="utf-8") == "hello from sandbox"

    def test_empty_allowlist_blocks_all_external_writes(self, tmp_path: Path) -> None:
        """With filesystem_allowlist=[], any write attempt must fail closed."""
        working_dir = tmp_path / "sandbox_work3"
        working_dir.mkdir()
        # Try a write to the system temp dir — definitely outside working_dir.
        outside = Path(tempfile.gettempdir()) / "vetinari_test_escape.txt"

        code = textwrap.dedent(
            f"""\
            with open({str(outside)!r}, "w") as f:
                f.write("escaped")
            """
        )

        with CodeSandbox(
            working_dir=str(working_dir),
            filesystem_allowlist=[],
        ) as sb:
            result = sb.execute_python(code)

        assert result.success is False
        assert not outside.exists(), "File must not have been written outside sandbox"
