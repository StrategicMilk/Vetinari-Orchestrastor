"""Tests for vetinari.project.sandbox.

Verifies that the sandbox module is importable and that its public functions
behave correctly when filesystem and subprocess operations are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vetinari.project.sandbox import (
    ProjectSandboxResult,
    _load_policy,
    _run_lint,
    _run_tests,
    apply_changes,
    cleanup_sandbox,
    create_sandbox,
    execute_in_sandbox,
    run_checks,
)

# ---------------------------------------------------------------------------
# _load_policy
# ---------------------------------------------------------------------------


def test_load_policy_returns_dict():
    """Policy loader should always return a dict even on missing file."""
    result = _load_policy()
    assert isinstance(result, dict)


def test_load_policy_uses_timeout_key(tmp_path: Path):
    """Policy loader should surface the timeout_seconds key from the YAML."""
    policy_file = tmp_path / "sandbox_policy.yaml"
    policy_file.write_text(
        "sandbox:\n  external:\n    timeout_seconds: 99\n",
        encoding="utf-8",
    )
    with patch("vetinari.project.sandbox._POLICY_PATH", policy_file):
        policy = _load_policy()
    assert policy.get("timeout_seconds") == 99


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------


def test_apply_changes_writes_files(tmp_path: Path):
    changes = {
        "src/main.py": "print('hello')\n",
        "README.md": "# readme\n",
    }
    result = apply_changes(tmp_path, changes)
    assert result is True
    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# readme\n"


def test_apply_changes_creates_parent_dirs(tmp_path: Path):
    changes = {"deep/nested/dir/file.py": "x = 1\n"}
    apply_changes(tmp_path, changes)
    assert (tmp_path / "deep" / "nested" / "dir" / "file.py").exists()


def test_apply_changes_empty_dict(tmp_path: Path):
    assert apply_changes(tmp_path, {}) is True


# ---------------------------------------------------------------------------
# _run_tests (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_tests_passes_on_zero_exit(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "3 passed"
    mock.stderr = ""
    with patch("subprocess.run", return_value=mock):
        passed, errors = _run_tests(tmp_path, timeout=30)
    assert passed is True
    assert errors == []


def test_run_tests_uses_allowlisted_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PATH", "safe-path")
    captured = {}
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "3 passed"
    mock.stderr = ""

    def capture_run(*_args, **kwargs):
        captured["env"] = kwargs["env"]
        return mock

    with patch("subprocess.run", side_effect=capture_run):
        passed, errors = _run_tests(tmp_path, timeout=30)

    assert passed is True
    assert errors == []
    assert captured["env"].get("PATH") == "safe-path"
    assert "OPENAI_API_KEY" not in captured["env"]


def test_run_tests_fails_on_nonzero_exit(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 1
    mock.stdout = "FAILED test_something.py::test_x"
    mock.stderr = ""
    with patch("subprocess.run", return_value=mock):
        passed, errors = _run_tests(tmp_path, timeout=30)
    assert passed is False
    assert len(errors) == 1


def test_run_tests_fails_when_pytest_missing(tmp_path: Path):
    """When pytest is not installed the check fails closed — a missing test runner is not evidence of passing."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        passed, errors = _run_tests(tmp_path, timeout=30)
    assert passed is False
    assert len(errors) == 1


def test_run_tests_fails_on_timeout(tmp_path: Path):
    import subprocess

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=30)):
        passed, errors = _run_tests(tmp_path, timeout=30)
    assert passed is False
    assert "timed out" in errors[0]


# ---------------------------------------------------------------------------
# _run_lint (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_lint_passes_on_zero_exit(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = ""
    mock.stderr = ""
    with patch("subprocess.run", return_value=mock):
        passed, errors = _run_lint(tmp_path, timeout=30)
    assert passed is True
    assert errors == []


def test_run_lint_fails_on_violations(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 1
    mock.stdout = "E501 line too long"
    mock.stderr = ""
    with patch("subprocess.run", return_value=mock):
        passed, errors = _run_lint(tmp_path, timeout=30)
    assert passed is False
    assert len(errors) == 1


def test_run_lint_fails_when_ruff_missing(tmp_path: Path):
    """When ruff is not installed the check fails closed — a missing linter is not evidence of passing."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        passed, errors = _run_lint(tmp_path, timeout=30)
    assert passed is False
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# create_sandbox (mocked subprocess)
# ---------------------------------------------------------------------------


def test_create_sandbox_returns_path(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stderr = ""
    with patch("subprocess.run", return_value=mock):
        sandbox = create_sandbox(tmp_path)
    assert isinstance(sandbox, Path)


def test_create_sandbox_raises_on_git_failure(tmp_path: Path):
    mock = MagicMock()
    mock.returncode = 128
    mock.stderr = "not a git repository"
    with patch("subprocess.run", return_value=mock):
        with pytest.raises(RuntimeError, match="not a git repository"):
            create_sandbox(tmp_path)


# ---------------------------------------------------------------------------
# run_checks (mocked internals)
# ---------------------------------------------------------------------------


def test_run_checks_success_when_both_pass(tmp_path: Path):
    with (
        patch("vetinari.project.sandbox._run_tests", return_value=(True, [])),
        patch("vetinari.project.sandbox._run_lint", return_value=(True, [])),
    ):
        result = run_checks(tmp_path)
    assert result.success is True
    assert result.test_passed is True
    assert result.lint_passed is True
    assert result.errors == []


def test_run_checks_fails_when_tests_fail(tmp_path: Path):
    with (
        patch("vetinari.project.sandbox._run_tests", return_value=(False, ["test error"])),
        patch("vetinari.project.sandbox._run_lint", return_value=(True, [])),
    ):
        result = run_checks(tmp_path)
    assert result.success is False
    assert result.test_passed is False
    assert result.lint_passed is True
    assert "test error" in result.errors


def test_run_checks_fails_when_lint_fails(tmp_path: Path):
    with (
        patch("vetinari.project.sandbox._run_tests", return_value=(True, [])),
        patch("vetinari.project.sandbox._run_lint", return_value=(False, ["E501"])),
    ):
        result = run_checks(tmp_path)
    assert result.success is False
    assert result.lint_passed is False


# ---------------------------------------------------------------------------
# execute_in_sandbox (all-in-one)
# ---------------------------------------------------------------------------


def test_execute_in_sandbox_returns_passing_result(tmp_path: Path):
    changes = {"hello.py": "x = 1\n"}

    with (
        patch("vetinari.project.sandbox.create_sandbox", return_value=tmp_path / "sandbox"),
        patch("vetinari.project.sandbox.apply_changes", return_value=True),
        patch(
            "vetinari.project.sandbox.run_checks",
            return_value=ProjectSandboxResult(
                success=True,
                test_passed=True,
                lint_passed=True,
                worktree_path=tmp_path / "sandbox",
            ),
        ),
        patch("vetinari.project.sandbox.cleanup_sandbox"),
    ):
        result = execute_in_sandbox(tmp_path, changes)

    assert result.success is True
    assert result.worktree_path is None  # cleaned up


def test_execute_in_sandbox_handles_create_failure(tmp_path: Path):
    with patch(
        "vetinari.project.sandbox.create_sandbox",
        side_effect=RuntimeError("not a git repository"),
    ):
        result = execute_in_sandbox(tmp_path, {})

    assert result.success is False
    assert "not a git repository" in result.errors[0]


def test_execute_in_sandbox_always_cleans_up(tmp_path: Path):
    cleanup_mock = MagicMock()
    with (
        patch("vetinari.project.sandbox.create_sandbox", return_value=tmp_path / "sandbox"),
        patch("vetinari.project.sandbox.apply_changes", return_value=True),
        patch(
            "vetinari.project.sandbox.run_checks",
            side_effect=Exception("unexpected error"),
        ),
        patch("vetinari.project.sandbox.cleanup_sandbox", cleanup_mock),
    ):
        with pytest.raises(Exception, match="unexpected error"):
            execute_in_sandbox(tmp_path, {})

    cleanup_mock.assert_called_once()
    assert cleanup_mock.call_args.args[0] == tmp_path / "sandbox"


# ---------------------------------------------------------------------------
# SandboxResult dataclass
# ---------------------------------------------------------------------------


def test_sandbox_result_repr():
    r = ProjectSandboxResult(success=True, test_passed=True, lint_passed=False)
    assert "success=True" in repr(r)
    assert "lint_passed=False" in repr(r)
