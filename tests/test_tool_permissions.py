"""Tests for granular tool permission enforcement (SESSION-28 security item 28.1).

Verifies that FileOperationsTool and GitOperationsTool enforce the correct
permission per operation class:
- Read-only ops require READ permission and work in PLANNING mode.
- Write/mutate ops require WRITE permission and are blocked in PLANNING mode.
- Confirmation-required ops (GIT_PUSH, FILE_DELETE) are blocked in all
  automated/headless contexts via fail-closed enforcement.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from vetinari.exceptions import SecurityError
from vetinari.execution_context import ToolPermission, get_context_manager
from vetinari.tools.file_tool import FileOperationsTool
from vetinari.tools.git_tool import GitOperationsTool
from vetinari.types import ExecutionMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_tool() -> tuple[FileOperationsTool, Path]:
    """Create a FileOperationsTool scoped to a fresh temp directory.

    Returns:
        Tuple of (tool, root_path).
    """
    root = Path(tempfile.mkdtemp(dir=Path(__file__).parent.parent / "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    return FileOperationsTool(str(root)), root


def _make_git_tool() -> tuple[GitOperationsTool, Path]:
    """Create a GitOperationsTool in a fresh temp git repository.

    Returns:
        Tuple of (tool, repo_path).
    """
    repo = Path(tempfile.mkdtemp(dir=Path(__file__).parent.parent / "outputs"))
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@vetinari.test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Vetinari Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    return GitOperationsTool(str(repo)), repo


# ---------------------------------------------------------------------------
# FileOperationsTool — read ops in read-only (PLANNING) mode
# ---------------------------------------------------------------------------


class TestFileReadOpsInPlanningMode:
    """Read-only file ops must work in PLANNING mode (FILE_READ is allowed there)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tool, self.root = _make_file_tool()
        # Write a file to read back — must be done in EXECUTION mode first.
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            self.tool.execute(operation="write", path="hello.txt", content="world")
        # Tests run with whatever mode is currently on the stack (default: PLANNING).
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def test_read_works_in_planning_mode(self):
        """file read must succeed in PLANNING mode (FILE_READ is granted)."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="read", path="hello.txt")
        assert result.success is True
        assert result.output == "world"

    def test_list_works_in_planning_mode(self):
        """file list must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="list", path=".")
        assert result.success is True
        assert isinstance(result.output, list)

    def test_info_works_in_planning_mode(self):
        """file info must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="info", path="hello.txt")
        assert result.success is True
        assert result.output["name"] == "hello.txt"

    def test_exists_works_in_planning_mode(self):
        """file exists must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="exists", path="hello.txt")
        assert result.success is True
        assert result.output is True


# ---------------------------------------------------------------------------
# FileOperationsTool — write ops blocked in read-only (PLANNING) mode
# ---------------------------------------------------------------------------


class TestFileWriteOpsBlockedInPlanningMode:
    """Write/mutate file ops must be blocked in PLANNING mode (FILE_WRITE not granted)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tool, self.root = _make_file_tool()
        # Seed a file so move/delete have something to act on.
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            self.tool.execute(operation="write", path="seed.txt", content="data")
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_blocked_in_planning_mode(self):
        """file write must be blocked in PLANNING mode — FILE_WRITE not granted."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="file_write"):
                self.tool.execute(operation="write", path="new.txt", content="bad")

    def test_mkdir_blocked_in_planning_mode(self):
        """file mkdir must be blocked in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="file_mkdir"):
                self.tool.execute(operation="mkdir", path="newdir")

    def test_move_blocked_in_planning_mode(self):
        """file move must be blocked in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="file_move"):
                self.tool.execute(operation="move", path="seed.txt", destination="moved.txt")

    def test_delete_blocked_in_planning_mode(self):
        """file delete must be blocked in PLANNING mode — FILE_DELETE not granted."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="file_delete"):
                self.tool.execute(operation="delete", path="seed.txt")

    def test_delete_blocked_by_confirmation_in_execution_mode(self):
        """file delete requires confirmation in EXECUTION mode — must block fail-closed."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            with pytest.raises(SecurityError, match="confirmation"):
                self.tool.execute(operation="delete", path="seed.txt")


# ---------------------------------------------------------------------------
# GitOperationsTool — read ops in read-only (PLANNING) mode
# ---------------------------------------------------------------------------


class TestGitReadOpsInPlanningMode:
    """Read-only git ops must work in PLANNING mode (GIT_READ is granted there)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tool, self.repo = _make_git_tool()
        # Seed one commit so log/diff have something to show.
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            (self.repo / "seed.txt").write_text("hello", encoding="utf-8")
            self.tool.execute(operation="add", files=["seed.txt"])
            self.tool.execute(operation="commit", message="initial")
        yield
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_status_works_in_planning_mode(self):
        """git status must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="status")
        assert result.success is True

    def test_log_works_in_planning_mode(self):
        """git log must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="log", n=5)
        assert result.success is True

    def test_diff_works_in_planning_mode(self):
        """git diff must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="diff")
        assert result.success is True

    def test_current_branch_works_in_planning_mode(self):
        """git current_branch must succeed in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            result = self.tool.execute(operation="current_branch")
        assert result.success is True


# ---------------------------------------------------------------------------
# GitOperationsTool — write ops blocked in read-only (PLANNING) mode
# ---------------------------------------------------------------------------


class TestGitWriteOpsBlockedInPlanningMode:
    """Write git ops must be blocked in PLANNING mode (GIT_COMMIT not granted)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tool, self.repo = _make_git_tool()
        yield
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_commit_blocked_in_planning_mode(self):
        """git commit must be blocked in PLANNING mode — GIT_COMMIT not granted."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="git_commit"):
                self.tool.execute(operation="commit", message="should fail")

    def test_add_blocked_in_planning_mode(self):
        """git add must be blocked in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="git_add"):
                self.tool.execute(operation="add", files=["anything.txt"])

    def test_branch_blocked_in_planning_mode(self):
        """git branch creation must be blocked in PLANNING mode."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="git_branch"):
                self.tool.execute(operation="branch", branch="new-branch")

    def test_push_blocked_in_planning_mode(self):
        """git push must be blocked in PLANNING mode — GIT_PUSH not granted."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.PLANNING):
            with pytest.raises(SecurityError, match="git_push"):
                self.tool.execute(operation="push")

    def test_push_blocked_by_confirmation_in_execution_mode(self):
        """git push requires the git_push permission which is not granted in EXECUTION mode — blocks fail-closed."""
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            with pytest.raises(SecurityError, match="git_push"):
                self.tool.execute(operation="push")
