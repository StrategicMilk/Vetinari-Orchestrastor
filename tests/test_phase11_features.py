"""
Tests for Phase 11 remaining features: File Tool, Git Tool, Sandbox Enforcement.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from vetinari.execution_context import get_context_manager

# ---------------------------------------------------------------------------
# File Tool tests
# ---------------------------------------------------------------------------
from vetinari.tools.file_tool import (
    FileInfo,
    FileOperations,
    FileOperationsTool,
    _safe_resolve,
)
from vetinari.types import ExecutionMode


class TestSafeResolve:
    """Path traversal protection."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def test_normal_path(self):
        result = _safe_resolve("foo/bar.txt", self.root)
        assert str(result).startswith(str(self.root.resolve()))

    def test_traversal_blocked(self):
        with pytest.raises(PermissionError):
            _safe_resolve("../../etc/passwd", self.root)

    def test_absolute_path_outside_blocked(self):
        with pytest.raises(PermissionError):
            _safe_resolve("/tmp/evil.txt", self.root)

    def test_dot_stays_in_root(self):
        result = _safe_resolve(".", self.root)
        assert result == self.root.resolve()


class TestFileOperations:
    """Core file operations."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = Path(tempfile.mkdtemp())
        self.ops = FileOperations(self.root)
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_and_read(self):
        self.ops.write_file("hello.txt", "Hello, World!")
        content = self.ops.read_file("hello.txt")
        assert content == "Hello, World!"

    def test_write_creates_parents(self):
        self.ops.write_file("a/b/c/deep.txt", "deep")
        assert (self.root / "a" / "b" / "c" / "deep.txt").exists()

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self.ops.read_file("nonexistent.txt")

    def test_file_exists(self):
        assert not self.ops.file_exists("nope.txt")
        self.ops.write_file("yep.txt", "yes")
        assert self.ops.file_exists("yep.txt")

    def test_get_file_info(self):
        self.ops.write_file("info.txt", "data")
        info = self.ops.get_file_info("info.txt")
        assert isinstance(info, FileInfo)
        assert info.is_file
        assert not info.is_dir
        assert info.size_bytes > 0
        assert info.name == "info.txt"

    def test_get_file_info_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            self.ops.get_file_info("missing.txt")

    def test_list_directory(self):
        self.ops.write_file("a.txt", "a")
        self.ops.write_file("b.txt", "b")
        self.ops.create_directory("sub")
        entries = self.ops.list_directory(".")
        names = [e.name for e in entries]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "sub" in names

    def test_list_directory_not_a_dir(self):
        self.ops.write_file("file.txt", "x")
        with pytest.raises(NotADirectoryError):
            self.ops.list_directory("file.txt")

    def test_create_directory(self):
        self.ops.create_directory("new/nested/dir")
        assert (self.root / "new" / "nested" / "dir").is_dir()

    def test_move_file(self):
        self.ops.write_file("src.txt", "content")
        self.ops.move_file("src.txt", "dst.txt")
        assert not (self.root / "src.txt").exists()
        assert (self.root / "dst.txt").exists()
        assert self.ops.read_file("dst.txt") == "content"

    def test_move_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self.ops.move_file("missing.txt", "dst.txt")

    def test_delete_file(self):
        self.ops.write_file("del.txt", "gone")
        assert self.ops.delete_file("del.txt")
        assert not (self.root / "del.txt").exists()

    def test_delete_nonexistent_returns_false(self):
        assert not self.ops.delete_file("nope.txt")

    def test_delete_directory(self):
        self.ops.create_directory("subdir")
        self.ops.write_file("subdir/file.txt", "x")
        assert self.ops.delete_file("subdir")
        assert not (self.root / "subdir").exists()

    def test_traversal_on_write(self):
        with pytest.raises(PermissionError):
            self.ops.write_file("../../evil.txt", "bad")

    def test_traversal_on_read(self):
        with pytest.raises(PermissionError):
            self.ops.read_file("../../etc/passwd")


class TestFileOperationsTool:
    """Tool wrapper integration."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = Path(tempfile.mkdtemp())
        self.tool = FileOperationsTool(str(self.root))
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_and_read_via_tool(self):
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            result = self.tool.execute(operation="write", path="test.txt", content="hello")
            assert result.success
            result = self.tool.execute(operation="read", path="test.txt")
            assert result.success
            assert result.output == "hello"

    def test_exists_via_tool(self):
        result = self.tool.execute(operation="exists", path="nope.txt")
        assert result.success
        assert not result.output

    def test_unknown_operation(self):
        result = self.tool.execute(operation="explode", path=".")
        assert not result.success

    def test_traversal_via_tool(self):
        result = self.tool.execute(operation="read", path="../../etc/passwd")
        assert not result.success
        assert "traversal" in result.error.lower()


# ---------------------------------------------------------------------------
# Git Tool tests
# ---------------------------------------------------------------------------

from vetinari.tools.git_tool import GitOperations, GitOperationsTool, GitResult


class TestGitOperations:
    """Git operations in a temp repo."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init"], cwd=str(self.repo), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(self.repo),
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(self.repo),
            capture_output=True,
        )
        self.git = GitOperations(self.repo)
        yield
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_status_empty_repo(self):
        r = self.git.status()
        assert r.success

    def test_add_and_commit(self):
        (self.repo / "file.txt").write_text("hello")
        r = self.git.add(["file.txt"])
        assert r.success
        r = self.git.commit("initial commit")
        assert r.success

    def test_log(self):
        (self.repo / "file.txt").write_text("hello")
        self.git.add(["file.txt"])
        self.git.commit("initial commit")
        r = self.git.log(n=5)
        assert r.success
        assert "initial commit" in r.stdout

    def test_create_branch_and_checkout(self):
        (self.repo / "file.txt").write_text("hello")
        self.git.add(["file.txt"])
        self.git.commit("initial")
        r = self.git.create_branch("feature")
        assert r.success
        r = self.git.current_branch()
        assert r.success
        assert r.stdout == "feature"
        # Switch back; accept either master or main
        r = self.git.checkout("master")
        if not r.success:
            r = self.git.checkout("main")

    def test_diff(self):
        (self.repo / "file.txt").write_text("hello")
        self.git.add(["file.txt"])
        self.git.commit("initial")
        (self.repo / "file.txt").write_text("changed")
        r = self.git.diff()
        assert r.success
        assert "changed" in r.stdout

    def test_stash(self):
        (self.repo / "file.txt").write_text("hello")
        self.git.add(["file.txt"])
        self.git.commit("initial")
        (self.repo / "file.txt").write_text("changed")
        self.git.add(["file.txt"])
        r = self.git.stash()
        assert r.success
        r = self.git.stash(pop=True)
        assert r.success

    def test_git_result_to_dict(self):
        r = GitResult(success=True, stdout="ok", stderr="", return_code=0)
        d = r.to_dict()
        assert d["success"]
        assert d["stdout"] == "ok"


class TestGitOperationsTool:
    """Git tool wrapper."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init"], cwd=str(self.repo), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(self.repo),
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(self.repo),
            capture_output=True,
        )
        self.tool = GitOperationsTool(str(self.repo))
        ctx = get_context_manager()
        ctx.switch_mode(ExecutionMode.EXECUTION, task_id="git-test")
        yield
        ctx.pop_context()
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_status_via_tool(self):
        result = self.tool.execute(operation="status")
        assert result.success

    def test_commit_requires_message(self):
        result = self.tool.execute(operation="commit")
        assert not result.success
        assert "message" in result.error

    def test_unknown_operation(self):
        result = self.tool.execute(operation="rebase")
        assert not result.success

    def test_full_workflow(self):
        (self.repo / "f.txt").write_text("data")
        r = self.tool.execute(operation="add", files=["f.txt"])
        assert r.success
        r = self.tool.execute(operation="commit", message="add file")
        assert r.success
        r = self.tool.execute(operation="log", n=1)
        assert r.success
        assert "add file" in r.output


# ---------------------------------------------------------------------------
# Sandbox Enforcement tests
# ---------------------------------------------------------------------------

from vetinari.code_sandbox import CodeSandbox, ExecutionResult


def _sandbox_output(result: ExecutionResult) -> str:
    """Extract user output + errors from sandbox JSON wrapper in stdout."""
    raw = result.stdout or result.output or ""
    start_marker = "===VETINARI_OUTPUT_START==="
    end_marker = "===VETINARI_OUTPUT_END==="
    if start_marker in raw:
        start = raw.index(start_marker) + len(start_marker)
        end = raw.index(end_marker) if end_marker in raw else len(raw)
        try:
            data = json.loads(raw[start:end].strip())
            return data.get("output", "") + data.get("errors", "")
        except json.JSONDecodeError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass
    return raw


class TestSandboxModuleRestrictions:
    """Verify module blocking is enforced."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sandbox = CodeSandbox(max_execution_time=10)
        yield
        self.sandbox.cleanup()

    def test_blocked_module_raises_import_error(self):
        result = self.sandbox.execute_python("import subprocess")
        output = _sandbox_output(result)
        assert "blocked" in output.lower()

    def test_allowed_module_works(self):
        result = self.sandbox.execute_python("import math\nprint(math.pi)")
        output = _sandbox_output(result)
        assert "3.14" in output

    def test_network_blocked_by_default(self):
        result = self.sandbox.execute_python("import socket")
        output = _sandbox_output(result)
        assert "blocked" in output.lower()

    def test_network_allowed_when_configured(self):
        sandbox = CodeSandbox(
            max_execution_time=10,
            allow_network=True,
            blocked_modules=[],
        )
        try:
            result = sandbox.execute_python("import socket\nprint('ok')")
            output = _sandbox_output(result)
            assert "ok" in output
        finally:
            sandbox.cleanup()

    def test_custom_blocked_modules(self):
        sandbox = CodeSandbox(
            max_execution_time=10,
            blocked_modules=["csv"],
        )
        try:
            result = sandbox.execute_python("import csv")
            output = _sandbox_output(result)
            assert "blocked" in output.lower()
        finally:
            sandbox.cleanup()

    def test_timeout_still_works(self):
        result = self.sandbox.execute_python(
            "import time\ntime.sleep(30)",
            timeout=2,
        )
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_basic_execution_still_works(self):
        result = self.sandbox.execute_python("x = 2 + 3\nprint(x)")
        output = _sandbox_output(result)
        assert "5" in output


class TestSandboxExecutionResult:
    """ExecutionResult dataclass."""

    def test_to_dict(self):
        r = ExecutionResult(success=True, output="hello", error="")
        d = r.to_dict()
        assert d["success"]
        assert d["output"] == "hello"

    def test_default_fields(self):
        r = ExecutionResult(success=False, output="")
        assert r.return_code == 0
        assert r.files_created == []


# ---------------------------------------------------------------------------
# FileInfo dataclass
# ---------------------------------------------------------------------------


class TestFileInfo:
    def test_to_dict(self):
        info = FileInfo(path="a.txt", name="a.txt", is_file=True, is_dir=False, size_bytes=42)
        d = info.to_dict()
        assert d["size_bytes"] == 42
        assert d["is_file"]
