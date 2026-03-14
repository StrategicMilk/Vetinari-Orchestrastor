"""Git Operations Tool.

===================
Provides safe git operations for Vetinari agents.

All commands run via ``subprocess`` with ``shell=False``.  The working
directory is locked to the project root.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class GitResult:
    """Result of a git operation."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
        }


class GitOperations:
    """Low-level git operations scoped to a repository root."""

    TIMEOUT = 30  # seconds

    def __init__(self, repo_path: str | Path | None = None):
        self.repo = Path(repo_path or ".").resolve()
        self._git = shutil.which("git") or "git"

    # -- helpers ------------------------------------------------------------

    def _run(self, args: list[str], timeout: int | None = None) -> GitResult:
        """Execute ``git <args>`` and return the result."""
        cmd = [self._git, *args]
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.TIMEOUT,
                cwd=str(self.repo),
            )
            return GitResult(
                success=proc.returncode == 0,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                return_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return GitResult(success=False, stderr="git command timed out", return_code=-1)
        except FileNotFoundError:
            return GitResult(success=False, stderr="git executable not found", return_code=-1)
        except Exception as exc:
            return GitResult(success=False, stderr=str(exc), return_code=-1)

    # -- public API ---------------------------------------------------------

    def status(self) -> GitResult:
        return self._run(["status", "--porcelain"])

    def log(self, n: int = 10) -> GitResult:
        return self._run(["log", f"-{n}", "--oneline"])

    def diff(self, base: str = "HEAD", head: str = "") -> GitResult:
        """Diff.

        Args:
            base: The base.
            head: The head.

        Returns:
            The GitResult result.
        """
        args = ["diff", base]
        if head:
            args.append(head)
        return self._run(args)

    def init_repo(self) -> GitResult:
        return self._run(["init"])

    def add(self, files: list[str] | None = None) -> GitResult:
        """Add.

        Returns:
            The GitResult result.
        """
        if files:
            return self._run(["add", *files])
        return self._run(["add", "."])

    def commit(self, message: str) -> GitResult:
        return self._run(["commit", "-m", message])

    def create_branch(self, name: str) -> GitResult:
        return self._run(["checkout", "-b", name])

    def checkout(self, branch: str) -> GitResult:
        return self._run(["checkout", branch])

    def current_branch(self) -> GitResult:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"])

    def push(self, remote: str = "origin", branch: str = "") -> GitResult:
        """Push.

        Args:
            remote: The remote.
            branch: The branch.

        Returns:
            The GitResult result.
        """
        args = ["push", remote]
        if branch:
            args.append(branch)
        return self._run(args, timeout=60)

    def pull(self, remote: str = "origin", branch: str = "") -> GitResult:
        """Pull.

        Args:
            remote: The remote.
            branch: The branch.

        Returns:
            The GitResult result.
        """
        args = ["pull", remote]
        if branch:
            args.append(branch)
        return self._run(args, timeout=60)

    def stash(self, pop: bool = False) -> GitResult:
        return self._run(["stash", "pop"] if pop else ["stash"])

    def tag(self, name: str, message: str = "") -> GitResult:
        """Tag.

        Args:
            name: The name.
            message: The message.

        Returns:
            The GitResult result.
        """
        if message:
            return self._run(["tag", "-a", name, "-m", message])
        return self._run(["tag", name])


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class GitOperationsTool(Tool):
    """Vetinari Tool wrapper around :class:`GitOperations`."""

    def __init__(self, repo_path: str | None = None):
        self._git = GitOperations(repo_path)
        metadata = ToolMetadata(
            name="git_operations",
            description="Version control operations via git CLI",
            version="1.0.0",
            category=ToolCategory.GIT_OPERATIONS,
            required_permissions=[ToolPermission.FILE_WRITE],
            parameters=[
                ToolParameter(
                    name="operation",
                    type=str,
                    description="Operation: status, log, diff, init, add, commit, branch, checkout, push, pull, stash, tag, current_branch",
                    required=True,
                ),
                ToolParameter(name="message", type=str, description="Commit/tag message", required=False),
                ToolParameter(name="branch", type=str, description="Branch name", required=False),
                ToolParameter(name="files", type=list, description="Files to add", required=False),
                ToolParameter(name="n", type=int, description="Number of log entries", required=False, default=10),
                ToolParameter(name="remote", type=str, description="Remote name", required=False, default="origin"),
            ],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        op = kwargs.get("operation", "")

        try:
            if op == "status":
                r = self._git.status()
            elif op == "log":
                r = self._git.log(n=kwargs.get("n", 10))
            elif op == "diff":
                r = self._git.diff(
                    base=kwargs.get("base", "HEAD"),
                    head=kwargs.get("head", ""),
                )
            elif op == "init":
                r = self._git.init_repo()
            elif op == "add":
                r = self._git.add(files=kwargs.get("files"))
            elif op == "commit":
                msg = kwargs.get("message", "")
                if not msg:
                    return ToolResult(success=False, output="", error="message is required for commit")
                r = self._git.commit(msg)
            elif op == "branch":
                name = kwargs.get("branch", "")
                if not name:
                    return ToolResult(success=False, output="", error="branch name is required")
                r = self._git.create_branch(name)
            elif op == "checkout":
                name = kwargs.get("branch", "")
                if not name:
                    return ToolResult(success=False, output="", error="branch name is required")
                r = self._git.checkout(name)
            elif op == "current_branch":
                r = self._git.current_branch()
            elif op == "push":
                r = self._git.push(
                    remote=kwargs.get("remote", "origin"),
                    branch=kwargs.get("branch", ""),
                )
            elif op == "pull":
                r = self._git.pull(
                    remote=kwargs.get("remote", "origin"),
                    branch=kwargs.get("branch", ""),
                )
            elif op == "stash":
                r = self._git.stash(pop=kwargs.get("pop", False))
            elif op == "tag":
                name = kwargs.get("name", "")
                if not name:
                    return ToolResult(success=False, output="", error="tag name is required")
                r = self._git.tag(name, message=kwargs.get("message", ""))
            else:
                return ToolResult(success=False, output="", error=f"Unknown operation: {op}")

            return ToolResult(
                success=r.success,
                output=r.stdout or r.stderr,
                error=r.stderr if not r.success else "",
            )

        except Exception as exc:
            logger.exception("GitOperationsTool error")
            return ToolResult(success=False, output="", error=str(exc))
