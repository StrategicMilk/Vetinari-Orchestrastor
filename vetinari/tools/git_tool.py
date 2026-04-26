"""Git Operations Tool.

Provides safe git operations for Vetinari agents, including low-level git
commands and higher-level helpers for conventional commits, branch management,
PR description generation, and conflict detection.

All commands run via ``subprocess`` with ``shell=False``.  The working
directory is locked to the project root.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vetinari.constants import GIT_OPERATION_TIMEOUT
from vetinari.execution_context import ToolPermission, get_context_manager
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)

# Conventional commit type registry
COMMIT_TYPES: dict[str, str] = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "refactor": "Code change that neither fixes a bug nor adds a feature",
    "docs": "Documentation only changes",
    "test": "Adding missing tests or correcting existing tests",
    "chore": "Changes to build process or auxiliary tools",
    "style": "Formatting, missing semicolons, etc; no code change",
    "perf": "Performance improvement",
    "ci": "Changes to CI configuration files and scripts",
}


@dataclass
class CommitInfo:
    """Structured commit information following Conventional Commits."""

    type: str  # feat, fix, refactor, etc.
    scope: str | None = None
    description: str = ""
    body: str = ""
    breaking: bool = False
    files_changed: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"CommitInfo(type={self.type!r}, scope={self.scope!r}, breaking={self.breaking!r})"

    def format_message(self) -> str:
        """Format as a conventional commit message string.

        Returns:
            The formatted conventional commit message.
        """
        prefix = self.type
        if self.scope:
            prefix = f"{self.type}({self.scope})"
        if self.breaking:
            prefix += "!"
        msg = f"{prefix}: {self.description}"
        if self.body:
            msg += f"\n\n{self.body}"
        return msg


@dataclass
class BranchInfo:
    """Information about a git branch."""

    name: str
    is_current: bool = False
    ahead: int = 0
    behind: int = 0
    last_commit: str = ""
    created_from: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"BranchInfo(name={self.name!r}, is_current={self.is_current!r}, ahead={self.ahead!r})"


@dataclass
class ConflictInfo:
    """Information about a merge conflict."""

    file_path: str
    conflict_type: str  # "content", "rename", "delete"
    ours_content: str = ""
    theirs_content: str = ""
    suggestion: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"ConflictInfo(file_path={self.file_path!r}, conflict_type={self.conflict_type!r})"


@dataclass
class GitResult:
    """Result of a git operation."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    def __repr__(self) -> str:
        return f"GitResult(success={self.success!r}, return_code={self.return_code!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


class GitOperations:
    """Low-level git operations scoped to a repository root."""

    TIMEOUT = 30  # seconds

    def __init__(self, repo_path: str | Path):
        if not repo_path:
            raise ValueError("repo_path is required — process cwd is not a safe default for sandboxed git operations")
        self.repo = Path(repo_path).resolve()
        self._git = shutil.which("git") or "git"

    # -- helpers ------------------------------------------------------------

    def _run(self, args: list[str], timeout: int | None = None) -> GitResult:
        """Execute ``git <args>`` and return the result."""
        cmd = [self._git, *args]
        try:
            proc = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
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
            logger.warning(
                "git %s in %s took too long (>%ss) — aborting",
                " ".join(args),
                self.repo,
                timeout or self.TIMEOUT,
            )
            return GitResult(success=False, stderr="git command timed out", return_code=-1)
        except FileNotFoundError:
            logger.warning("git executable not found — ensure git is installed and on PATH (repo: %s)", self.repo)
            return GitResult(success=False, stderr="git executable not found", return_code=-1)
        except Exception as exc:
            logger.warning(
                "git %s failed with unexpected error in repo %s: %s — returning failed GitResult",
                " ".join(args),
                self.repo,
                exc,
            )
            return GitResult(success=False, stderr=str(exc), return_code=-1)

    # -- public API ---------------------------------------------------------

    def status(self) -> GitResult:
        """Run ``git status --porcelain`` and return the result.

        Returns:
            GitResult with porcelain-formatted status output.
        """
        return self._run(["status", "--porcelain"])

    def log(self, n: int = 10) -> GitResult:
        """Run ``git log --oneline`` for the most recent commits.

        Args:
            n: Number of log entries to retrieve.

        Returns:
            GitResult with one-line commit log output.
        """
        return self._run(["log", f"-{n}", "--oneline"])

    def diff(self, base: str = "HEAD", head: str = "") -> GitResult:
        """Show differences between commits, the index, or the working tree.

        Args:
            base: Base reference for the diff (default ``"HEAD"``).
            head: Optional head reference; when empty, diffs against the working tree.

        Returns:
            GitResult with the diff output in stdout.
        """
        args = ["diff", base]
        if head:
            args.append(head)
        return self._run(args)

    def init_repo(self) -> GitResult:
        """Initialize a new git repository in the configured repo path.

        Returns:
            GitResult indicating whether ``git init`` succeeded.
        """
        return self._run(["init"])

    def add(self, files: list[str] | None = None) -> GitResult:
        """Stage files for the next commit.

        Args:
            files: Specific file paths to stage. When ``None``, stages all changes.

        Returns:
            GitResult indicating whether the add operation succeeded.
        """
        if files:
            return self._run(["add", *files])
        return self._run(["add", "."])

    def commit(self, message: str) -> GitResult:
        """Create a commit with the given message from currently staged changes.

        Args:
            message: The commit message text.

        Returns:
            GitResult indicating whether the commit succeeded.
        """
        return self._run(["commit", "-m", message])

    def create_branch(self, name: str) -> GitResult:
        """Create and switch to a new branch via ``git checkout -b``.

        Args:
            name: The new branch name.

        Returns:
            GitResult indicating whether branch creation succeeded.
        """
        return self._run(["checkout", "-b", name])

    def checkout(self, branch: str) -> GitResult:
        """Switch to an existing branch.

        Args:
            branch: The branch name to switch to.

        Returns:
            GitResult indicating whether the checkout succeeded.
        """
        return self._run(["checkout", branch])

    def current_branch(self) -> GitResult:
        """Return the name of the currently checked-out branch.

        Returns:
            GitResult with the branch name in stdout.
        """
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"])

    def push(self, remote: str = "origin", branch: str = "") -> GitResult:
        """Push local commits to a remote repository.

        Args:
            remote: Remote name to push to (default ``"origin"``).
            branch: Branch name to push; when empty, pushes the current branch.

        Returns:
            GitResult indicating whether the push succeeded.
        """
        args = ["push", remote]
        if branch:
            args.append(branch)
        return self._run(args, timeout=GIT_OPERATION_TIMEOUT)

    def pull(self, remote: str = "origin", branch: str = "") -> GitResult:
        """Fetch and merge changes from a remote repository.

        Args:
            remote: Remote name to pull from (default ``"origin"``).
            branch: Branch name to pull; when empty, pulls the current branch.

        Returns:
            GitResult indicating whether the pull succeeded.
        """
        args = ["pull", remote]
        if branch:
            args.append(branch)
        return self._run(args, timeout=GIT_OPERATION_TIMEOUT)

    def stash(self, pop: bool = False) -> GitResult:
        """Stash or restore uncommitted changes.

        Args:
            pop: When True, pops the most recent stash; otherwise stashes changes.

        Returns:
            GitResult indicating whether the stash operation succeeded.
        """
        return self._run(["stash", "pop"] if pop else ["stash"])

    def tag(self, name: str, message: str = "") -> GitResult:
        """Create a git tag on the current commit.

        Args:
            name: Tag name to create.
            message: Optional annotation message; when provided creates an annotated tag.

        Returns:
            GitResult indicating whether the tag was created successfully.
        """
        if message:
            return self._run(["tag", "-a", name, "-m", message])
        return self._run(["tag", name])

    # -- high-level helpers -------------------------------------------------

    def classify_changes(self, diff_text: str | None = None) -> str:
        """Classify staged (or unstaged) changes into a conventional commit type.

        If *diff_text* is ``None`` the method inspects the currently staged
        changes, falling back to unstaged changes.

        Args:
            diff_text: Optional diff text to classify. When ``None`` the live
                staged diff is used.

        Returns:
            A conventional commit type string (e.g. ``"feat"``, ``"fix"``).
        """
        if diff_text is None:
            r = self._run(["diff", "--cached", "--stat"])
            diff_text = r.stdout

        if not diff_text:
            r = self._run(["diff", "--stat"])
            diff_text = r.stdout

        lower = diff_text.lower()

        if "test" in lower:
            return "test"
        if any(d in lower for d in ["readme", "docs/", ".md", "documentation"]):
            return "docs"
        if any(d in lower for d in ["ci/", ".github/", "workflow", "pipeline"]):
            return "ci"
        if any(d in lower for d in ["fix", "bug", "patch", "hotfix"]):
            return "fix"
        if any(d in lower for d in ["refactor", "rename", "move", "reorganize"]):
            return "refactor"
        if any(d in lower for d in ["perf", "optim", "speed", "cache"]):
            return "perf"
        return "feat"

    def generate_commit_message(
        self,
        description: str | None = None,
        scope: str | None = None,
    ) -> CommitInfo:
        """Generate a conventional commit message from staged changes.

        Args:
            description: Optional commit description. When ``None`` one is
                inferred from the changed file paths.
            scope: Optional conventional-commit scope. When ``None`` one is
                inferred from the changed file paths.

        Returns:
            A :class:`CommitInfo` instance ready to be formatted.
        """
        diff_stat_result = self._run(["diff", "--cached", "--stat"])
        diff_names_result = self._run(["diff", "--cached", "--name-only"])

        diff_stat = diff_stat_result.stdout
        diff_names = diff_names_result.stdout

        files = [f.strip() for f in diff_names.split("\n") if f.strip()] if diff_names else []
        commit_type = self.classify_changes(diff_stat)

        if not description:
            description = self._infer_description(files)

        if not scope and files:
            scope = self._infer_scope(files)

        return CommitInfo(
            type=commit_type,
            scope=scope,
            description=description,
            files_changed=files,
        )

    def generate_pr_description(self, base_branch: str = "main") -> dict[str, Any]:
        """Generate a PR description from commit history since *base_branch*.

        Args:
            base_branch: The branch to compare against. Defaults to ``"main"``.

        Returns:
            A dict with keys ``"title"`` (str), ``"body"`` (str), and
            ``"commits"`` (int).
        """
        log_result = self._run(["log", f"{base_branch}..HEAD", "--oneline"])
        diff_result = self._run(["diff", f"{base_branch}..HEAD", "--stat"])

        log_output = log_result.stdout
        diff_stat = diff_result.stdout

        commits = [line.strip() for line in log_output.split("\n") if line.strip()] if log_output else []

        title = commits[0].split(" ", 1)[1] if commits else "Update"

        body_lines = ["## Summary", ""]
        for c in commits:
            parts = c.split(" ", 1)
            body_lines.append(f"- {parts[1] if len(parts) > 1 else c}")

        body_lines.extend(["", "## Changes", "", diff_stat or "No changes detected"])

        return {
            "title": title[:70],
            "body": "\n".join(body_lines),
            "commits": len(commits),
        }

    def detect_conflicts(self, target_branch: str = "main") -> list[ConflictInfo]:
        """Detect potential merge conflicts with *target_branch*.

        Performs a dry-run merge and parses the git output for CONFLICT lines.

        Args:
            target_branch: The branch to test merging into the current branch.
                Defaults to ``"main"``.

        Returns:
            A list of :class:`ConflictInfo` instances, one per conflicting file.
            Returns an empty list when no conflicts are detected.
        """
        result = self._run(["merge", "--no-commit", "--no-ff", target_branch, "--dry-run"])

        conflicts: list[ConflictInfo] = []
        if result.return_code != 0 and "conflict" in result.stderr.lower():
            for line in result.stderr.split("\n"):
                if "CONFLICT" in line:
                    match = re.search(r"CONFLICT.*?:\s*(.+)", line)
                    file_path = match.group(1).strip() if match else "unknown"
                    conflict_type = "content"
                    if "rename" in line.lower():
                        conflict_type = "rename"
                    elif "delete" in line.lower():
                        conflict_type = "delete"
                    conflicts.append(ConflictInfo(file_path=file_path, conflict_type=conflict_type))
        return conflicts

    # -- private helpers ----------------------------------------------------

    def _infer_description(self, files: list[str]) -> str:
        """Infer a commit description from changed file paths.

        Args:
            files: List of changed file paths.

        Returns:
            A short human-readable description string.
        """
        if not files:
            return "update files"
        if len(files) == 1:
            return f"update {files[0].split('/')[-1]}"
        common_dir = self._common_directory(files)
        if common_dir:
            return f"update {common_dir} ({len(files)} files)"
        return f"update {len(files)} files"

    def _infer_scope(self, files: list[str]) -> str | None:
        """Infer the conventional-commit scope from changed file paths.

        Args:
            files: List of changed file paths.

        Returns:
            A single scope string when all files share one top-level directory,
            otherwise ``None``.
        """
        dirs: set[str] = set()
        for f in files:
            parts = f.split("/")
            if len(parts) > 1:
                dirs.add(parts[0] if parts[0] != "vetinari" else parts[1] if len(parts) > 2 else parts[0])
        if len(dirs) == 1:
            return dirs.pop()
        return None

    def _common_directory(self, files: list[str]) -> str:
        """Find the common directory prefix shared by all *files*.

        Args:
            files: List of file paths.

        Returns:
            The common directory path string, or an empty string when there is
            no shared prefix.
        """
        if not files:
            return ""
        parts = [f.split("/") for f in files]
        common: list[str] = []
        for segments in zip(*parts):
            if len(set(segments)) == 1:
                common.append(segments[0])
            else:
                break
        return "/".join(common)


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class GitOperationsTool(Tool):
    """Vetinari Tool wrapper around :class:`GitOperations`."""

    def __init__(self, repo_path: str):
        self._git = GitOperations(repo_path)
        metadata = ToolMetadata(
            name="git_operations",
            description="Version control operations via git CLI",
            version="1.0.0",
            category=ToolCategory.GIT_OPERATIONS,
            required_permissions=[],
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
        """Dispatch a git operation based on the ``operation`` keyword argument.

        Args:
            **kwargs: Must include ``operation`` (str) plus any operation-specific
                parameters such as ``message``, ``branch``, ``files``, or ``n``.

        Returns:
            ToolResult with the operation output or an error description.
        """
        op = kwargs.get("operation", "")

        # Enforce granular git permissions.
        # Read-only ops (status, log, diff, current_branch) require GIT_READ —
        # they are safe in PLANNING mode.
        # Write ops (commit, add, branch, checkout, stash, tag, pull, init) require
        # GIT_COMMIT — only allowed in EXECUTION mode.
        # push is irreversible and requires GIT_PUSH, which also requires confirmation.
        _ctx = get_context_manager()
        if op in ("status", "log", "diff", "current_branch"):
            _ctx.enforce_permission(ToolPermission.GIT_READ, f"git_{op}")
        elif op in ("commit", "add", "init", "stash", "tag", "branch", "checkout", "pull"):
            _ctx.enforce_permission(ToolPermission.GIT_COMMIT, f"git_{op}")
        elif op == "push":
            _ctx.enforce_permission(ToolPermission.GIT_PUSH, "git_push")

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
