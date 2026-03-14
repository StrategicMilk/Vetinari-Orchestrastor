"""Enhanced Git Workflow for Vetinari.

Provides conventional commit generation, branch management, PR description
generation, and conflict detection. Used by DOCUMENTER agent in git mode.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Conventional commit types
COMMIT_TYPES = {
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

    def format_message(self) -> str:
        """Format as a conventional commit message string.

        Returns:
            The result string.
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


@dataclass
class ConflictInfo:
    """Information about a merge conflict."""

    file_path: str
    conflict_type: str  # "content", "rename", "delete"
    ours_content: str = ""
    theirs_content: str = ""
    suggestion: str = ""


class GitWorkflow:
    """Enhanced git workflow utilities.

    Wraps git CLI operations and provides higher-level helpers for
    conventional commits, branch management, PR descriptions, and
    conflict detection.
    """

    def __init__(self, repo_path: str = "."):
        self._repo_path = repo_path

    def _run_git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command. Returns (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(  # noqa: S603
                ["git", *list(args)],  # noqa: S607
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return 1, "", str(e)

    # ------------------------------------------------------------------
    # Change classification
    # ------------------------------------------------------------------

    def classify_changes(self, diff_text: str | None = None) -> str:
        """Classify changes into a conventional commit type.

        If *diff_text* is ``None`` the method inspects the currently staged
        changes, falling back to unstaged changes.

        Returns:
            The result string.
        """
        if diff_text is None:
            _, diff_text, _ = self._run_git("diff", "--cached", "--stat")

        if not diff_text:
            _, diff_text, _ = self._run_git("diff", "--stat")

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

    # ------------------------------------------------------------------
    # Commit helpers
    # ------------------------------------------------------------------

    def generate_commit_message(self, description: str | None = None, scope: str | None = None) -> CommitInfo:
        """Generate a conventional commit message from staged changes.

        Args:
            description: The description.
            scope: The scope.

        Returns:
            The CommitInfo result.
        """
        _, diff_stat, _ = self._run_git("diff", "--cached", "--stat")
        _, diff_names, _ = self._run_git("diff", "--cached", "--name-only")

        files = [f.strip() for f in diff_names.split("\n") if f.strip()] if diff_names else []
        commit_type = self.classify_changes(diff_stat)

        if not description:
            description = self._infer_description(files, diff_stat)

        if not scope and files:
            scope = self._infer_scope(files)

        return CommitInfo(
            type=commit_type,
            scope=scope,
            description=description,
            files_changed=files,
        )

    def _infer_description(self, files: list[str], diff_stat: str) -> str:
        """Infer a commit description from changed files."""
        if not files:
            return "update files"
        if len(files) == 1:
            return f"update {files[0].split('/')[-1]}"
        common_dir = self._common_directory(files)
        if common_dir:
            return f"update {common_dir} ({len(files)} files)"
        return f"update {len(files)} files"

    def _infer_scope(self, files: list[str]) -> str | None:
        """Infer scope from file paths."""
        dirs: set = set()
        for f in files:
            parts = f.split("/")
            if len(parts) > 1:
                dirs.add(parts[0] if parts[0] != "vetinari" else parts[1] if len(parts) > 2 else parts[0])
        if len(dirs) == 1:
            return dirs.pop()
        return None

    def _common_directory(self, files: list[str]) -> str:
        """Find common directory prefix among *files*."""
        if not files:
            return ""
        parts = [f.split("/") for f in files]
        common: list = []
        for segments in zip(*parts):
            if len(set(segments)) == 1:
                common.append(segments[0])
            else:
                break
        return "/".join(common)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def create_branch(self, name: str, base: str = "HEAD") -> bool:
        """Create and checkout a new branch from *base*.

        Args:
            name: The name.
            base: The base.

        Returns:
            True if successful, False otherwise.
        """
        code, _, _ = self._run_git("checkout", "-b", name, base)
        return code == 0

    def get_branches(self) -> list[BranchInfo]:
        """List all local branches with metadata.

        Returns:
            List of results.
        """
        code, output, _ = self._run_git("branch", "-v")
        branches: list = []
        if code != 0:
            return branches
        for line in output.split("\n"):
            if not line.strip():
                continue
            is_current = line.startswith("*")
            line = line.lstrip("* ").strip()
            parts = line.split(None, 2)
            if len(parts) >= 2:
                branches.append(
                    BranchInfo(
                        name=parts[0],
                        is_current=is_current,
                        last_commit=parts[1] if len(parts) > 1 else "",
                    )
                )
        return branches

    # ------------------------------------------------------------------
    # PR description generation
    # ------------------------------------------------------------------

    def generate_pr_description(self, base_branch: str = "main") -> dict[str, str]:
        """Generate a PR description from commit history since *base_branch*.

        Returns:
            The result string.
        """
        _, log_output, _ = self._run_git("log", f"{base_branch}..HEAD", "--oneline")
        _, diff_stat, _ = self._run_git("diff", f"{base_branch}..HEAD", "--stat")

        commits = [l.strip() for l in log_output.split("\n") if l.strip()] if log_output else []  # noqa: E741

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

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(self, target_branch: str = "main") -> list[ConflictInfo]:
        """Detect potential merge conflicts with *target_branch*.

        Returns:
            List of results.
        """
        code, _, stderr = self._run_git("merge", "--no-commit", "--no-ff", target_branch, "--dry-run")

        conflicts: list = []
        if code != 0 and "conflict" in stderr.lower():
            for line in stderr.split("\n"):
                if "CONFLICT" in line:
                    match = re.search(r"CONFLICT.*?:\s*(.+)", line)
                    file_path = match.group(1).strip() if match else "unknown"
                    conflict_type = "content"
                    if "rename" in line.lower():
                        conflict_type = "rename"
                    elif "delete" in line.lower():
                        conflict_type = "delete"
                    conflicts.append(
                        ConflictInfo(
                            file_path=file_path,
                            conflict_type=conflict_type,
                        )
                    )
        return conflicts

    # ------------------------------------------------------------------
    # Repository status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current repository status as a structured dict.

        Returns:
            The result string.
        """
        _, status, _ = self._run_git("status", "--porcelain")
        _, branch, _ = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        _, commit, _ = self._run_git("rev-parse", "--short", "HEAD")

        files: dict[str, list[str]] = {
            "modified": [],
            "added": [],
            "deleted": [],
            "untracked": [],
        }
        for line in (status or "").split("\n"):
            if not line.strip():
                continue
            code_part = line[:2]
            path = line[3:].strip()
            if "M" in code_part:
                files["modified"].append(path)
            elif "A" in code_part:
                files["added"].append(path)
            elif "D" in code_part:
                files["deleted"].append(path)
            elif "?" in code_part:
                files["untracked"].append(path)

        return {
            "branch": branch,
            "commit": commit,
            "files": files,
            "clean": not any(files.values()),
        }


# ------------------------------------------------------------------
# Module-level singleton accessor
# ------------------------------------------------------------------

_git_workflow: GitWorkflow | None = None


def get_git_workflow(repo_path: str = ".") -> GitWorkflow:
    """Return a module-level :class:`GitWorkflow` singleton.

    Returns:
        The GitWorkflow result.
    """
    global _git_workflow
    if _git_workflow is None:
        _git_workflow = GitWorkflow(repo_path)
    return _git_workflow
