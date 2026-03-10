"""Tests for vetinari.git_workflow module.

Covers dataclass formatting, change classification, commit message generation,
branch helpers, PR description generation, conflict detection, and repo status.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.git_workflow import (
    COMMIT_TYPES,
    BranchInfo,
    CommitInfo,
    ConflictInfo,
    GitWorkflow,
    get_git_workflow,
)


# ------------------------------------------------------------------
# CommitInfo formatting
# ------------------------------------------------------------------


class TestCommitInfo:
    """Tests for CommitInfo.format_message()."""

    def test_simple_message(self):
        ci = CommitInfo(type="feat", description="add login page")
        assert ci.format_message() == "feat: add login page"

    def test_message_with_scope(self):
        ci = CommitInfo(type="fix", scope="auth", description="resolve token expiry")
        assert ci.format_message() == "fix(auth): resolve token expiry"

    def test_message_with_breaking(self):
        ci = CommitInfo(type="feat", description="change API", breaking=True)
        assert ci.format_message() == "feat!: change API"

    def test_message_with_scope_and_breaking(self):
        ci = CommitInfo(
            type="refactor", scope="db", description="drop legacy table", breaking=True
        )
        assert ci.format_message() == "refactor(db)!: drop legacy table"

    def test_message_with_body(self):
        ci = CommitInfo(type="docs", description="update README", body="Added usage section.")
        msg = ci.format_message()
        assert msg.startswith("docs: update README")
        assert "\n\nAdded usage section." in msg

    def test_default_fields(self):
        ci = CommitInfo(type="chore")
        assert ci.scope is None
        assert ci.description == ""
        assert ci.body == ""
        assert ci.breaking is False
        assert ci.files_changed == []

    def test_files_changed_tracked(self):
        ci = CommitInfo(type="feat", files_changed=["a.py", "b.py"])
        assert len(ci.files_changed) == 2


# ------------------------------------------------------------------
# COMMIT_TYPES completeness
# ------------------------------------------------------------------


class TestCommitTypes:
    """Verify COMMIT_TYPES dict covers the standard set."""

    EXPECTED = {"feat", "fix", "refactor", "docs", "test", "chore", "style", "perf", "ci"}

    def test_all_types_present(self):
        assert set(COMMIT_TYPES.keys()) == self.EXPECTED

    def test_descriptions_non_empty(self):
        for key, desc in COMMIT_TYPES.items():
            assert isinstance(desc, str) and len(desc) > 0, f"{key} has empty description"


# ------------------------------------------------------------------
# BranchInfo dataclass
# ------------------------------------------------------------------


class TestBranchInfo:
    def test_defaults(self):
        bi = BranchInfo(name="main")
        assert bi.name == "main"
        assert bi.is_current is False
        assert bi.ahead == 0
        assert bi.behind == 0
        assert bi.last_commit == ""
        assert bi.created_from == ""

    def test_custom_values(self):
        bi = BranchInfo(name="feature/x", is_current=True, ahead=3, behind=1)
        assert bi.is_current is True
        assert bi.ahead == 3
        assert bi.behind == 1


# ------------------------------------------------------------------
# ConflictInfo dataclass
# ------------------------------------------------------------------


class TestConflictInfo:
    def test_defaults(self):
        ci = ConflictInfo(file_path="src/app.py", conflict_type="content")
        assert ci.file_path == "src/app.py"
        assert ci.conflict_type == "content"
        assert ci.ours_content == ""
        assert ci.theirs_content == ""
        assert ci.suggestion == ""

    def test_rename_conflict(self):
        ci = ConflictInfo(file_path="old.py", conflict_type="rename", suggestion="keep ours")
        assert ci.conflict_type == "rename"
        assert ci.suggestion == "keep ours"


# ------------------------------------------------------------------
# classify_changes
# ------------------------------------------------------------------


class TestClassifyChanges:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_test_files(self):
        assert self.gw.classify_changes("tests/test_foo.py | 10 +") == "test"

    def test_docs_files(self):
        assert self.gw.classify_changes("README.md | 5 +") == "docs"

    def test_ci_files(self):
        assert self.gw.classify_changes(".github/workflows/ci.yml | 2 +") == "ci"

    def test_fix_keyword(self):
        assert self.gw.classify_changes("bugfix/auth.py | 3 +") == "fix"

    def test_refactor_keyword(self):
        assert self.gw.classify_changes("refactor_utils.py | 8 +") == "refactor"

    def test_perf_keyword(self):
        assert self.gw.classify_changes("perf_cache.py | 4 +") == "perf"

    def test_default_feat(self):
        assert self.gw.classify_changes("src/new_feature.py | 20 +") == "feat"

    def test_empty_text_triggers_git(self):
        """When diff_text is None, _run_git is called."""
        with patch.object(self.gw, "_run_git", return_value=(0, "src/app.py | 5 +", "")):
            result = self.gw.classify_changes(None)
        assert result == "feat"

    def test_docs_markdown_extension(self):
        assert self.gw.classify_changes("guide.md | 12 +") == "docs"

    def test_pipeline_keyword(self):
        assert self.gw.classify_changes("pipeline_config.py | 1 +") == "ci"


# ------------------------------------------------------------------
# Internal helpers: _infer_scope, _common_directory, _infer_description
# ------------------------------------------------------------------


class TestInternalHelpers:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    # _infer_scope -------------------------------------------------

    def test_infer_scope_single_dir(self):
        assert self.gw._infer_scope(["src/a.py", "src/b.py"]) == "src"

    def test_infer_scope_multiple_dirs(self):
        assert self.gw._infer_scope(["src/a.py", "lib/b.py"]) is None

    def test_infer_scope_empty(self):
        assert self.gw._infer_scope([]) is None

    def test_infer_scope_vetinari_prefix(self):
        """If the top-level dir is 'vetinari', use the next segment."""
        result = self.gw._infer_scope(["vetinari/agents/foo.py", "vetinari/agents/bar.py"])
        assert result == "agents"

    def test_infer_scope_flat_files(self):
        """Files without directory separators produce no scope."""
        assert self.gw._infer_scope(["a.py", "b.py"]) is None

    # _common_directory --------------------------------------------

    def test_common_directory_shared(self):
        assert self.gw._common_directory(["a/b/c.py", "a/b/d.py"]) == "a/b"

    def test_common_directory_no_common(self):
        assert self.gw._common_directory(["x/a.py", "y/b.py"]) == ""

    def test_common_directory_empty(self):
        assert self.gw._common_directory([]) == ""

    def test_common_directory_single(self):
        assert self.gw._common_directory(["foo/bar/baz.py"]) == "foo/bar/baz.py"

    # _infer_description -------------------------------------------

    def test_infer_description_no_files(self):
        assert self.gw._infer_description([], "") == "update files"

    def test_infer_description_single_file(self):
        desc = self.gw._infer_description(["vetinari/cli.py"], "")
        assert "cli.py" in desc

    def test_infer_description_common_dir(self):
        desc = self.gw._infer_description(["src/a.py", "src/b.py"], "")
        assert "src" in desc
        assert "2 files" in desc

    def test_infer_description_no_common_dir(self):
        desc = self.gw._infer_description(["x/a.py", "y/b.py"], "")
        assert "2 files" in desc


# ------------------------------------------------------------------
# generate_commit_message
# ------------------------------------------------------------------


class TestGenerateCommitMessage:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_returns_commit_info(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "src/app.py | 5 +", ""),   # diff --cached --stat
                (0, "src/app.py", ""),           # diff --cached --name-only
            ],
        ):
            ci = self.gw.generate_commit_message()
        assert isinstance(ci, CommitInfo)
        assert ci.type == "feat"
        assert ci.files_changed == ["src/app.py"]

    def test_explicit_description_and_scope(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "tests/test_x.py | 2 +", ""),
                (0, "tests/test_x.py", ""),
            ],
        ):
            ci = self.gw.generate_commit_message(description="add coverage", scope="tests")
        assert ci.description == "add coverage"
        assert ci.scope == "tests"
        assert ci.type == "test"

    def test_no_staged_changes(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "", ""),  # diff --cached --stat  (empty)
                (0, "", ""),  # diff --cached --name-only (empty)
                (0, "", ""),  # diff --stat fallback in classify_changes
            ],
        ):
            ci = self.gw.generate_commit_message()
        assert ci.description == "update files"
        assert ci.files_changed == []


# ------------------------------------------------------------------
# Branch helpers
# ------------------------------------------------------------------


class TestBranchHelpers:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_create_branch_success(self):
        with patch.object(self.gw, "_run_git", return_value=(0, "", "")):
            assert self.gw.create_branch("feature/x") is True

    def test_create_branch_failure(self):
        with patch.object(self.gw, "_run_git", return_value=(1, "", "error")):
            assert self.gw.create_branch("bad-name") is False

    def test_get_branches_parses_output(self):
        output = (
            "* main       abc1234 latest commit\n"
            "  feature/x  def5678 wip stuff\n"
        )
        with patch.object(self.gw, "_run_git", return_value=(0, output, "")):
            branches = self.gw.get_branches()
        assert len(branches) == 2
        assert branches[0].name == "main"
        assert branches[0].is_current is True
        assert branches[1].name == "feature/x"
        assert branches[1].is_current is False

    def test_get_branches_error(self):
        with patch.object(self.gw, "_run_git", return_value=(1, "", "error")):
            assert self.gw.get_branches() == []


# ------------------------------------------------------------------
# PR description generation
# ------------------------------------------------------------------


class TestPRDescription:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_structure(self):
        log_output = "abc1234 feat: add login\ndef5678 fix: token bug"
        stat_output = " 2 files changed, 50 insertions(+), 10 deletions(-)"
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, log_output, ""),
                (0, stat_output, ""),
            ],
        ):
            pr = self.gw.generate_pr_description("main")
        assert "title" in pr
        assert "body" in pr
        assert "commits" in pr
        assert pr["commits"] == 2

    def test_title_from_first_commit(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "abc feat: awesome feature", ""),
                (0, "", ""),
            ],
        ):
            pr = self.gw.generate_pr_description()
        assert pr["title"] == "feat: awesome feature"

    def test_no_commits(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "", ""),
                (0, "", ""),
            ],
        ):
            pr = self.gw.generate_pr_description()
        assert pr["title"] == "Update"
        assert pr["commits"] == 0

    def test_title_truncated(self):
        long_msg = "abc " + "x" * 100
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, long_msg, ""),
                (0, "", ""),
            ],
        ):
            pr = self.gw.generate_pr_description()
        assert len(pr["title"]) <= 70

    def test_body_contains_summary_and_changes(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "abc feat: one\ndef fix: two", ""),
                (0, "stat output", ""),
            ],
        ):
            pr = self.gw.generate_pr_description()
        assert "## Summary" in pr["body"]
        assert "## Changes" in pr["body"]
        assert "- feat: one" in pr["body"]
        assert "- fix: two" in pr["body"]


# ------------------------------------------------------------------
# Conflict detection
# ------------------------------------------------------------------


class TestConflictDetection:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_no_conflicts(self):
        with patch.object(self.gw, "_run_git", return_value=(0, "", "")):
            assert self.gw.detect_conflicts("main") == []

    def test_content_conflict(self):
        stderr = "CONFLICT (content): Merge conflict in src/app.py\nAutomatic merge failed."
        with patch.object(self.gw, "_run_git", return_value=(1, "", stderr)):
            conflicts = self.gw.detect_conflicts("main")
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "content"
        assert "src/app.py" in conflicts[0].file_path

    def test_rename_conflict(self):
        stderr = "CONFLICT (rename/delete): old.py renamed to new.py"
        with patch.object(self.gw, "_run_git", return_value=(1, "", stderr)):
            conflicts = self.gw.detect_conflicts("main")
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "rename"

    def test_delete_conflict(self):
        stderr = "CONFLICT (modify/delete): removed.py deleted in HEAD"
        with patch.object(self.gw, "_run_git", return_value=(1, "", stderr)):
            conflicts = self.gw.detect_conflicts("main")
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "delete"

    def test_multiple_conflicts(self):
        stderr = (
            "CONFLICT (content): Merge conflict in a.py\n"
            "CONFLICT (content): Merge conflict in b.py\n"
            "Automatic merge failed."
        )
        with patch.object(self.gw, "_run_git", return_value=(1, "", stderr)):
            conflicts = self.gw.detect_conflicts("main")
        assert len(conflicts) == 2

    def test_non_conflict_error(self):
        """A non-zero exit without 'conflict' in stderr yields no conflicts."""
        with patch.object(self.gw, "_run_git", return_value=(1, "", "fatal: not a git repo")):
            assert self.gw.detect_conflicts("main") == []


# ------------------------------------------------------------------
# get_status in a real git repo
# ------------------------------------------------------------------


class TestGetStatus:
    def setup_method(self):
        self.gw = GitWorkflow.__new__(GitWorkflow)
        self.gw._repo_path = "."

    def test_status_structure(self):
        porcelain = " M vetinari/cli.py\n?? new_file.py"
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, porcelain, ""),
                (0, "main", ""),
                (0, "abc1234", ""),
            ],
        ):
            st = self.gw.get_status()
        assert st["branch"] == "main"
        assert st["commit"] == "abc1234"
        assert "vetinari/cli.py" in st["files"]["modified"]
        assert "new_file.py" in st["files"]["untracked"]
        assert st["clean"] is False

    def test_clean_repo(self):
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, "", ""),
                (0, "main", ""),
                (0, "abc1234", ""),
            ],
        ):
            st = self.gw.get_status()
        assert st["clean"] is True
        assert st["files"]["modified"] == []

    def test_added_and_deleted(self):
        porcelain = "A  added.py\n D deleted.py"
        with patch.object(
            self.gw,
            "_run_git",
            side_effect=[
                (0, porcelain, ""),
                (0, "dev", ""),
                (0, "fff0000", ""),
            ],
        ):
            st = self.gw.get_status()
        assert "added.py" in st["files"]["added"]
        assert "deleted.py" in st["files"]["deleted"]


# ------------------------------------------------------------------
# Real git repo integration test
# ------------------------------------------------------------------


class TestRealGitRepo:
    """Integration-level test using the actual repository."""

    def test_get_status_live(self):
        """get_status returns valid data in this real git repo."""
        # Use the actual worktree root
        repo_root = str(Path(__file__).parent.parent)
        gw = GitWorkflow(repo_path=repo_root)
        st = gw.get_status()
        assert "branch" in st
        assert "commit" in st
        assert "files" in st
        assert isinstance(st["branch"], str)
        assert len(st["branch"]) > 0

    def test_get_branches_live(self):
        """get_branches returns at least one branch."""
        repo_root = str(Path(__file__).parent.parent)
        gw = GitWorkflow(repo_path=repo_root)
        branches = gw.get_branches()
        assert len(branches) >= 1
        current = [b for b in branches if b.is_current]
        assert len(current) == 1


# ------------------------------------------------------------------
# _run_git error handling
# ------------------------------------------------------------------


class TestRunGit:
    def test_timeout_returns_error_tuple(self):
        gw = GitWorkflow(repo_path=".")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
            code, stdout, stderr = gw._run_git("status")
        assert code == 1
        assert stdout == ""

    def test_file_not_found_returns_error_tuple(self):
        gw = GitWorkflow(repo_path=".")
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            code, stdout, stderr = gw._run_git("status")
        assert code == 1
        assert "git not found" in stderr


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------


class TestGetGitWorkflow:
    def test_returns_git_workflow(self):
        import vetinari.git_workflow as mod

        # Reset the module-level singleton
        mod._git_workflow = None
        gw = get_git_workflow("/tmp")
        assert isinstance(gw, GitWorkflow)

    def test_singleton_reuse(self):
        import vetinari.git_workflow as mod

        mod._git_workflow = None
        gw1 = get_git_workflow()
        gw2 = get_git_workflow()
        assert gw1 is gw2

    def test_singleton_reset(self):
        import vetinari.git_workflow as mod

        mod._git_workflow = None
        gw1 = get_git_workflow("/a")
        mod._git_workflow = None
        gw2 = get_git_workflow("/b")
        assert gw1 is not gw2
