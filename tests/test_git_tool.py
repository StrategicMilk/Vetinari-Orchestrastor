"""Tests for vetinari.tools.git_tool — git operations, conventional commits, and branch management."""

from __future__ import annotations

from vetinari.tools.git_tool import (
    COMMIT_TYPES,
    BranchInfo,
    CommitInfo,
    ConflictInfo,
    GitOperations,
)


class TestCommitTypes:
    """Tests for the COMMIT_TYPES constant."""

    def test_standard_types_present(self):
        for t in ("feat", "fix", "refactor", "docs", "test", "chore"):
            assert t in COMMIT_TYPES

    def test_all_values_are_strings(self):
        for key, value in COMMIT_TYPES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestCommitInfo:
    """Tests for the CommitInfo dataclass."""

    def test_format_message_simple(self):
        ci = CommitInfo(type="feat", description="add login page")
        assert ci.format_message() == "feat: add login page"

    def test_format_message_with_scope(self):
        ci = CommitInfo(type="fix", scope="auth", description="token expiry bug")
        assert ci.format_message() == "fix(auth): token expiry bug"

    def test_format_message_breaking(self):
        ci = CommitInfo(type="feat", scope="api", description="change response format", breaking=True)
        msg = ci.format_message()
        assert "!" in msg
        assert msg.startswith("feat(api)!")

    def test_format_message_with_body(self):
        ci = CommitInfo(type="docs", description="update README", body="Added install section")
        msg = ci.format_message()
        assert "docs: update README" in msg
        assert "Added install section" in msg
        assert "\n\n" in msg

    def test_files_changed_default(self):
        ci = CommitInfo(type="fix", description="bug")
        assert ci.files_changed == []


class TestBranchInfo:
    """Tests for the BranchInfo dataclass."""

    def test_defaults(self):
        bi = BranchInfo(name="feature/login")
        assert bi.is_current is False
        assert bi.ahead == 0
        assert bi.behind == 0

    def test_fields(self):
        bi = BranchInfo(name="main", is_current=True, ahead=2, behind=1)
        assert bi.name == "main"
        assert bi.is_current is True
        assert bi.ahead == 2


class TestConflictInfo:
    """Tests for the ConflictInfo dataclass."""

    def test_fields(self):
        ci = ConflictInfo(file_path="src/app.py", conflict_type="content")
        assert ci.file_path == "src/app.py"
        assert ci.conflict_type == "content"
        assert ci.suggestion == ""

    def test_with_content(self):
        ci = ConflictInfo(
            file_path="f.py",
            conflict_type="content",
            ours_content="version A",
            theirs_content="version B",
            suggestion="merge both",
        )
        assert ci.ours_content == "version A"
        assert ci.theirs_content == "version B"


class TestClassifyChanges:
    """Tests for GitOperations.classify_changes()."""

    def setup_method(self):
        self.git = GitOperations(".")

    def test_detects_test_files(self):
        assert self.git.classify_changes("test_auth.py | 10 ++") == "test"

    def test_detects_docs(self):
        assert self.git.classify_changes("README.md | 5 +") == "docs"

    def test_detects_fix(self):
        assert self.git.classify_changes("hotfix applied to auth | 2 -") == "fix"

    def test_detects_refactor(self):
        assert self.git.classify_changes("refactor auth module | 20 +-") == "refactor"

    def test_detects_perf(self):
        assert self.git.classify_changes("optimise query cache | 8 +-") == "perf"

    def test_detects_ci(self):
        assert self.git.classify_changes(".github/workflows/ci.yml | 3 +") == "ci"

    def test_defaults_to_feat(self):
        assert self.git.classify_changes("some new module added | 50 +") == "feat"


class TestInferDescription:
    """Tests for GitOperations._infer_description()."""

    def setup_method(self):
        self.git = GitOperations(".")

    def test_no_files(self):
        assert self.git._infer_description([]) == "update files"

    def test_single_file(self):
        result = self.git._infer_description(["vetinari/tools/git_tool.py"])
        assert "git_tool.py" in result

    def test_multiple_files_same_dir(self):
        result = self.git._infer_description(["vetinari/tools/a.py", "vetinari/tools/b.py"])
        assert "2 files" in result

    def test_multiple_files_no_common_dir(self):
        result = self.git._infer_description(["a/x.py", "b/y.py"])
        assert "2 files" in result


class TestInferScope:
    """Tests for GitOperations._infer_scope()."""

    def setup_method(self):
        self.git = GitOperations(".")

    def test_single_top_dir(self):
        result = self.git._infer_scope(["agents/planner.py", "agents/runner.py"])
        assert result == "agents"

    def test_vetinari_prefix_uses_second_level(self):
        result = self.git._infer_scope(["vetinari/tools/a.py", "vetinari/tools/b.py"])
        assert result == "tools"

    def test_mixed_dirs_returns_none(self):
        result = self.git._infer_scope(["vetinari/tools/a.py", "vetinari/agents/b.py"])
        assert result is None

    def test_empty_returns_none(self):
        result = self.git._infer_scope([])
        assert result is None


class TestGeneratePrDescription:
    """Tests for GitOperations.generate_pr_description() shape."""

    def setup_method(self):
        self.git = GitOperations(".")

    def test_returns_dict_with_required_keys(self, monkeypatch):
        def fake_run(args, timeout=None):
            from vetinari.tools.git_tool import GitResult

            if "--oneline" in args:
                return GitResult(success=True, stdout="abc1234 feat: add login\ndef5678 fix: token bug")
            return GitResult(success=True, stdout="2 files changed")

        monkeypatch.setattr(self.git, "_run", fake_run)
        result = self.git.generate_pr_description()
        assert "title" in result
        assert "body" in result
        assert "commits" in result
        assert result["commits"] == 2
        assert len(result["title"]) <= 70

    def test_empty_repo_returns_update_title(self, monkeypatch):
        from vetinari.tools.git_tool import GitResult

        monkeypatch.setattr(self.git, "_run", lambda *a, **kw: GitResult(success=True, stdout=""))
        result = self.git.generate_pr_description()
        assert result["title"] == "Update"
        assert result["commits"] == 0
