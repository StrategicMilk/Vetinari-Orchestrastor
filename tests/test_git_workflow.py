"""Tests for vetinari.git_workflow — conventional commits and branch management."""

from __future__ import annotations

from vetinari.git_workflow import (
    COMMIT_TYPES,
    BranchInfo,
    CommitInfo,
    ConflictInfo,
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
