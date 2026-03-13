"""Tests for vetinari.code_search — code search adapters and results."""

from __future__ import annotations

from vetinari.code_search import (
    CodeSearchResult,
    SearchBackendStatus,
)


class TestSearchBackendStatus:
    """Tests for the SearchBackendStatus enum."""

    def test_values(self):
        assert SearchBackendStatus.AVAILABLE.value == "available"
        assert SearchBackendStatus.UNAVAILABLE.value == "unavailable"
        assert SearchBackendStatus.INDEXING.value == "indexing"
        assert SearchBackendStatus.ERROR.value == "error"

    def test_member_count(self):
        assert len(SearchBackendStatus) == 4


class TestCodeSearchResult:
    """Tests for the CodeSearchResult dataclass."""

    def test_fields(self):
        result = CodeSearchResult(
            file_path="src/main.py",
            language="python",
            content="def hello():",
            line_start=10,
            line_end=15,
            score=0.95,
        )
        assert result.file_path == "src/main.py"
        assert result.language == "python"
        assert result.score == 0.95

    def test_defaults(self):
        result = CodeSearchResult("f.py", "py", "x", 1, 1, 0.5)
        assert result.context_before == ""
        assert result.context_after == ""

    def test_to_dict(self):
        result = CodeSearchResult(
            file_path="a.py",
            language="python",
            content="code",
            line_start=1,
            line_end=5,
            score=0.8,
            context_before="before",
            context_after="after",
        )
        d = result.to_dict()
        assert d["file_path"] == "a.py"
        assert d["score"] == 0.8
        assert d["context_before"] == "before"
        assert d["context_after"] == "after"

    def test_to_dict_keys(self):
        result = CodeSearchResult("f.py", "py", "x", 1, 2, 0.5)
        d = result.to_dict()
        expected_keys = {
            "file_path", "language", "content",
            "line_start", "line_end", "score",
            "context_before", "context_after",
        }
        assert set(d.keys()) == expected_keys
