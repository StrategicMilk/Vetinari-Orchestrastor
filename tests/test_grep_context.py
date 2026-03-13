"""Tests for vetinari.grep_context — surgical code extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vetinari.grep_context import GrepContext, GrepMatch


class TestGrepMatch:
    """Tests for the GrepMatch dataclass."""

    def test_fields(self):
        m = GrepMatch(file_path="a.py", line_number=10, line_content="x = 1")
        assert m.file_path == "a.py"
        assert m.line_number == 10
        assert m.line_content == "x = 1"

    def test_defaults(self):
        m = GrepMatch("f.py", 1, "code")
        assert m.context_before == []
        assert m.context_after == []


class TestGrepContext:
    """Tests for the GrepContext class."""

    def test_backend_property(self):
        gc = GrepContext()
        assert gc.backend in ("ripgrep", "python-re")

    def test_extract_patterns_empty_inputs(self):
        gc = GrepContext()
        assert gc.extract_patterns([], ["pattern"]) == []
        assert gc.extract_patterns(["file.py"], []) == []

    def test_extract_patterns_nonexistent_files(self):
        gc = GrepContext()
        result = gc.extract_patterns(["/nonexistent/file.py"], ["def"])
        assert result == []

    def test_extract_patterns_finds_matches(self):
        gc = GrepContext()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("import os\n")
            f.write("def hello():\n")
            f.write("    return 'world'\n")
            f.write("class Foo:\n")
            f.write("    pass\n")
            f.flush()
            fname = f.name

        results = gc.extract_patterns([fname], [r"def \w+"], context_lines=1)
        assert len(results) >= 1
        assert any("hello" in m.line_content for m in results)

    def test_extract_definitions(self):
        gc = GrepContext()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("x = 1\n")
            f.write("def my_func():\n")
            f.write("    return 42\n")
            f.write("\n")
            f.write("class MyClass:\n")
            f.write("    pass\n")
            f.flush()
            fname = f.name

        result = gc.extract_definitions(fname, ["my_func"])
        assert "my_func" in result
