"""Tests for grep-based context extraction."""

import os
import tempfile
import pytest
from vetinari.grep_context import GrepContext, GrepMatch, get_grep_context


@pytest.fixture
def sample_py_file(tmp_path):
    """Create a sample Python file for testing."""
    code = '''import os
import json

class DataProcessor:
    """Processes data from various sources."""

    def __init__(self, config):
        self.config = config
        self.api_key = os.environ.get("API_KEY")

    def process(self, data):
        """Process the input data."""
        result = []
        for item in data:
            if item.get("valid"):
                result.append(self._transform(item))
        return result

    def _transform(self, item):
        return {"id": item["id"], "value": item["value"] * 2}

def helper_function():
    password = "should_not_be_hardcoded"
    return eval(password)
'''
    f = tmp_path / "sample.py"
    f.write_text(code)
    return str(f)


class TestGrepContext:
    def test_init(self):
        gc = GrepContext()
        assert gc.backend in ("ripgrep", "python-re")

    def test_extract_patterns(self, sample_py_file):
        gc = GrepContext()
        matches = gc.extract_patterns(
            [sample_py_file],
            [r"api_key|password"],
            context_lines=1,
        )
        assert len(matches) >= 2
        assert any("api_key" in m.line_content.lower() for m in matches)

    def test_extract_definitions(self, sample_py_file):
        gc = GrepContext()
        result = gc.extract_definitions(sample_py_file, ["process", "DataProcessor"])
        assert "def process" in result
        assert "class DataProcessor" in result

    def test_extract_imports(self, sample_py_file):
        gc = GrepContext()
        imports = gc.extract_imports(sample_py_file)
        assert "import os" in imports
        assert "import json" in imports

    def test_extract_security_patterns(self, sample_py_file):
        gc = GrepContext()
        matches = gc.extract_security_patterns([sample_py_file])
        assert len(matches) >= 2  # api_key, password, eval

    def test_extract_relevant_context(self, sample_py_file):
        gc = GrepContext()
        ctx = gc.extract_relevant_context(sample_py_file, ["process"], budget_chars=500)
        assert "process" in ctx.lower()
        assert len(ctx) <= 600  # Allow some margin

    def test_format_for_prompt(self):
        gc = GrepContext()
        matches = [
            GrepMatch("test.py", 10, "x = 1", ["# before"], ["# after"]),
            GrepMatch("test.py", 20, "y = 2", [], []),
        ]
        result = gc.format_for_prompt(matches, max_chars=500)
        assert "test.py:10" in result
        assert "x = 1" in result

    def test_budget_respected(self, sample_py_file):
        gc = GrepContext()
        ctx = gc.extract_relevant_context(sample_py_file, ["process"], budget_chars=100)
        assert len(ctx) <= 200  # Allow some overhead for markers

    def test_empty_inputs(self):
        gc = GrepContext()
        assert gc.extract_patterns([], ["test"]) == []
        assert gc.extract_patterns(["nonexistent.py"], ["test"]) == []
        assert gc.extract_relevant_context("nonexistent.py", ["test"]) == ""

    def test_singleton(self):
        g1 = get_grep_context()
        g2 = get_grep_context()
        assert g1 is g2
