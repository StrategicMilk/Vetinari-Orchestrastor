"""Direct behavior tests for scripts/maintenance/validate_vetinari.py.

Exercises the validate_vetinari helper's search-tool check and the
check() pass/fail recording mechanism. These are behavior tests, not
import-only certification — each test exercises a real code path.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "maintenance"


@pytest.fixture
def validate_mod():
    """Import validate_vetinari as a live module, then clean up after each test.

    Inserts scripts/ into sys.path so the script is importable, reloads
    it on each use to avoid stale module-level state between tests, and
    removes both the path entry and the cached module in teardown.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        mod = importlib.import_module("validate_vetinari")
        importlib.reload(mod)
        yield mod
    finally:
        sys.path.pop(0)
        sys.modules.pop("validate_vetinari", None)


@pytest.fixture(autouse=True)
def _reset_state(validate_mod: object) -> None:
    """Clear module-level error and pass lists before every test.

    validate_vetinari uses module-level mutable lists; without a reset
    each test would inherit accumulated state from prior tests.
    """
    validate_mod.errors.clear()  # type: ignore[attr-defined]
    validate_mod.passed.clear()  # type: ignore[attr-defined]


class TestSearchToolCheck:
    """Tests that validate_vetinari.test_search_tool() uses the canonical backend."""

    def test_search_tool_uses_canonical_backend(self, validate_mod: object) -> None:
        """test_search_tool() must not raise and must validate the canonical backend.

        Calls the function directly, then independently verifies the same
        contract so a regression in either the helper or the tool is caught.
        """
        # The function itself must not raise
        validate_mod.test_search_tool()  # type: ignore[attr-defined]

        # Independent verification: the tool must report the canonical backend
        from vetinari.constants import DEFAULT_SEARCH_BACKEND
        from vetinari.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool(backend=DEFAULT_SEARCH_BACKEND)
        assert tool.backend_name == DEFAULT_SEARCH_BACKEND, (
            f"WebSearchTool.backend_name should be '{DEFAULT_SEARCH_BACKEND}', got '{tool.backend_name}'"
        )

    def test_search_tool_not_hardcoded_duckduckgo(self) -> None:
        """Regression guard: 'duckduckgo' must not appear in test_search_tool().

        Reads the source of validate_vetinari.py and isolates the body of
        test_search_tool so the guard cannot be tripped by unrelated functions
        that legitimately mention the string.
        """
        source = (SCRIPTS_DIR / "validate_vetinari.py").read_text(encoding="utf-8")

        # Extract just the test_search_tool function body
        start = source.find("def test_search_tool()")
        assert start != -1, "test_search_tool function not found in validate_vetinari.py"

        # Find the next top-level function definition after test_search_tool
        next_def = source.find("\ndef ", start + 1)
        func_body = source[start:next_def] if next_def != -1 else source[start:]

        assert "duckduckgo" not in func_body, (
            "test_search_tool() still hardcodes 'duckduckgo' — "
            "it must use DEFAULT_SEARCH_BACKEND from vetinari.constants"
        )


class TestCheckMechanism:
    """Tests for the check() pass/fail recording helper."""

    def test_check_function_records_pass(self, validate_mod: object) -> None:
        """check() appends the check name to passed when the callable succeeds."""
        validate_mod.check("test-pass", lambda: None)  # type: ignore[attr-defined]

        assert "test-pass" in validate_mod.passed  # type: ignore[attr-defined]
        assert len(validate_mod.errors) == 0  # type: ignore[attr-defined]

    def test_check_function_records_fail(self, validate_mod: object) -> None:
        """check() appends the check name + error to errors when the callable raises."""
        validate_mod.check("test-fail", lambda: 1 / 0)  # type: ignore[attr-defined]

        failed_names = [name for name, _ in validate_mod.errors]  # type: ignore[attr-defined]
        assert "test-fail" in failed_names
        assert "test-fail" not in validate_mod.passed  # type: ignore[attr-defined]


class TestCoreImportsCheck:
    """Tests that the three-agent factory imports are intact."""

    def test_core_imports_check(self, validate_mod: object) -> None:
        """test_imports() must not raise, confirming the 3-agent factory is importable.

        This is the minimum smoke test for the Foreman -> Worker -> Inspector
        pipeline being intact in the current environment.
        """
        # Should not raise; if it does the agent imports are broken
        validate_mod.test_imports()  # type: ignore[attr-defined]
        assert validate_mod.errors == []  # type: ignore[attr-defined]
