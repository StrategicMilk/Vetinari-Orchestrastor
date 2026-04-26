"""Tests for vetinari.project.scanner.

Verifies that the scanner module is importable and that its public functions
produce correct output when subprocess calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_subprocess_result_mock
from vetinari.project.scanner import (
    ScanFinding,
    ScanResult,
    _classify_ruff_code,
    _detect_test_presence,
    _run_dead_code_check,
    _run_ruff,
    prioritize_findings,
    scan_project,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal fake project tree for scanner tests."""
    (tmp_path / "mypackage").mkdir()
    (tmp_path / "mypackage" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "mypackage" / "core.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tmp_project_with_tests(tmp_project: Path) -> Path:
    """Extend tmp_project with a tests/ directory."""
    (tmp_project / "tests").mkdir()
    (tmp_project / "tests" / "test_core.py").write_text("def test_x(): pass\n", encoding="utf-8")
    return tmp_project


# ---------------------------------------------------------------------------
# _classify_ruff_code
# ---------------------------------------------------------------------------


def test_classify_ruff_code_security():
    cat, sev = _classify_ruff_code("S101")
    assert cat == "security"
    assert sev == "high"


def test_classify_ruff_code_bug():
    cat, sev = _classify_ruff_code("F401")
    assert cat == "bug"
    assert sev == "medium"


def test_classify_ruff_code_style():
    cat, sev = _classify_ruff_code("E501")
    assert cat == "style"
    assert sev == "low"


def test_classify_ruff_code_unknown():
    cat, sev = _classify_ruff_code("XYZ999")
    assert cat == "maintainability"
    assert sev == "low"


def test_classify_ruff_code_empty():
    cat, sev = _classify_ruff_code("")
    assert cat == "maintainability"
    assert sev == "low"


# ---------------------------------------------------------------------------
# _detect_test_presence
# ---------------------------------------------------------------------------


def test_detect_test_presence_with_tests_dir(tmp_project_with_tests: Path):
    assert _detect_test_presence(tmp_project_with_tests) is True


def test_detect_test_presence_without_tests(tmp_project: Path):
    assert _detect_test_presence(tmp_project) is False


def test_detect_test_presence_test_file_pattern(tmp_path: Path):
    """A test_*.py file in the source tree counts as having tests."""
    (tmp_path / "test_something.py").write_text("", encoding="utf-8")
    assert _detect_test_presence(tmp_path) is True


# ---------------------------------------------------------------------------
# prioritize_findings
# ---------------------------------------------------------------------------


def test_prioritize_findings_order():
    findings = [
        ScanFinding("style", "low", "a.py", 1, "style issue", "ruff"),
        ScanFinding("security", "high", "b.py", 5, "sql injection", "ruff"),
        ScanFinding("bug", "medium", "c.py", 10, "unused var", "ruff"),
    ]
    sorted_findings = prioritize_findings(findings)
    assert sorted_findings[0].category == "security"
    assert sorted_findings[1].category == "bug"
    assert sorted_findings[2].category == "style"


def test_prioritize_findings_empty():
    assert prioritize_findings([]) == []


def test_prioritize_findings_stable_within_category():
    """Within the same category and severity, sort by file path then line."""
    findings = [
        ScanFinding("style", "low", "z.py", 1, "msg", "ruff"),
        ScanFinding("style", "low", "a.py", 10, "msg", "ruff"),
        ScanFinding("style", "low", "a.py", 2, "msg", "ruff"),
    ]
    result = prioritize_findings(findings)
    assert result[0].file_path == "a.py"
    assert result[0].line == 2
    assert result[1].file_path == "a.py"
    assert result[1].line == 10
    assert result[2].file_path == "z.py"


# ---------------------------------------------------------------------------
# _run_ruff (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_ruff_parses_json_output(tmp_project: Path):
    fake_output = json.dumps([
        {
            "code": "E501",
            "filename": "mypackage/core.py",
            "location": {"row": 3},
            "message": "Line too long",
        }
    ])
    mock_result = MagicMock()
    mock_result.stdout = fake_output
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        findings = _run_ruff(tmp_project)

    assert findings is not None
    assert len(findings) == 1
    assert findings[0].tool == "ruff"
    assert findings[0].file_path == "mypackage/core.py"
    assert findings[0].line == 3
    assert findings[0].category == "style"


def test_run_ruff_returns_none_when_not_installed(tmp_project: Path):
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _run_ruff(tmp_project)
    assert result is None


def test_run_ruff_empty_stdout(tmp_project: Path):
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        findings = _run_ruff(tmp_project)

    assert findings == []


def test_run_ruff_invalid_json(tmp_project: Path):
    mock_result = MagicMock()
    mock_result.stdout = "not valid json {"
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        findings = _run_ruff(tmp_project)

    assert findings == []


# ---------------------------------------------------------------------------
# _run_dead_code_check (mocked subprocess)
# ---------------------------------------------------------------------------


def test_run_dead_code_check_parses_output(tmp_project: Path):
    fake_output = "mypackage/core.py:5: unused variable 'x' (confidence 80%)"
    mock_result = MagicMock()
    mock_result.stdout = fake_output
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        findings = _run_dead_code_check(tmp_project)

    assert findings is not None
    assert len(findings) == 1
    assert findings[0].tool == "vulture"
    assert findings[0].line == 5
    assert findings[0].is_reachable is False


def test_run_dead_code_check_none_when_not_installed(tmp_project: Path):
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _run_dead_code_check(tmp_project)
    assert result is None


# ---------------------------------------------------------------------------
# scan_project integration (mocked subprocess)
# ---------------------------------------------------------------------------


def test_scan_project_returns_scan_result(tmp_project: Path):
    ruff_output = json.dumps([
        {
            "code": "S101",
            "filename": "mypackage/core.py",
            "location": {"row": 1},
            "message": "Use of assert detected",
        }
    ])
    mock_result = MagicMock()
    mock_result.stdout = ruff_output
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        result = scan_project(tmp_project)

    assert isinstance(result, ScanResult)
    assert result.project_path == tmp_project
    assert result.scan_time_ms >= 0.0
    assert result.has_tests is False


def test_scan_project_detects_tests(tmp_project_with_tests: Path):
    mock_result = MagicMock()
    mock_result.stdout = "[]"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        result = scan_project(tmp_project_with_tests)

    assert result.has_tests is True


def test_scan_project_tools_used_list(tmp_project: Path):
    """When all tools succeed, they all appear in tools_used."""
    mocks = [make_subprocess_result_mock() for _ in range(5)]  # ruff, vulture, semgrep, pip-audit, pyright

    with patch("subprocess.run", side_effect=mocks):
        result = scan_project(tmp_project)

    assert "ruff" in result.tools_used
    assert "vulture" in result.tools_used
