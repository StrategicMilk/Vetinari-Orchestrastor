"""Tests for vetinari.llm_capabilities — advanced LLM-powered analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.llm_capabilities import (
    diagnose_error,
    explain_complex_function,
    generate_changelog,
)


class TestDiagnoseError:
    """Tests for LLM-powered error diagnosis."""

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_parses_structured_response(self, mock_call: MagicMock) -> None:
        """Parses DIAGNOSIS/LIKELY_CAUSE/SUGGESTED_FIX format."""
        mock_call.return_value = (
            "DIAGNOSIS: Missing import for datetime module\n"
            "LIKELY_CAUSE: Line 15 uses datetime.now() without import\n"
            "SUGGESTED_FIX: Add 'from datetime import datetime' at top"
        )
        result = diagnose_error("process_date", "NameError: name 'datetime' is not defined")
        assert result is not None
        assert "datetime" in result["diagnosis"].lower()
        assert "likely_cause" in result
        assert "suggested_fix" in result

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_handles_unstructured_response(self, mock_call: MagicMock) -> None:
        """Returns raw response when format doesn't match."""
        mock_call.return_value = "The function fails because of a typo in the variable name."
        result = diagnose_error("calc_total", "NameError: 'totl'")
        assert result is not None
        assert "diagnosis" in result

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_returns_none_when_unavailable(self, mock_call: MagicMock) -> None:
        """Returns None when LLM is unavailable."""
        mock_call.return_value = None
        result = diagnose_error("func", "error")
        assert result is None


class TestGenerateChangelog:
    """Tests for changelog generation."""

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_generates_changelog(self, mock_call: MagicMock) -> None:
        """Returns formatted changelog text."""
        mock_call.return_value = "## Added\n- New user authentication system\n\n## Fixed\n- Login timeout issue"
        commits = [
            {"hash": "abc1234", "message": "feat: add JWT auth"},
            {"hash": "def5678", "message": "fix: login timeout"},
        ]
        result = generate_changelog(commits, version="1.2.0")
        assert result is not None
        assert "Added" in result or "Fixed" in result

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_returns_none_when_unavailable(self, mock_call: MagicMock) -> None:
        """Returns None when LLM is unavailable."""
        mock_call.return_value = None
        result = generate_changelog([{"message": "commit"}])
        assert result is None


class TestExplainComplexFunction:
    """Tests for 'why' comment generation."""

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_generates_why_comment(self, mock_call: MagicMock) -> None:
        """Returns a 'Why:' comment for complex functions."""
        mock_call.return_value = "# Why: This uses a two-pass approach because..."
        result = explain_complex_function(
            "merge_sort_with_insertion",
            "def merge_sort_with_insertion(arr):\n    ...",
            cyclomatic_complexity=8,
        )
        assert result is not None
        assert "Why" in result or "because" in result.lower()

    @patch("vetinari.llm_helpers.quick_llm_call")
    def test_returns_none_when_unavailable(self, mock_call: MagicMock) -> None:
        """Returns None when LLM is unavailable."""
        mock_call.return_value = None
        result = explain_complex_function("func", "def func(): pass")
        assert result is None
