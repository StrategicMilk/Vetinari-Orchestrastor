"""Tests for vetinari.errors — error remediation mapping."""

from __future__ import annotations

import pytest

from vetinari.errors import (
    ERROR_REMEDIATIONS,
    ErrorRemediation,
    find_remediation,
    format_remediation,
)


class TestFindRemediation:
    """Tests for error pattern matching."""

    def test_matches_unreachable(self) -> None:
        """Matches 'UNREACHABLE' error to local inference remediation."""
        result = find_remediation("Local inference: UNREACHABLE")
        assert result is not None
        assert result.title == "Local inference unreachable"
        assert len(result.steps) >= 3

    def test_matches_model_not_found(self) -> None:
        """Matches model not found errors."""
        result = find_remediation("model not found: mistral-7b")
        assert result is not None
        assert "not found" in result.title.lower()

    def test_matches_circuit_breaker(self) -> None:
        """Matches circuit breaker open errors."""
        result = find_remediation("Circuit breaker OPEN for model xyz")
        assert result is not None
        assert "circuit" in result.title.lower()

    def test_matches_cuda_oom(self) -> None:
        """Matches CUDA out of memory errors."""
        result = find_remediation("CUDA error: out of memory")
        assert result is not None
        assert result.severity == "critical"

    def test_matches_port_conflict(self) -> None:
        """Matches port already in use errors."""
        result = find_remediation("Address already in use: port 5000")
        assert result is not None
        assert "port" in result.title.lower()

    def test_returns_none_for_unknown(self) -> None:
        """Returns None for unrecognized errors."""
        result = find_remediation("something completely unrelated")
        assert result is None

    def test_case_insensitive(self) -> None:
        """Pattern matching is case insensitive and returns the matching remediation."""
        result = find_remediation("local INFERENCE: unreachable")
        assert result is not None
        assert len(result.title) > 0


class TestFormatRemediation:
    """Tests for remediation formatting."""

    def test_format_includes_title(self) -> None:
        """Formatted output includes the title."""
        remediation = ERROR_REMEDIATIONS[0]
        output = format_remediation(remediation)
        assert remediation.title in output

    def test_format_includes_steps(self) -> None:
        """Formatted output includes numbered steps."""
        remediation = ERROR_REMEDIATIONS[0]
        output = format_remediation(remediation)
        assert "1." in output
        assert "2." in output

    def test_format_includes_severity(self) -> None:
        """Formatted output includes severity level."""
        remediation = ERROR_REMEDIATIONS[0]
        output = format_remediation(remediation)
        assert remediation.severity.upper() in output


class TestErrorRemediationRegistry:
    """Tests for the remediation registry integrity."""

    def test_all_patterns_compile(self) -> None:
        """All regex patterns compile without error."""
        import re

        compiled = [re.compile(r.pattern) for r in ERROR_REMEDIATIONS]
        assert len(compiled) == len(ERROR_REMEDIATIONS)

    def test_minimum_remediations(self) -> None:
        """Registry has at least 10 remediations."""
        assert len(ERROR_REMEDIATIONS) >= 10

    def test_all_have_steps(self) -> None:
        """Every remediation has at least one step."""
        for r in ERROR_REMEDIATIONS:
            assert len(r.steps) >= 1, f"Remediation '{r.title}' has no steps"
