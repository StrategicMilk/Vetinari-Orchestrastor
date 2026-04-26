"""Evaluation tests for builder agent output quality.

These tests use the scoring protocol in ``vetinari.evaluation`` to verify
that ``evaluate_code_quality`` correctly scores syntactically valid code,
code with syntax errors, and incomplete stub implementations.
"""

from __future__ import annotations

import pytest

from vetinari.evaluation import EvalResult, combine_results, evaluate_code_quality
from vetinari.exceptions import ExecutionError


@pytest.mark.eval
class TestBuilderQuality:
    """Evaluation tests for generated Python code quality scoring."""

    def test_valid_code_scores_high(self, mock_code_response: str) -> None:
        """Syntactically valid Python with imports and functions scores above 0.7."""
        result: EvalResult = evaluate_code_quality(mock_code_response)

        assert result.score > 0.7, (
            f"Expected score > 0.7 for valid Python code, got {result.score:.2f}. Reasoning: {result.reasoning}"
        )
        assert result.passed is True
        assert result.metrics["syntax_score"] == 1.0

    def test_syntax_error_scores_low(self) -> None:
        """Code containing a syntax error scores below 0.3 and fails evaluation."""
        broken_code = """
def broken_function(x
    return x + 1
"""
        result: EvalResult = evaluate_code_quality(broken_code)

        assert result.score < 0.3, (
            f"Expected score < 0.3 for code with syntax errors, got {result.score:.2f}. Reasoning: {result.reasoning}"
        )
        assert result.passed is False
        assert result.metrics["syntax_score"] == 0.0

    def test_incomplete_function_detected(self) -> None:
        """A function with a pass-only body incurs a completeness penalty."""
        stub_code = """
import logging

logger = logging.getLogger(__name__)


def process_data(data: dict) -> dict:
    \"\"\"Process the provided data.\"\"\"
    pass
"""
        result: EvalResult = evaluate_code_quality(stub_code)

        # Syntax is valid so syntax_score must be 1.0
        assert result.metrics["syntax_score"] == 1.0
        # Function completeness must be penalised
        assert result.metrics["function_score"] < 1.0, (
            f"Expected function_score < 1.0 for stub function, got {result.metrics['function_score']}"
        )
        func_issues = result.metrics.get("function_issues", [])
        assert any("pass" in issue or "stub" in issue for issue in func_issues), (
            f"Expected stub/pass mention in function_issues, got: {func_issues}"
        )

    def test_empty_string_scores_full(self) -> None:
        """An empty code string parses and scores 1.0 — no violations detected."""
        result: EvalResult = evaluate_code_quality("")

        # Empty string parses without error
        assert result.metrics["syntax_score"] == 1.0
        # All three dimensions score 1.0: syntax OK, no imports to fail, no stubs
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed is True

    def test_wildcard_import_penalised(self) -> None:
        """Code that uses wildcard imports receives an import dimension penalty."""
        wildcard_code = """
from os.path import *
from typing import *


def get_path(name: str) -> str:
    return join("/tmp", name)
"""
        result: EvalResult = evaluate_code_quality(wildcard_code)

        assert result.metrics["import_score"] < 1.0, (
            f"Expected import_score < 1.0 for wildcard imports, got {result.metrics['import_score']}"
        )
        import_issues = result.metrics.get("import_issues", [])
        assert any("wildcard" in issue for issue in import_issues), (
            f"Expected 'wildcard' in import_issues, got: {import_issues}"
        )

    def test_combine_results_aggregates_correctly(self) -> None:
        """combine_results returns the mean score and logical-AND of passed flags."""
        r1 = EvalResult(score=0.9, passed=True, reasoning="good")
        r2 = EvalResult(score=0.6, passed=True, reasoning="acceptable")
        r3 = EvalResult(score=0.4, passed=False, reasoning="needs work")

        combined = combine_results([r1, r2, r3])

        expected_mean = (0.9 + 0.6 + 0.4) / 3
        assert combined.score == pytest.approx(expected_mean, abs=1e-9)
        # All must pass for combined.passed to be True; r3 failed so it's False
        assert combined.passed is False
        assert combined.metrics["result_count"] == 3
        assert combined.metrics["individual_scores"] == [0.9, 0.6, 0.4]

    def test_combine_results_all_pass(self) -> None:
        """combine_results returns passed=True when all individual results pass."""
        r1 = EvalResult(score=0.8, passed=True, reasoning="good")
        r2 = EvalResult(score=0.75, passed=True, reasoning="good")

        combined = combine_results([r1, r2])

        assert combined.passed is True
        assert combined.score == pytest.approx(0.775, abs=1e-9)

    def test_combine_results_empty_raises(self) -> None:
        """combine_results raises ExecutionError when given an empty list."""
        with pytest.raises(ExecutionError, match="at least one"):
            combine_results([])
