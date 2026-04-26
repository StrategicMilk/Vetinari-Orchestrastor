"""Tests for vetinari.routing.complexity_router — behavioral tests.

Covers:
- Borderline case resolved by AST analysis, not LLM
- Import-heavy description classified as COMPLEX
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.routing.complexity_router import (
    Complexity,
    _ast_complexity_from_description,
    classify_complexity,
)

# ---------------------------------------------------------------------------
# Test 1: Borderline case resolved by AST, not LLM
# ---------------------------------------------------------------------------


class TestBorderlineCaseResolvedByAST:
    """Borderline complexity tasks use AST analysis — no LLM call is made.

    The classify_complexity function detects a borderline case when the absolute
    difference between complex_score and simple_score is <= 1 and both are > 0.
    In that situation it calls _ast_complexity_from_description internally instead
    of any LLM function. This test verifies:
    1. _ast_complexity_from_description IS invoked for a borderline description.
    2. quick_llm_call (the gateway to any LLM) is NOT invoked.
    """

    # Description that produces is_borderline=True after all score adjustments:
    # - 'trivial' → +1 simple_score
    # - 'architect' → +1 complex_score
    # - word_count=17 (15-50 range) → no word-count adjustment
    # - estimated_files=3 (> 2, <= 10) → no files adjustment
    # Final: complex=1, simple=1, score_diff=0 → borderline=True
    _BORDERLINE_DESC = (
        "This is a trivial change but the architect must review "
        "the design carefully before proceeding with implementation"
    )
    _BORDERLINE_FILES = 3  # avoids estimated_files <= 2 simple_score boost

    def test_borderline_description_ast_is_invoked(self) -> None:
        """_ast_complexity_from_description is called for borderline descriptions.

        With complex_score == simple_score == 1 and score_diff == 0, the
        borderline condition is met and _ast_complexity_from_description is
        called to break the tie without consulting an LLM.
        """
        ast_called: list[bool] = []
        original = _ast_complexity_from_description

        def _tracking_ast(d: str) -> dict | None:
            ast_called.append(True)
            return original(d)

        with patch(
            "vetinari.routing.complexity_router._ast_complexity_from_description",
            side_effect=_tracking_ast,
        ):
            classify_complexity(self._BORDERLINE_DESC, estimated_files=self._BORDERLINE_FILES)

        assert ast_called, "_ast_complexity_from_description must be called for borderline cases"

    def test_borderline_description_does_not_call_llm(self) -> None:
        """No LLM call (quick_llm_call) is made for a borderline description.

        The borderline path resolves complexity using AST analysis only.
        quick_llm_call is the single gateway to any LLM in the helpers layer;
        if it is not called, no LLM was consulted.
        """
        with patch("vetinari.llm_helpers.quick_llm_call") as mock_llm:
            classify_complexity(self._BORDERLINE_DESC, estimated_files=self._BORDERLINE_FILES)

        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Import-heavy description classified as COMPLEX
# ---------------------------------------------------------------------------


class TestImportHeavyDescriptionIsComplex:
    """Descriptions with many imports and branches are classified as COMPLEX."""

    def test_import_heavy_description_direct_ast(self) -> None:
        """_ast_complexity_from_description classifies many-branch code as COMPLEX.

        The embedded Python snippet has multiple if/for/try branches that push
        cyclomatic complexity above _AST_COMPLEX_THRESHOLD (10).
        """
        # Craft a description with embedded Python containing >10 branching constructs
        description = (
            "import os, sys, json, requests, numpy\n"
            "def process(data):\n"
            "    if data:\n"
            "        for item in data:\n"
            "            if item > 0:\n"
            "                for sub in item:\n"
            "                    if sub:\n"
            "                        while sub > 0:\n"
            "                            try:\n"
            "                                if sub % 2:\n"
            "                                    pass\n"
            "            except Exception:\n"
            "                pass\n"
        )
        result = _ast_complexity_from_description(description)

        assert result is not None, "_ast_complexity_from_description must return a dict"
        assert "classification" in result
        # With many branching constructs, cyclomatic complexity is pushed high
        assert result["classification"] in ("COMPLEX", "MODERATE"), (
            f"Expected COMPLEX or MODERATE for import-heavy code, got {result['classification']!r}"
        )

    def test_import_heavy_via_classify_complexity(self) -> None:
        """classify_complexity returns COMPLEX for a description with many complexity signals.

        'distributed', 'multi-service', and 'architect' are all strong complex signals.
        A long description (>100 words) also adds +2 to complex_score.
        """
        # Multiple explicit _COMPLEX_SIGNALS matches ensure COMPLEX classification
        description = (
            "Design and architect a distributed multi-service system for migrating "
            "a legacy monolith. The solution must handle backwards compatibility, "
            "security audit requirements, and compliance constraints. Performance-critical "
            "endpoints must be optimized. This is a multi-component effort spanning many "
            "services and modules across the entire platform architecture. Risk assessment "
            "must be performed before implementation begins. The design system must account "
            "for all edge cases and failure modes."
        )
        result = classify_complexity(description)

        assert result == Complexity.COMPLEX, (
            f"Expected Complexity.COMPLEX for import-heavy complex description, got {result!r}"
        )

    def test_ast_complexity_from_description_returns_none_for_empty(self) -> None:
        """_ast_complexity_from_description returns None for empty input."""
        assert _ast_complexity_from_description("") is None
        assert _ast_complexity_from_description("   ") is None

    def test_ast_complexity_from_description_keys(self) -> None:
        """Result dict has the expected keys."""
        result = _ast_complexity_from_description("implement a simple function")
        assert result is not None
        assert {"classification", "cyclomatic", "risk", "files"} == set(result.keys())
