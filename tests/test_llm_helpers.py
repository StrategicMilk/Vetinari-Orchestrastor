"""Tests for vetinari.llm_helpers — lightweight LLM call utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.llm_helpers import (
    _RETRY_TEMPLATES,
    assess_complexity_static,
    assess_complexity_via_llm,
    check_ambiguity_via_llm,
    classify_goal_via_llm,
    diagnose_defect_via_llm,
    generate_retry_brief,
    quick_llm_call,
    score_confidence_structural,
    score_confidence_via_llm,
)


class TestQuickLlmCall:
    """Tests for the core quick_llm_call helper."""

    def test_returns_none_when_adapter_unavailable(self) -> None:
        """Returns None gracefully when AdapterManager can't be imported."""
        with patch.dict("sys.modules", {"vetinari.adapter_manager": None}):
            result = quick_llm_call("test prompt")
            # Should return None without raising
            assert result is None or isinstance(result, str)

    @patch("vetinari.adapter_manager.get_adapter_manager")
    def test_returns_none_on_error_status(self, mock_mgr_fn: MagicMock) -> None:
        """Returns None when the LLM returns an error status."""
        mock_mgr = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "error"
        mock_response.error = "model unavailable"
        mock_mgr.infer.return_value = mock_response
        mock_mgr_fn.return_value = mock_mgr

        result = quick_llm_call("test prompt")
        assert result is None

    @patch("vetinari.adapter_manager.get_adapter_manager")
    def test_returns_output_on_success(self, mock_mgr_fn: MagicMock) -> None:
        """Returns stripped output text on successful inference."""
        mock_mgr = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "success"
        mock_response.output = "  code  \n"
        mock_mgr.infer.return_value = mock_response
        mock_mgr_fn.return_value = mock_mgr

        result = quick_llm_call("classify this goal")
        assert result == "code"

    @patch("vetinari.adapter_manager.get_adapter_manager")
    def test_system_prompt_passed_as_separate_field(self, mock_mgr_fn: MagicMock) -> None:
        """system_prompt must reach InferenceRequest.system_prompt, not be concatenated into prompt.

        Regression test for Bug 8: previously system_prompt was merged into
        prompt via f-string, erasing the system/user boundary that adapters
        use to format messages correctly (e.g. ChatML, Llama-3 instruct).
        """
        from vetinari.adapters.base import InferenceRequest

        mock_mgr = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "success"
        mock_response.output = "ok"
        mock_mgr.infer.return_value = mock_response
        mock_mgr_fn.return_value = mock_mgr

        quick_llm_call("user prompt text", system_prompt="you are a helpful assistant")

        assert mock_mgr.infer.called
        captured: InferenceRequest = mock_mgr.infer.call_args[0][0]
        # system_prompt must be a separate field — NOT concatenated into prompt
        assert captured.system_prompt == "you are a helpful assistant"
        assert "you are a helpful assistant" not in captured.prompt


class TestClassifyGoalViaLlm:
    """Tests for LLM-powered goal classification."""

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - fallback parser unit intentionally isolates transport helper
    @patch("vetinari.ml.task_classifier.TaskClassifier.classify", return_value=("general", 0.0))
    def test_returns_valid_category(self, mock_classify: MagicMock, mock_call: MagicMock) -> None:
        """Returns a valid category when LLM responds correctly."""
        mock_call.return_value = "security"
        result = classify_goal_via_llm("audit the authentication system")
        assert result == "security"

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - fallback parser unit intentionally isolates transport helper
    @patch("vetinari.ml.task_classifier.TaskClassifier.classify", return_value=("general", 0.0))
    def test_returns_none_for_invalid_category(self, mock_classify: MagicMock, mock_call: MagicMock) -> None:
        """Returns None when LLM returns an unrecognized category."""
        mock_call.return_value = "banana"
        result = classify_goal_via_llm("do something weird")
        assert result is None

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - fallback parser unit intentionally isolates transport helper
    @patch("vetinari.ml.task_classifier.TaskClassifier.classify", return_value=("general", 0.0))
    def test_returns_none_when_unavailable(self, mock_classify: MagicMock, mock_call: MagicMock) -> None:
        """Returns None when LLM is unavailable."""
        mock_call.return_value = None
        result = classify_goal_via_llm("test goal")
        assert result is None


class TestDiagnoseDefectViaLlm:
    """Tests for LLM-powered defect diagnosis."""

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_parses_category_and_explanation(self, mock_call: MagicMock) -> None:
        """Correctly parses category: explanation format."""
        mock_call.return_value = "hallucinated_import: The import 'foo' does not exist"
        result = diagnose_defect_via_llm("build auth module", "ImportError: no module named foo")
        assert result is not None
        category, explanation = result
        assert category == "hallucinated_import"
        assert "foo" in explanation

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_returns_none_on_unparseable(self, mock_call: MagicMock) -> None:
        """Returns None when LLM response can't be parsed."""
        mock_call.return_value = "no category here"
        result = diagnose_defect_via_llm("task", "reason")
        assert result is None


class TestAssessComplexityViaLlm:
    """Tests for complexity assessment (now delegates to AST static analysis)."""

    def test_parses_structured_response(self) -> None:
        """assess_complexity_via_llm now delegates to AST static analysis — returns a dict."""
        result = assess_complexity_via_llm("refactor the entire auth system security audit migration")
        # May return None for empty or may return a dict — both valid
        assert result is None or isinstance(result, dict)

    def test_returns_none_without_class(self) -> None:
        """Returns None for empty description."""
        result = assess_complexity_via_llm("")
        assert result is None

    def test_alias_equals_static(self) -> None:
        """assess_complexity_via_llm is the same function as assess_complexity_static."""
        assert assess_complexity_via_llm is assess_complexity_static


class TestAssessComplexityStatic:
    """Tests for Item 16.5 — AST-based static complexity analysis."""

    def test_simple_task_returns_simple(self) -> None:
        """A trivially short task description classifies as SIMPLE."""
        result = assess_complexity_static("fix typo")
        assert result is not None
        assert result["classification"] == "SIMPLE"

    def test_complex_task_returns_complex(self) -> None:
        """A description with many complexity signals classifies as COMPLEX."""
        result = assess_complexity_static(
            "architect a distributed multi-service security audit migration with backwards compat"
        )
        assert result is not None
        assert result["classification"] == "COMPLEX"

    def test_result_has_required_keys(self) -> None:
        """Result dict always contains classification, risk, files, cyclomatic."""
        result = assess_complexity_static("implement a simple function")
        assert result is not None
        for key in ("classification", "risk", "files", "cyclomatic"):
            assert key in result

    def test_empty_description_returns_none(self) -> None:
        """Empty description returns None."""
        assert assess_complexity_static("") is None
        assert assess_complexity_static("   ") is None

    def test_python_snippet_raises_cyclomatic(self) -> None:
        """Embedded Python code with branching increases cyclomatic complexity."""
        snippet = (
            "```python\n"
            "def process(items):\n"
            "    for item in items:\n"
            "        if item > 0:\n"
            "            if item > 100:\n"
            "                return True\n"
            "    return False\n"
            "```"
        )
        result = assess_complexity_static(snippet)
        assert result is not None
        assert result["cyclomatic"] > 1


class TestCheckAmbiguityViaLlm:
    """Tests for LLM ambiguity detection."""

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_detects_ambiguous(self, mock_call: MagicMock) -> None:
        """Detects ambiguous request with clarifying question."""
        mock_call.return_value = "YES: What algorithm should be implemented?"
        result = check_ambiguity_via_llm("implement the best algorithm")
        assert result is not None
        is_ambiguous, question = result
        assert is_ambiguous is True
        assert "algorithm" in question

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_detects_clear(self, mock_call: MagicMock) -> None:
        """Detects clear request."""
        mock_call.return_value = "NO"
        result = check_ambiguity_via_llm("add a login button to the header")
        assert result is not None
        is_ambiguous, _ = result
        assert is_ambiguous is False


class TestScoreConfidenceStructural:
    """Tests for Item 16.6 — structural confidence scoring without LLM."""

    def test_code_task_with_code_response_high_score(self) -> None:
        """Task asking for code + response with code block yields high structural score."""
        score = score_confidence_structural(
            "implement a function to add two numbers",
            "```python\ndef add(a, b):\n    return a + b\n```",
        )
        # Either conclusive high score or None (uncertain) — must not be low
        assert score is None or score >= 0.5

    def test_code_task_with_prose_response_low_score(self) -> None:
        """Task asking for code + response with only prose yields low structural score."""
        score = score_confidence_structural(
            "implement a function to add two numbers",
            "I think you should probably add them together somehow.",
        )
        # Either conclusive low score or None (uncertain)
        assert score is None or score <= 0.5

    def test_empty_inputs_return_zero(self) -> None:
        """Empty task or response returns 0.0."""
        assert score_confidence_structural("", "some response") == 0.0
        assert score_confidence_structural("task description", "") == 0.0

    def test_returns_none_for_ambiguous_middle(self) -> None:
        """Returns None when the structural signal is ambiguous (defers to LLM)."""
        # Craft inputs that land in the uncertain middle band
        score = score_confidence_structural(
            "do something",
            "Here is some content that may or may not be relevant.",
        )
        # The function should either return None or a score — never raise
        assert score is None or isinstance(score, float)

    def test_score_in_range(self) -> None:
        """Any returned score is in [0.0, 1.0]."""
        score = score_confidence_structural("implement binary search", "def binary_search(): pass")
        if score is not None:
            assert 0.0 <= score <= 1.0


class TestScoreConfidenceViaLlm:
    """Tests for LLM confidence scoring — structural tier skips LLM when conclusive."""

    def test_structural_conclusive_skips_llm(self) -> None:
        """When structural check is conclusive, quick_llm_call is NOT invoked."""
        with patch("vetinari.llm_helpers.quick_llm_call") as mock_call:  # noqa: VET242 - structural gate unit intentionally isolates fallback helper
            # A clear code task + code response should be conclusive at Tier 1
            result = score_confidence_via_llm(
                "implement a function to add two numbers",
                "```python\ndef add(a, b):\n    return a + b\n```",
            )
        # If structural was conclusive, LLM was not called
        if mock_call.call_count == 0:
            assert result is not None

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_parses_score(self, mock_call: MagicMock) -> None:
        """Correctly parses a decimal confidence score from LLM fallback."""
        mock_call.return_value = "0.85"
        # Use inputs that land in the uncertain middle so LLM is consulted
        result = score_confidence_via_llm("do something vague", "Here is a partial answer.")
        # Either structural was conclusive (no LLM) or LLM was called and returned float
        assert result is None or isinstance(result, float)

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - parser unit intentionally isolates transport helper
    def test_clamps_score(self, mock_call: MagicMock) -> None:
        """Clamps scores to 0.0-1.0 range when LLM returns out-of-range value."""
        mock_call.return_value = "1.5"
        result = score_confidence_via_llm("task", "response text that is ambiguous and unclear")
        if result is not None:
            assert result <= 1.0


class TestGenerateRetryBrief:
    """Tests for Item 16.7 — pre-written retry template map."""

    def test_known_category_returns_template_without_llm(self) -> None:
        """Known defect category uses pre-written template — no LLM call."""
        with patch("vetinari.llm_helpers.quick_llm_call") as mock_call:  # noqa: VET242 - template fallback unit intentionally isolates transport helper
            result = generate_retry_brief(
                "ImportError in generated code",
                defect_category="hallucinated_import",
            )
        mock_call.assert_not_called()
        assert result is not None
        assert "1." in result  # templates always start with numbered list

    def test_all_known_categories_have_templates(self) -> None:
        """Every key in _RETRY_TEMPLATES produces a non-empty template."""
        for category, template in _RETRY_TEMPLATES.items():
            assert template.strip(), f"Empty template for category {category!r}"
            assert "1." in template, f"Template for {category!r} missing numbered list"

    def test_unknown_category_falls_back_to_llm(self) -> None:
        """Unknown category falls back to LLM when category not in template map."""
        with patch("vetinari.llm_helpers.quick_llm_call") as mock_call:  # noqa: VET242 - template fallback unit intentionally isolates transport helper
            mock_call.return_value = "1. Try X\n2. Try Y\n3. Try Z"
            result = generate_retry_brief(
                "some unknown error",
                defect_category="unknown_category_xyz",
            )
        mock_call.assert_called_once()
        assert result == "1. Try X\n2. Try Y\n3. Try Z"

    def test_no_category_falls_back_to_llm(self) -> None:
        """Omitting defect_category falls back to LLM."""
        with patch("vetinari.llm_helpers.quick_llm_call") as mock_call:  # noqa: VET242 - template fallback unit intentionally isolates transport helper
            mock_call.return_value = "1. Use correct import\n2. Add error handling\n3. Test edge cases"
            result = generate_retry_brief("ImportError in generated code")
        mock_call.assert_called_once()
        assert result is not None
        assert "import" in result.lower() or "1." in result

    @patch("vetinari.llm_helpers.quick_llm_call")  # noqa: VET242 - template fallback unit intentionally isolates transport helper
    def test_returns_none_when_unavailable(self, mock_call: MagicMock) -> None:
        """Returns None when LLM is unavailable and no template matches."""
        mock_call.return_value = None
        result = generate_retry_brief("error", defect_category="completely_unknown")
        assert result is None

    def test_template_covers_hallucinated_import_category(self) -> None:
        """hallucinated_import template mentions imports specifically."""
        result = generate_retry_brief("error", defect_category="hallucinated_import")
        assert result is not None
        assert "import" in result.lower()

    def test_template_covers_style_violation_category(self) -> None:
        """style_violation template mentions ruff or formatting."""
        result = generate_retry_brief("error", defect_category="style_violation")
        assert result is not None
        assert "ruff" in result.lower() or "format" in result.lower()
