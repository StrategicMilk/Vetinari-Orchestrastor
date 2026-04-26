"""Tests for vetinari.ml.task_classifier — TaskClassifier behavioral tests.

Covers:
- High-confidence ML classification (sklearn path)
- Low-confidence LLM fallback routing (via classify_goal_via_llm)
- Cold-start keyword fallback when sklearn is unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.ml.task_classifier import CONFIDENCE_THRESHOLD, MIN_TRAINING_EXAMPLES, TaskClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled_examples(n: int = MIN_TRAINING_EXAMPLES) -> list[tuple[str, str]]:
    """Generate ``n`` labeled training examples spread across two categories.

    Returns a list of (text, label) tuples with enough samples to exceed
    MIN_TRAINING_EXAMPLES so the trained model path is activated.

    Args:
        n: Total number of examples to generate.

    Returns:
        List of (text, label) tuples.
    """
    examples: list[tuple[str, str]] = []
    for i in range(n):
        if i % 2 == 0:
            examples.append((f"implement a function to compute value number {i}", "code"))
        else:
            examples.append((f"research the best library for data processing task {i}", "research"))
    return examples


# ---------------------------------------------------------------------------
# Test 1: High-confidence classification (sklearn path)
# ---------------------------------------------------------------------------


class TestHighConfidenceClassification:
    """TaskClassifier returns high-confidence result when the trained model is confident."""

    def test_trained_model_returns_high_confidence(self) -> None:
        """Training with MIN_TRAINING_EXAMPLES examples produces confidence >= CONFIDENCE_THRESHOLD.

        The test trains the classifier with balanced code/research examples, then
        classifies a strongly code-flavored description. The ML model must return
        the correct category with confidence >= 0.6 (CONFIDENCE_THRESHOLD).
        """
        clf = TaskClassifier()
        for text, label in _make_labeled_examples(MIN_TRAINING_EXAMPLES):
            clf.add_example(text, label)

        # A clearly code-flavored description — trained model should be confident
        category, confidence = clf.classify("implement a function and build a class module")

        assert category != "unknown", "Category must not be 'unknown' when model is confident"
        assert confidence >= CONFIDENCE_THRESHOLD, (
            f"Expected confidence >= {CONFIDENCE_THRESHOLD}, got {confidence!r} for category {category!r}"
        )
        assert category == "code", f"Expected 'code' category, got {category!r}"


# ---------------------------------------------------------------------------
# Test 2: Low-confidence LLM fallback via classify_goal_via_llm
# ---------------------------------------------------------------------------


class TestLowConfidenceLLMFallback:
    """When TaskClassifier returns low confidence, classify_goal_via_llm invokes the LLM."""

    def test_llm_fallback_called_when_classifier_confidence_is_low(self) -> None:
        """classify_goal_via_llm calls quick_llm_call when TaskClassifier confidence is low.

        The test patches TaskClassifier.classify to return a confidence below
        CONFIDENCE_THRESHOLD, then verifies that quick_llm_call is invoked as the
        LLM fallback.
        """
        # Import here so patching targets are resolved at test time
        from vetinari.llm_helpers import classify_goal_via_llm

        low_confidence_result = ("general", 0.2)  # below CONFIDENCE_THRESHOLD of 0.6

        with patch(
            "vetinari.ml.task_classifier.TaskClassifier.classify",
            return_value=low_confidence_result,
        ):
            with patch(
                "vetinari.llm_helpers.quick_llm_call",
                return_value="code",
            ) as mock_llm:
                result = classify_goal_via_llm("implement a thing")

        mock_llm.assert_called_once()
        assert result == "code"

    def test_llm_fallback_not_called_when_confidence_is_high(self) -> None:
        """classify_goal_via_llm skips LLM when TaskClassifier is confident.

        Ensures the fast path (no LLM) is taken when confidence >= CONFIDENCE_THRESHOLD.
        """
        from vetinari.llm_helpers import classify_goal_via_llm

        high_confidence_result = ("code", 0.9)  # above CONFIDENCE_THRESHOLD

        with patch(
            "vetinari.ml.task_classifier.TaskClassifier.classify",
            return_value=high_confidence_result,
        ):
            with patch(
                "vetinari.llm_helpers.quick_llm_call",
            ) as mock_llm:
                result = classify_goal_via_llm("implement a class")

        mock_llm.assert_not_called()
        assert result == "code"


# ---------------------------------------------------------------------------
# Test 3: Cold-start keyword fallback when sklearn is unavailable
# ---------------------------------------------------------------------------


class TestColdStartKeywordFallback:
    """TaskClassifier falls back to keyword matching when sklearn is not installed."""

    def test_classify_uses_keyword_fallback_when_sklearn_unavailable(self) -> None:
        """When sklearn raises ImportError, classify returns a valid (category, confidence) tuple.

        The test mocks the sklearn import to be unavailable, instantiates a fresh
        TaskClassifier, and calls classify with a code-flavored description. The
        keyword-based GoalClassifier must return a valid result.
        """
        # Patch builtins.__import__ to raise ImportError for sklearn
        import builtins

        original_import = builtins.__import__

        def _import_blocker(name: str, *args: object, **kwargs: object) -> object:
            if name == "sklearn" or name.startswith("sklearn."):
                raise ImportError(f"Mocked ImportError: {name} is not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_blocker):
            clf = TaskClassifier()
            # Force the probe to run inside the patch context
            clf._sklearn_available = None  # reset cached probe result
            category, confidence = clf.classify("implement a function")

        # The keyword fallback must return a valid (str, float) tuple
        assert isinstance(category, str), f"category must be str, got {type(category)}"
        assert isinstance(confidence, float), f"confidence must be float, got {type(confidence)}"
        assert category != "", "category must not be empty string"
        assert 0.0 <= confidence <= 1.0, f"confidence must be in [0, 1], got {confidence}"

    def test_classify_keyword_fallback_returns_code_for_implement(self) -> None:
        """Keyword fallback routes 'implement a function' to the code category."""
        clf = TaskClassifier()
        # Force sklearn unavailability by setting the flag directly
        clf._sklearn_available = False

        category, confidence = clf.classify("implement a function")

        assert category == "code", f"Expected 'code' from keyword fallback, got {category!r}"
        assert confidence > 0.0, "Confidence must be positive for a keyword match"
