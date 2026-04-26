"""Tests for PipelineHelpersMixin._analyze_input().

Regression coverage for bug: _analyze_input() never populated 'goal_type',
causing pipeline_engine.py to always build RequestSpec with category='general'.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestAnalyzeInputGoalType:
    """_analyze_input() must populate 'goal_type' from the classifier category."""

    def _make_mixin(self):
        """Return a minimal PipelineHelpersMixin instance for testing."""
        from vetinari.orchestration.pipeline_helpers import PipelineHelpersMixin

        class _Harness(PipelineHelpersMixin):
            pass

        return _Harness()

    def test_goal_type_populated_from_classifier_category(self):
        """When classifier returns 'code', goal_type must be 'code', not 'general'."""
        mixin = self._make_mixin()
        classifier_response = {
            "category": "code",
            "confidence": 0.9,
            "complexity": "standard",
            "cross_cutting": [],
            "matched_keywords": ["implement"],
            "source": "keyword",
        }
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            return_value=classifier_response,
        ):
            result = mixin._analyze_input("implement a sorting algorithm", {})

        assert result["goal_type"] == "code", (
            f"Expected 'code' but got {result['goal_type']!r} — "
            "pipeline will always build RequestSpec(category='general') without this fix"
        )

    def test_goal_type_populated_for_research_category(self):
        """When classifier returns 'research', goal_type must be 'research'."""
        mixin = self._make_mixin()
        classifier_response = {
            "category": "research",
            "confidence": 0.8,
            "complexity": "standard",
            "cross_cutting": [],
            "matched_keywords": ["analyze"],
            "source": "keyword",
        }
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            return_value=classifier_response,
        ):
            result = mixin._analyze_input("analyze this dataset thoroughly", {})

        assert result["goal_type"] == "research"

    def test_goal_type_fallback_path_code_keywords(self):
        """When classifier is unavailable, keyword fallback must still set goal_type."""
        mixin = self._make_mixin()
        # Patch the import inside the function so the except branch fires
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            side_effect=RuntimeError("classifier unavailable"),
        ):
            result = mixin._analyze_input("implement a REST API endpoint", {})

        assert result["goal_type"] == "code", (
            f"Expected 'code' from keyword fallback, got {result['goal_type']!r}"
        )

    def test_goal_type_fallback_path_defaults_to_general(self):
        """When classifier is unavailable and no keywords match, goal_type is 'general'."""
        mixin = self._make_mixin()
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            side_effect=RuntimeError("classifier unavailable"),
        ):
            result = mixin._analyze_input("hello world", {})

        assert result["goal_type"] == "general"

    def test_goal_type_present_in_result_keys(self):
        """'goal_type' must be a key in the returned dict (not just via default path)."""
        mixin = self._make_mixin()
        classifier_response = {
            "category": "security",
            "confidence": 0.75,
            "complexity": "standard",
            "cross_cutting": [],
            "matched_keywords": ["auth"],
            "source": "llm",
        }
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            return_value=classifier_response,
        ):
            result = mixin._analyze_input("add auth token verification", {})

        assert "goal_type" in result
        assert result["goal_type"] == "security"


class TestRetrieveMemoryForPlanningEntryTypes:
    """_retrieve_memory_for_planning() must pass lowercase entry_types to store.search().

    Regression: the filter previously used uppercase strings ("DECISION", "PATTERN",
    "WARNING", "SOLUTION") but MemoryType enum values are lowercase, so the filter
    silently returned nothing.
    """

    def _make_mixin(self):
        from vetinari.orchestration.pipeline_helpers import PipelineHelpersMixin

        class _Concrete(PipelineHelpersMixin):
            pass

        return _Concrete()

    def test_search_called_with_lowercase_entry_types(self):
        """store.search() must receive lowercase entry_types matching MemoryType enum."""
        from unittest.mock import patch

        mixin = self._make_mixin()
        mock_store = MagicMock()
        mock_store.search.return_value = []

        # The import is local inside the function body, so patch at the source module
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            mixin._retrieve_memory_for_planning("implement user authentication")

        mock_store.search.assert_called_once()
        call_args = mock_store.search.call_args
        entry_types = call_args.kwargs.get("entry_types") or (
            call_args.args[1] if len(call_args.args) > 1 else None
        )
        assert entry_types is not None, "entry_types arg was not passed to store.search()"
        for et in entry_types:
            assert et == et.lower(), (
                f"entry_type {et!r} is not lowercase — would silently miss MemoryType enum matches"
            )

    def test_search_receives_expected_type_set(self):
        """The four canonical entry types are passed: decision, pattern, warning, solution."""
        from unittest.mock import patch

        mixin = self._make_mixin()
        mock_store = MagicMock()
        mock_store.search.return_value = []

        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            return_value=mock_store,
        ):
            mixin._retrieve_memory_for_planning("refactor the auth module")

        call_args = mock_store.search.call_args
        entry_types = set(
            call_args.kwargs.get("entry_types") or (
                call_args.args[1] if len(call_args.args) > 1 else []
            )
        )
        assert entry_types == {"decision", "pattern", "warning", "solution"}

    def test_memory_store_unavailable_returns_empty_list(self):
        """When memory store raises, returns empty list (does not propagate)."""
        from unittest.mock import patch

        mixin = self._make_mixin()

        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            result = mixin._retrieve_memory_for_planning("any goal")

        assert result == []
