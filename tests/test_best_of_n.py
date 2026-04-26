"""Tests for vetinari.models.best_of_n — Best-of-N candidate selection."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from vetinari.models.best_of_n import BestOfNSelector, get_n_for_tier

# ---------------------------------------------------------------------------
# get_n_for_tier
# ---------------------------------------------------------------------------


class TestGetNForTier:
    def test_custom_tier_returns_four(self) -> None:
        assert get_n_for_tier("custom") == 4

    def test_standard_tier_returns_two(self) -> None:
        assert get_n_for_tier("standard") == 2

    def test_express_tier_returns_one(self) -> None:
        assert get_n_for_tier("express") == 1

    def test_unknown_tier_falls_back_to_one(self) -> None:
        assert get_n_for_tier("unknown_tier") == 1

    def test_case_insensitive(self) -> None:
        assert get_n_for_tier("CUSTOM") == 4
        assert get_n_for_tier("Standard") == 2
        assert get_n_for_tier("EXPRESS") == 1

    def test_strips_whitespace(self) -> None:
        assert get_n_for_tier("  custom  ") == 4

    def test_none_tier_returns_one(self) -> None:
        """None tier must not crash — returns 1 as a safe passthrough value."""
        assert get_n_for_tier(None) == 1  # type: ignore[arg-type]

    def test_integer_tier_returns_one(self) -> None:
        """Non-string tier (e.g. int from a misconfigured caller) returns 1."""
        assert get_n_for_tier(42) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BestOfNSelector.generate_and_select
# ---------------------------------------------------------------------------


class TestBestOfNSelector:
    @pytest.fixture
    def candidates(self) -> list[str]:
        return ["response_a", "response_b", "response_c", "response_d"]

    @pytest.fixture
    def selector_with_sequence(self, candidates: list[str]) -> BestOfNSelector:
        """Selector whose generate_fn returns candidates in order."""
        generate_fn = MagicMock(side_effect=candidates)
        return BestOfNSelector(generate_fn=generate_fn)

    def test_n_candidates_generated(self, selector_with_sequence: BestOfNSelector) -> None:
        scorer = MagicMock(return_value=0.5)
        selector_with_sequence.generate_and_select("prompt", n=3, scorer=scorer)
        assert selector_with_sequence.generate_fn.call_count == 3  # type: ignore[attr-defined]

    def test_generate_fn_called_with_prompt(self, selector_with_sequence: BestOfNSelector) -> None:
        scorer = MagicMock(return_value=0.5)
        selector_with_sequence.generate_and_select("my_prompt", n=2, scorer=scorer)
        selector_with_sequence.generate_fn.assert_called_with("my_prompt")  # type: ignore[attr-defined]

    def test_highest_scoring_candidate_returned(self) -> None:
        candidates = ["low_quality", "high_quality", "medium_quality"]
        generate_fn = MagicMock(side_effect=candidates)
        selector = BestOfNSelector(generate_fn=generate_fn)

        scores = {"low_quality": 0.2, "high_quality": 0.9, "medium_quality": 0.5}
        scorer = MagicMock(side_effect=lambda c: scores[c])

        result = selector.generate_and_select("prompt", n=3, scorer=scorer)
        assert result == "high_quality"

    def test_scorer_called_for_each_candidate(self) -> None:
        candidates = ["a", "b", "c"]
        generate_fn = MagicMock(side_effect=candidates)
        selector = BestOfNSelector(generate_fn=generate_fn)
        scorer = MagicMock(return_value=0.5)

        selector.generate_and_select("prompt", n=3, scorer=scorer)

        assert scorer.call_count == 3
        scorer.assert_has_calls([call("a"), call("b"), call("c")])

    def test_n_equals_one_passthrough_no_scorer_called(self) -> None:
        generate_fn = MagicMock(return_value="only_candidate")
        selector = BestOfNSelector(generate_fn=generate_fn)
        scorer = MagicMock()

        result = selector.generate_and_select("prompt", n=1, scorer=scorer)

        assert result == "only_candidate"
        generate_fn.assert_called_once_with("prompt")
        scorer.assert_not_called()

    def test_n_equals_one_generate_fn_called_once(self) -> None:
        generate_fn = MagicMock(return_value="single")
        selector = BestOfNSelector(generate_fn=generate_fn)

        selector.generate_and_select("p", n=1, scorer=MagicMock())

        assert generate_fn.call_count == 1

    def test_first_candidate_wins_on_score_tie(self) -> None:
        candidates = ["first", "second"]
        generate_fn = MagicMock(side_effect=candidates)
        selector = BestOfNSelector(generate_fn=generate_fn)
        scorer = MagicMock(return_value=0.7)  # identical scores

        result = selector.generate_and_select("prompt", n=2, scorer=scorer)
        assert result == "first"

    def test_invalid_n_raises_value_error(self) -> None:
        selector = BestOfNSelector(generate_fn=MagicMock())
        with pytest.raises(ValueError, match="n must be >= 1"):
            selector.generate_and_select("prompt", n=0, scorer=MagicMock())

    def test_custom_n_override(self) -> None:
        """Callers can pass any N value, ignoring tier defaults."""
        candidates = ["x"] * 7
        generate_fn = MagicMock(side_effect=candidates)
        selector = BestOfNSelector(generate_fn=generate_fn)
        scorer = MagicMock(return_value=0.5)

        selector.generate_and_select("prompt", n=7, scorer=scorer)

        assert generate_fn.call_count == 7

    def test_last_candidate_wins_when_last_is_best(self) -> None:
        candidates = ["poor", "mediocre", "excellent"]
        generate_fn = MagicMock(side_effect=candidates)
        selector = BestOfNSelector(generate_fn=generate_fn)

        scores = {"poor": 0.1, "mediocre": 0.4, "excellent": 0.95}
        scorer = MagicMock(side_effect=lambda c: scores[c])

        result = selector.generate_and_select("prompt", n=3, scorer=scorer)
        assert result == "excellent"
