"""Tests for vetinari.cascade_router — cost-optimised cascade routing."""

from __future__ import annotations

from vetinari.cascade_router import (
    CascadeResult,
    CascadeRouter,
    CascadeTier,
    _heuristic_confidence,
)


class TestCascadeTier:
    """Tests for the CascadeTier dataclass."""

    def test_defaults(self):
        tier = CascadeTier(model_id="small-7b")
        assert tier.cost_per_1k_tokens == 0.0
        assert tier.priority == 0
        assert tier.max_tokens_override is None
        assert tier.tags == []

    def test_fields(self):
        tier = CascadeTier(
            model_id="large-72b",
            cost_per_1k_tokens=0.015,
            priority=2,
            tags=["coding", "reasoning"],
        )
        assert tier.model_id == "large-72b"
        assert tier.cost_per_1k_tokens == 0.015
        assert tier.priority == 2
        assert len(tier.tags) == 2


class TestCascadeResult:
    """Tests for the CascadeResult dataclass."""

    def test_to_dict(self):
        result = CascadeResult(
            response="Hello",
            model_id="model-a",
            confidence=0.85,
            escalation_count=1,
            tiers_tried=["small", "medium"],
        )
        d = result.to_dict()
        assert d["model_id"] == "model-a"
        assert d["confidence"] == 0.85
        assert d["escalation_count"] == 1
        assert len(d["tiers_tried"]) == 2


class TestHeuristicConfidence:
    """Tests for the _heuristic_confidence function."""

    def test_empty_response(self):
        assert _heuristic_confidence("") == 0.0

    def test_confident_response(self):
        text = "The answer is 42. This is based on the calculation of 6 times 7."
        score = _heuristic_confidence(text)
        assert score > 0.7

    def test_uncertain_response(self):
        text = "I'm not sure about this, it is unclear what the answer should be."
        score = _heuristic_confidence(text)
        assert score < 0.7

    def test_short_response_lowers_confidence(self):
        score = _heuristic_confidence("Yes")
        assert score < 0.7

    def test_refusal_lowers_confidence(self):
        text = "I can't help with that request because I'm not able to process it."
        score = _heuristic_confidence(text)
        assert score < 1.0  # refusal phrases reduce confidence from 1.0


class TestCascadeRouter:
    """Tests for the CascadeRouter class."""

    def test_add_tier(self):
        router = CascadeRouter()
        router.add_tier("small", cost_per_1k=0.001, priority=0)
        router.add_tier("large", cost_per_1k=0.015, priority=1)
        tiers = router.get_tiers()
        assert len(tiers) == 2

    def test_add_tier_sorts_by_priority(self):
        router = CascadeRouter()
        router.add_tier("large", cost_per_1k=0.015, priority=2)
        router.add_tier("small", cost_per_1k=0.001, priority=0)
        tiers = router.get_tiers()
        # get_tiers returns CascadeTier objects or dicts; handle both
        first_id = tiers[0].model_id if hasattr(tiers[0], "model_id") else tiers[0]["model_id"]
        second_id = tiers[1].model_id if hasattr(tiers[1], "model_id") else tiers[1]["model_id"]
        assert first_id == "small"
        assert second_id == "large"

    def test_get_stats(self):
        router = CascadeRouter()
        stats = router.get_stats()
        assert isinstance(stats, dict)
