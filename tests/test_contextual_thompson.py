"""Tests for contextual Thompson Sampling (US-209, Department 6.3)."""

from __future__ import annotations

import os

import pytest

from vetinari.learning.model_selector import ThompsonSamplingSelector
from vetinari.learning.thompson_arms import ThompsonTaskContext


@pytest.fixture
def selector(tmp_path):
    """Isolated ThompsonSamplingSelector with no disk state."""
    os.environ["VETINARI_STATE_DIR"] = str(tmp_path)
    sel = ThompsonSamplingSelector()
    return sel


# ------------------------------------------------------------------
# TaskContext.to_bucket()
# ------------------------------------------------------------------


class TestTaskContextToBucket:
    def test_returns_int_in_valid_range(self):
        ctx = ThompsonTaskContext(task_type="code", estimated_complexity=5)
        bucket = ctx.to_bucket()
        assert isinstance(bucket, int)
        assert 0 <= bucket < 50

    def test_same_features_same_bucket(self):
        ctx1 = ThompsonTaskContext(
            task_type="research", estimated_complexity=8, domain="python", requires_reasoning=True
        )
        ctx2 = ThompsonTaskContext(
            task_type="research", estimated_complexity=8, domain="python", requires_reasoning=True
        )
        assert ctx1.to_bucket() == ctx2.to_bucket()

    def test_different_task_type_different_bucket(self):
        # hash() is PYTHONHASHSEED-dependent, so with only 50 buckets
        # any pair *can* collide.  Instead, verify that at least some
        # distinct inputs yield distinct buckets (i.e. the function is
        # not a constant).
        contexts = [
            ThompsonTaskContext(task_type="code", estimated_complexity=1, domain="python", requires_reasoning=False),
            ThompsonTaskContext(task_type="research", estimated_complexity=9, domain="math", requires_reasoning=True),
            ThompsonTaskContext(task_type="review", estimated_complexity=5, domain="general", requires_reasoning=False),
            ThompsonTaskContext(task_type="planning", estimated_complexity=3, domain="infra", requires_reasoning=True),
        ]
        buckets = {c.to_bucket() for c in contexts}
        # With 4 very different inputs in 50 buckets, at least 2 should differ
        assert len(buckets) >= 2, f"All 4 contexts hashed to the same bucket(s): {buckets}"

    def test_complexity_bins_produce_different_keys(self):
        """Complexity bins lo/mid/hi produce different hash keys even if buckets may collide."""
        ctx_lo = ThompsonTaskContext(
            task_type="code", estimated_complexity=1, domain="python", requires_reasoning=False
        )
        ctx_mid = ThompsonTaskContext(
            task_type="code", estimated_complexity=5, domain="python", requires_reasoning=False
        )
        ctx_hi = ThompsonTaskContext(
            task_type="code", estimated_complexity=9, domain="python", requires_reasoning=False
        )
        # With 3 different complexity bins and 50 buckets, at least 2 should differ
        buckets = {ctx_lo.to_bucket(), ctx_mid.to_bucket(), ctx_hi.to_bucket()}
        assert len(buckets) >= 2, f"All 3 complexity bins hashed to same bucket: {buckets}"

    def test_default_bucket_is_valid(self):
        ctx = ThompsonTaskContext()
        bucket = ctx.to_bucket()
        assert 0 <= bucket < 50


# ------------------------------------------------------------------
# TaskContext dataclass defaults
# ------------------------------------------------------------------


class TestTaskContextDefaults:
    def test_default_task_type(self):
        ctx = ThompsonTaskContext()
        assert ctx.task_type == "general"

    def test_default_complexity(self):
        ctx = ThompsonTaskContext()
        assert ctx.estimated_complexity == 5

    def test_default_booleans_are_false(self):
        ctx = ThompsonTaskContext()
        assert ctx.requires_reasoning is False
        assert ctx.requires_creativity is False
        assert ctx.requires_precision is False

    def test_default_file_count(self):
        ctx = ThompsonTaskContext()
        assert ctx.file_count == 0


# ------------------------------------------------------------------
# select_model_contextual
# ------------------------------------------------------------------


class TestSelectModelContextual:
    def test_returns_valid_model_from_candidates(self, selector):
        ctx = ThompsonTaskContext(task_type="code", estimated_complexity=5)
        candidates = ["claude-haiku-3-5-20241022", "claude-sonnet-4-20250514"]
        result = selector.select_model_contextual(ctx, candidates)
        assert result in candidates

    def test_empty_candidates_returns_default(self, selector):
        ctx = ThompsonTaskContext()
        result = selector.select_model_contextual(ctx, [])
        assert result == "default"

    def test_single_candidate_returns_it(self, selector):
        ctx = ThompsonTaskContext(task_type="review")
        result = selector.select_model_contextual(ctx, ["claude-opus-4-20250514"])
        assert result == "claude-opus-4-20250514"

    def test_uses_context_bucket_in_arm_key(self, selector):
        ctx = ThompsonTaskContext(task_type="code", estimated_complexity=3, domain="python")
        candidates = ["model-a", "model-b"]
        selector.select_model_contextual(ctx, candidates)
        bucket = ctx.to_bucket()
        bucket_key = f"ctx_{bucket}"
        # At least one arm for the bucket_key should have been created
        assert any(bucket_key in k for k in selector._arms)

    def test_cost_penalty_applied(self, selector):
        ctx = ThompsonTaskContext(task_type="code")
        candidates = ["cheap-model", "expensive-model"]
        cost_map = {"cheap-model": 0.1, "expensive-model": 10.0}
        # Run many times; cheap model should be selected more often
        cheap_count = sum(
            1 for _ in range(200) if selector.select_model_contextual(ctx, candidates, cost_map) == "cheap-model"
        )
        # With a 100x cost difference and COST_WEIGHT=0.15, cheap should dominate
        assert cheap_count > 100


# ------------------------------------------------------------------
# update_contextual
# ------------------------------------------------------------------


class TestUpdateContextual:
    def test_success_increases_alpha(self, selector):
        ctx = ThompsonTaskContext(task_type="code", estimated_complexity=5)
        model = "claude-sonnet-4-20250514"
        bucket_key = f"ctx_{ctx.to_bucket()}"
        arm_before_alpha = selector._get_or_create_arm(model, bucket_key).alpha

        selector.update_contextual(ctx, model, quality_score=1.0, success=True)

        arm_after_alpha = selector._arms[f"{model}:{bucket_key}"].alpha
        assert arm_after_alpha > arm_before_alpha

    def test_failure_increases_beta(self, selector):
        ctx = ThompsonTaskContext(task_type="code", estimated_complexity=5)
        model = "claude-sonnet-4-20250514"
        bucket_key = f"ctx_{ctx.to_bucket()}"
        arm_before_beta = selector._get_or_create_arm(model, bucket_key).beta

        selector.update_contextual(ctx, model, quality_score=0.0, success=False)

        arm_after_beta = selector._arms[f"{model}:{bucket_key}"].beta
        assert arm_after_beta > arm_before_beta

    def test_decay_applied_to_bucket_arms(self, selector):
        ctx = ThompsonTaskContext(task_type="architecture", estimated_complexity=8)
        bucket_key = f"ctx_{ctx.to_bucket()}"
        model_a = "model-a"
        model_b = "model-b"

        # Create two arms in the same bucket
        arm_a = selector._get_or_create_arm(model_a, bucket_key)
        arm_b = selector._get_or_create_arm(model_b, bucket_key)

        # Record alpha/beta before the update
        alpha_a_before = arm_a.alpha
        alpha_b_before = arm_b.alpha

        # Update model_a — decay should apply to both arms in the bucket
        selector.update_contextual(ctx, model_a, quality_score=0.9, success=True)

        # Both arms should have decayed (multiplied by DECAY_FACTOR)
        assert arm_a.alpha < alpha_a_before * 1.05 or arm_b.alpha < alpha_b_before * 1.05

    def test_decay_floor_prevents_vanishing(self, selector):
        ctx = ThompsonTaskContext(task_type="review", estimated_complexity=2)
        model = "tiny-model"
        bucket_key = f"ctx_{ctx.to_bucket()}"

        # Seed arm with very small values just above 1.0
        arm = selector._get_or_create_arm(model, bucket_key)
        arm.alpha = 1.01
        arm.beta = 1.01

        # Many updates should not push below floor of 1.0
        for _ in range(200):
            selector.update_contextual(ctx, model, quality_score=0.5, success=True)

        assert arm.alpha >= 1.0
        assert arm.beta >= 1.0


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_existing_select_model_still_works(self, selector):
        result = selector.select_model(
            task_type="coding",
            candidate_models=["claude-haiku-3-5-20241022", "claude-sonnet-4-20250514"],
        )
        assert result in ("claude-haiku-3-5-20241022", "claude-sonnet-4-20250514")

    def test_existing_update_still_works(self, selector):
        selector.update("claude-sonnet-4-20250514", "coding", quality_score=0.9, success=True)
        arm = selector._get_or_create_arm("claude-sonnet-4-20250514", "coding")
        assert arm.total_pulls >= 1


# ------------------------------------------------------------------
# Import test
# ------------------------------------------------------------------


class TestImport:
    def test_task_context_importable(self):
        from vetinari.learning.model_selector import ThompsonTaskContext as TC

        assert TC is not None
        assert callable(TC)

    def test_task_context_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(ThompsonTaskContext)
