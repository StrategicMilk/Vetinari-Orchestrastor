"""Tests for vetinari.models.inference_config.

Covers:
- InferenceConfig dataclass construction and field access
- BudgetPolicy tier budget assignment (express / standard / custom)
- Complexity scaling: low (1-3) -> 50%, mid (4-7) -> 100%, high (8-10) -> 150%
- Unknown tier fallback to "standard"
- Custom tier_budgets and defaults override
- Singleton accessor get_budget_policy() and reset behaviour
"""

from __future__ import annotations

import pytest

from vetinari.models.inference_config import (
    BudgetPolicy,
    InferenceConfig,
    _reset_policy_for_testing,
    get_budget_policy,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a clean singleton."""
    _reset_policy_for_testing()
    yield
    _reset_policy_for_testing()


# ── InferenceConfig dataclass ──────────────────────────────────────────────────


class TestInferenceConfig:
    def test_construction_with_all_fields(self):
        cfg = InferenceConfig(
            thinking_budget=2048,
            max_tokens=1024,
            temperature=0.5,
            tier="standard",
        )
        assert cfg.thinking_budget == 2048
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.5
        assert cfg.tier == "standard"

    def test_construction_with_none_budget(self):
        cfg = InferenceConfig(
            thinking_budget=None,
            max_tokens=None,
            temperature=0.0,
            tier="express",
        )
        assert cfg.thinking_budget is None
        assert cfg.max_tokens is None

    def test_tier_field_stored(self):
        cfg = InferenceConfig(thinking_budget=100, max_tokens=200, temperature=0.1, tier="custom")
        assert cfg.tier == "custom"


# ── BudgetPolicy tier defaults ─────────────────────────────────────────────────


class TestBudgetPolicyTierDefaults:
    def test_express_tier_base_budget_at_mid_complexity(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("express", complexity=5)
        # mid complexity multiplier 1.0, base 1024
        assert cfg.thinking_budget == 1024

    def test_standard_tier_base_budget_at_mid_complexity(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=5)
        # mid complexity multiplier 1.0, base 4096
        assert cfg.thinking_budget == 4096

    def test_custom_tier_base_budget_at_mid_complexity(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("custom", complexity=5)
        # mid complexity multiplier 1.0, base 16384
        assert cfg.thinking_budget == 16384

    def test_tier_stored_on_returned_config(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("express", complexity=5)
        assert cfg.tier == "express"

    def test_case_insensitive_tier(self):
        policy = BudgetPolicy()
        cfg_lower = policy.get_config("standard", complexity=5)
        cfg_upper = policy.get_config("STANDARD", complexity=5)
        assert cfg_lower.thinking_budget == cfg_upper.thinking_budget

    def test_available_tiers_contains_defaults(self):
        policy = BudgetPolicy()
        tiers = policy.available_tiers()
        assert "express" in tiers
        assert "standard" in tiers
        assert "custom" in tiers

    def test_available_tiers_returns_list(self):
        policy = BudgetPolicy()
        assert isinstance(policy.available_tiers(), list)


# ── Complexity scaling ─────────────────────────────────────────────────────────


class TestComplexityScaling:
    """Complexity 1-3 -> 50%, 4-7 -> 100%, 8-10 -> 150% of tier base."""

    # Standard base = 4096 for easy arithmetic verification.

    def test_low_complexity_1_reduces_budget_by_half(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=1)
        assert cfg.thinking_budget == int(4096 * 0.5)

    def test_low_complexity_2_reduces_budget_by_half(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=2)
        assert cfg.thinking_budget == int(4096 * 0.5)

    def test_low_complexity_3_reduces_budget_by_half(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=3)
        assert cfg.thinking_budget == int(4096 * 0.5)

    def test_mid_complexity_4_uses_full_budget(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=4)
        assert cfg.thinking_budget == 4096

    def test_mid_complexity_5_uses_full_budget(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=5)
        assert cfg.thinking_budget == 4096

    def test_mid_complexity_7_uses_full_budget(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=7)
        assert cfg.thinking_budget == 4096

    def test_high_complexity_8_increases_budget_by_half(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=8)
        assert cfg.thinking_budget == int(4096 * 1.5)

    def test_high_complexity_10_increases_budget_by_half(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("standard", complexity=10)
        assert cfg.thinking_budget == int(4096 * 1.5)

    def test_express_low_complexity(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("express", complexity=1)
        assert cfg.thinking_budget == int(1024 * 0.5)

    def test_custom_high_complexity(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("custom", complexity=10)
        assert cfg.thinking_budget == int(16384 * 1.5)

    def test_complexity_clamped_below_1_treated_as_1(self):
        policy = BudgetPolicy()
        cfg_clamped = policy.get_config("standard", complexity=0)
        cfg_one = policy.get_config("standard", complexity=1)
        assert cfg_clamped.thinking_budget == cfg_one.thinking_budget

    def test_complexity_clamped_above_10_treated_as_10(self):
        policy = BudgetPolicy()
        cfg_clamped = policy.get_config("standard", complexity=99)
        cfg_ten = policy.get_config("standard", complexity=10)
        assert cfg_clamped.thinking_budget == cfg_ten.thinking_budget


# ── Unknown tier fallback ──────────────────────────────────────────────────────


class TestUnknownTierFallback:
    def test_unknown_tier_falls_back_to_standard_budget(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("premium", complexity=5)
        # Should use standard base (4096) at mid complexity
        assert cfg.thinking_budget == 4096

    def test_unknown_tier_tier_label_becomes_standard(self):
        policy = BudgetPolicy()
        cfg = policy.get_config("nonexistent", complexity=5)
        assert cfg.tier == "standard"


# ── Custom tier_budgets and defaults ──────────────────────────────────────────


class TestCustomTierBudgets:
    def test_custom_budget_overrides_express_default(self):
        policy = BudgetPolicy(tier_budgets={"express": 512})
        cfg = policy.get_config("express", complexity=5)
        assert cfg.thinking_budget == 512

    def test_new_tier_added_to_policy(self):
        policy = BudgetPolicy(tier_budgets={"turbo": 8192})
        cfg = policy.get_config("turbo", complexity=5)
        assert cfg.thinking_budget == 8192

    def test_custom_max_tokens_applied(self):
        policy = BudgetPolicy(max_tokens=2000)
        cfg = policy.get_config("standard", complexity=5)
        assert cfg.max_tokens == 2000

    def test_custom_temperature_applied(self):
        policy = BudgetPolicy(temperature=0.2)
        cfg = policy.get_config("standard", complexity=5)
        assert cfg.temperature == pytest.approx(0.2)

    def test_existing_tiers_preserved_when_adding_custom(self):
        policy = BudgetPolicy(tier_budgets={"turbo": 8192})
        cfg = policy.get_config("standard", complexity=5)
        assert cfg.thinking_budget == 4096


# ── Singleton accessor ─────────────────────────────────────────────────────────


class TestGetBudgetPolicySingleton:
    def test_returns_budget_policy_instance(self):
        policy = get_budget_policy()
        assert isinstance(policy, BudgetPolicy)

    def test_same_instance_on_repeated_calls(self):
        p1 = get_budget_policy()
        p2 = get_budget_policy()
        assert p1 is p2

    def test_second_call_ignores_different_args(self):
        """Once constructed, subsequent calls with different args return same instance."""
        p1 = get_budget_policy(tier_budgets={"express": 999})
        p2 = get_budget_policy(tier_budgets={"express": 1})
        assert p1 is p2

    def test_reset_allows_reconstruction(self):
        p1 = get_budget_policy()
        _reset_policy_for_testing()
        p2 = get_budget_policy()
        assert p1 is not p2

    def test_singleton_produces_valid_config(self):
        policy = get_budget_policy()
        cfg = policy.get_config("standard", complexity=5)
        assert isinstance(cfg, InferenceConfig)
        assert cfg.thinking_budget == 4096

    def test_singleton_config_scales_with_complexity(self):
        policy = get_budget_policy()
        low = policy.get_config("standard", complexity=2)
        high = policy.get_config("standard", complexity=9)
        assert high.thinking_budget > low.thinking_budget
