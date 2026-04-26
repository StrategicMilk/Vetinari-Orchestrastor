"""Tests for Thompson Sampling strategy selection (US-507)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.exceptions import ConfigurationError
from vetinari.learning.model_selector import ThompsonSamplingSelector
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def selector(tmp_path, monkeypatch):
    """Fresh ThompsonSamplingSelector with isolated state dir."""
    monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))
    return ThompsonSamplingSelector()


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------


class TestSelectStrategy:
    """Tests for the select_strategy method."""

    def test_returns_valid_prompt_template_variant(self, selector):
        """select_strategy returns a value from the defined value space."""
        result = selector.select_strategy(AgentType.WORKER.value, "build", "prompt_template_variant")
        assert result in ["standard", "concise", "detailed", "structured"]

    def test_returns_valid_context_window_size(self, selector):
        """select_strategy returns a valid context_window_size."""
        result = selector.select_strategy(AgentType.FOREMAN.value, "plan", "context_window_size")
        assert result in [2048, 4096, 8192, 16384]

    def test_returns_valid_temperature(self, selector):
        """select_strategy returns a valid temperature value."""
        result = selector.select_strategy("ORACLE", "analysis", "temperature")
        assert result in [0.0, 0.3, 0.5, 0.7, 1.0]

    def test_returns_valid_decomposition_granularity(self, selector):
        """select_strategy returns a valid decomposition_granularity."""
        result = selector.select_strategy(AgentType.FOREMAN.value, "plan", "decomposition_granularity")
        assert result in ["coarse", "medium", "fine"]

    def test_invalid_strategy_key_raises(self, selector):
        """select_strategy raises ValueError for unknown strategy keys."""
        with pytest.raises(ConfigurationError, match="Unknown strategy_key"):
            selector.select_strategy(AgentType.WORKER.value, "build", "nonexistent_key")

    def test_multiple_calls_return_valid_values(self, selector):
        """Repeated calls always produce values from the defined space."""
        for _ in range(50):
            result = selector.select_strategy(AgentType.INSPECTOR.value, "review", "temperature")
            assert result in [0.0, 0.3, 0.5, 0.7, 1.0]


# ---------------------------------------------------------------------------
# update_strategy
# ---------------------------------------------------------------------------


class TestUpdateStrategy:
    """Tests for the update_strategy method."""

    def test_high_quality_increases_alpha(self, selector):
        """update_strategy with quality > 0.5 increments alpha by 1."""
        # Create the arm first
        selector.select_strategy(AgentType.WORKER.value, "build", "temperature")
        arm_key = "strategy:WORKER:build:temperature:0.3:strategy"
        arm = selector._arms.get(arm_key)

        if arm is None:
            # The arm key depends on which value was selected; find the right one
            strategy_arms = {
                k: v for k, v in selector._arms.items() if k.startswith("strategy:WORKER:build:temperature:")
            }
            arm_key = next(iter(strategy_arms))
            arm = strategy_arms[arm_key]

        before_alpha = arm.alpha

        # Extract the value from the key: "strategy:WORKER:build:temperature:<value>:strategy"
        value = arm_key.split(":")[4]
        # Convert numeric values
        try:
            value = float(value)
        except ValueError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass  # value stays as string — expected for non-numeric arms

        selector.update_strategy(AgentType.WORKER.value, "build", "temperature", value, quality_score=0.8)

        assert arm.alpha == before_alpha + 1.0

    def test_low_quality_increases_beta(self, selector):
        """update_strategy with quality <= 0.5 increments beta by 1."""
        arm_key = "strategy:INSPECTOR:review:temperature:0.3:strategy"
        selector._get_or_create_arm("strategy:INSPECTOR:review:temperature:0.3", "strategy")
        arm = selector._arms[arm_key]
        before_beta = arm.beta

        selector.update_strategy(AgentType.INSPECTOR.value, "review", "temperature", 0.3, quality_score=0.2)

        assert arm.beta == before_beta + 1.0

    def test_boundary_quality_half_increases_beta(self, selector):
        """quality_score of exactly 0.5 is treated as failure (beta += 1)."""
        arm_key = "strategy:WORKER:build:temperature:0.5:strategy"
        selector._get_or_create_arm("strategy:WORKER:build:temperature:0.5", "strategy")
        arm = selector._arms[arm_key]
        before_beta = arm.beta

        selector.update_strategy(AgentType.WORKER.value, "build", "temperature", 0.5, quality_score=0.5)

        assert arm.beta == before_beta + 1.0

    def test_update_increments_total_pulls(self, selector):
        """update_strategy increments total_pulls on the arm."""
        arm_key = "strategy:WORKER:build:temperature:0.7:strategy"
        selector._get_or_create_arm("strategy:WORKER:build:temperature:0.7", "strategy")
        arm = selector._arms[arm_key]

        selector.update_strategy(AgentType.WORKER.value, "build", "temperature", 0.7, quality_score=0.9)

        assert arm.total_pulls == 1

    def test_invalid_strategy_key_raises(self, selector):
        """update_strategy raises ValueError for unknown strategy keys."""
        with pytest.raises(ConfigurationError, match="Unknown strategy_key"):
            selector.update_strategy(AgentType.WORKER.value, "build", "bad_key", "value", quality_score=0.8)


# ---------------------------------------------------------------------------
# Arm separation: strategy arms vs model arms
# ---------------------------------------------------------------------------


class TestArmSeparation:
    """Verify strategy arms are separate from model and mode arms."""

    def test_strategy_arms_use_strategy_prefix(self, selector):
        """Strategy arms are keyed with 'strategy:' prefix."""
        selector.select_strategy(AgentType.WORKER.value, "build", "temperature")

        strategy_keys = [k for k in selector._arms if k.startswith("strategy:")]
        assert len(strategy_keys) > 0

    def test_strategy_arms_do_not_collide_with_model_arms(self, selector):
        """Strategy and model arms occupy disjoint key spaces."""
        # Create a model arm
        selector.select_model("coding", ["claude-sonnet-4-20250514"])
        # Create a strategy arm
        selector.select_strategy(AgentType.WORKER.value, "build", "temperature")

        model_keys = {k for k in selector._arms if not k.startswith(("strategy:", "mode_", "tier_", "ctx_"))}
        strategy_keys = {k for k in selector._arms if k.startswith("strategy:")}

        assert model_keys.isdisjoint(strategy_keys)

    def test_strategy_arms_do_not_collide_with_mode_arms(self, selector):
        """Strategy and mode arms occupy disjoint key spaces."""
        selector.select_mode(AgentType.WORKER.value, "coding", ["build", "refactor"])
        selector.select_strategy(AgentType.WORKER.value, "build", "prompt_template_variant")

        mode_keys = {k for k in selector._arms if k.startswith("mode_")}
        strategy_keys = {k for k in selector._arms if k.startswith("strategy:")}

        assert mode_keys.isdisjoint(strategy_keys)

    def test_updating_strategy_does_not_affect_model_arm(self, selector):
        """Updating a strategy arm leaves model arms unchanged."""
        selector.update("claude-sonnet-4-20250514", "coding", quality_score=0.9, success=True)
        model_arm = selector._get_or_create_arm("claude-sonnet-4-20250514", "coding")
        alpha_before = model_arm.alpha

        selector.update_strategy(AgentType.WORKER.value, "build", "temperature", 0.3, quality_score=0.9)

        assert model_arm.alpha == alpha_before


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestStrategyPersistence:
    """Verify strategy arms survive save/load cycle."""

    def test_strategy_arms_persist_and_reload(self, tmp_path, monkeypatch):
        """Strategy arms are saved to SQLite and can be reloaded."""
        monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "vetinari.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        sel1 = ThompsonSamplingSelector()

        # Create and update a strategy arm
        sel1.update_strategy(AgentType.WORKER.value, "build", "temperature", 0.7, quality_score=0.9)
        sel1._save_state()

        # Verify state persisted to SQLite
        from vetinari.database import get_connection

        conn = get_connection()
        rows = conn.execute("SELECT arm_key FROM thompson_arms").fetchall()
        strategy_keys = [r[0] for r in rows if r[0].startswith("strategy:")]
        assert len(strategy_keys) > 0

        # Load into a fresh selector
        sel2 = ThompsonSamplingSelector()
        arm_key = "strategy:WORKER:build:temperature:0.7:strategy"
        assert arm_key in sel2._arms
        arm = sel2._arms[arm_key]
        # alpha should reflect the update (prior 1.0 + 1.0 from quality > 0.5)
        assert arm.alpha > 1.0
        reset_for_testing()

    def test_strategy_and_model_arms_coexist_in_state(self, tmp_path, monkeypatch):
        """Both strategy and model arms are saved in the same SQLite table."""
        monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "vetinari.db"))
        from vetinari.database import reset_for_testing

        reset_for_testing()

        sel = ThompsonSamplingSelector()

        sel.update("claude-sonnet-4-20250514", "coding", quality_score=0.9, success=True)
        sel.update_strategy(
            AgentType.INSPECTOR.value, "review", "prompt_template_variant", "concise", quality_score=0.8
        )
        sel._save_state()

        from vetinari.database import get_connection

        conn = get_connection()
        rows = conn.execute("SELECT arm_key FROM thompson_arms").fetchall()
        all_keys = [r[0] for r in rows]

        model_keys = [k for k in all_keys if not k.startswith(("strategy:", "mode_", "tier_", "ctx_"))]
        strategy_keys = [k for k in all_keys if k.startswith("strategy:")]

        assert len(model_keys) > 0
        assert len(strategy_keys) > 0
        reset_for_testing()


# ---------------------------------------------------------------------------
# Learned preference convergence
# ---------------------------------------------------------------------------


class TestStrategyConvergence:
    """Verify that repeated positive feedback shifts selection toward the rewarded value."""

    def test_rewarded_value_selected_more_often(self, selector):
        """After many positive updates for one value, it should be selected most often."""
        # Strongly reward "detailed" prompt template variant
        for _ in range(30):
            selector.update_strategy(
                AgentType.WORKER.value, "build", "prompt_template_variant", "detailed", quality_score=0.95
            )
        # Mildly punish others
        for other in ["standard", "concise", "structured"]:
            for _ in range(10):
                selector.update_strategy(
                    AgentType.WORKER.value, "build", "prompt_template_variant", other, quality_score=0.2
                )

        # Sample 100 times and count
        counts: dict[str, int] = {}
        for _ in range(100):
            val = selector.select_strategy(AgentType.WORKER.value, "build", "prompt_template_variant")
            counts[val] = counts.get(val, 0) + 1

        # "detailed" should dominate
        assert counts.get("detailed", 0) > 50
