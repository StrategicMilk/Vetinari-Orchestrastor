"""Tests for vetinari.learning.operator_selector — Thompson Sampling for mutation operators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.learning.operator_selector import OperatorArm, OperatorSelector
from vetinari.learning.prompt_mutator import MutationOperator

# ── OperatorArm tests ────────────────────────────────────────────────


class TestOperatorArm:
    """Tests for the OperatorArm dataclass."""

    def test_default_uniform_prior(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.mean == 0.5

    def test_sample_returns_float_in_range(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        for _ in range(100):
            sample = arm.sample()
            assert 0.0 <= sample <= 1.0

    def test_update_positive_increases_alpha(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        initial_alpha = arm.alpha
        arm.update(quality_delta=0.1)
        assert arm.alpha > initial_alpha
        assert arm.total_pulls == 1

    def test_update_negative_increases_beta(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        initial_beta = arm.beta
        arm.update(quality_delta=-0.1)
        assert arm.beta > initial_beta
        assert arm.total_pulls == 1

    def test_update_zero_increases_beta(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        initial_beta = arm.beta
        arm.update(quality_delta=0.0)
        assert arm.beta > initial_beta

    def test_repeated_positive_updates_increase_mean(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        for _ in range(20):
            arm.update(quality_delta=0.2)
        assert arm.mean > 0.5

    def test_repeated_negative_updates_decrease_mean(self) -> None:
        arm = OperatorArm(operator="test", agent_type="builder", mode="build")
        for _ in range(20):
            arm.update(quality_delta=-0.2)
        assert arm.mean < 0.5


# ── OperatorSelector tests ───────────────────────────────────────────


class TestOperatorSelector:
    """Tests for the OperatorSelector class."""

    @pytest.fixture
    def tmp_state_path(self, tmp_path: Path) -> Path:
        return tmp_path / "operator_state.json"

    @pytest.fixture
    def selector(self, tmp_state_path: Path) -> OperatorSelector:
        return OperatorSelector(state_path=tmp_state_path)

    def test_select_returns_valid_operator(self, selector: OperatorSelector) -> None:
        op = selector.select_operator("builder", "build")
        assert isinstance(op, MutationOperator)

    def test_select_returns_different_operators_sometimes(self, selector: OperatorSelector) -> None:
        """Thompson Sampling should explore — not always return the same operator."""
        results = {selector.select_operator("builder", "build") for _ in range(50)}
        # With uniform priors, we should see at least 2 different operators
        assert len(results) >= 2

    def test_update_changes_arm_parameters(self, selector: OperatorSelector) -> None:
        op = MutationOperator.INSTRUCTION_REPHRASE
        selector.update(op, "builder", "build", quality_delta=0.5)
        stats = selector.get_stats("builder", "build")
        rephrase_stats = [s for s in stats if s["operator"] == op.value]
        assert len(rephrase_stats) == 1
        assert rephrase_stats[0]["pulls"] == 1
        # Alpha should have increased (positive delta)
        assert rephrase_stats[0]["alpha"] > 1.0

    def test_positive_updates_increase_selection_probability(
        self,
        selector: OperatorSelector,
    ) -> None:
        """An operator with many positive updates should be selected more often."""
        target_op = MutationOperator.REASONING_SCAFFOLD
        for _ in range(30):
            selector.update(target_op, "builder", "build", quality_delta=0.5)

        # Sample many times and check the favoured operator wins often
        selections = [selector.select_operator("builder", "build") for _ in range(100)]
        target_count = selections.count(target_op)
        assert target_count > 20  # Should be selected frequently

    def test_different_agent_modes_have_separate_arms(
        self,
        selector: OperatorSelector,
    ) -> None:
        """Arms for builder/build should be independent from oracle/architecture."""
        op = MutationOperator.CONSTRAINT_INJECTION
        selector.update(op, "builder", "build", quality_delta=0.5)
        selector.update(op, "oracle", "architecture", quality_delta=-0.5)

        builder_stats = selector.get_stats("builder", "build")
        oracle_stats = selector.get_stats("oracle", "architecture")

        builder_ci = next(s for s in builder_stats if s["operator"] == op.value)
        oracle_ci = next(s for s in oracle_stats if s["operator"] == op.value)

        # Builder arm should be optimistic, oracle arm pessimistic
        assert builder_ci["mean"] > oracle_ci["mean"]

    def test_get_stats_returns_all_operators(self, selector: OperatorSelector) -> None:
        # Update one operator to create at least one arm
        selector.update(MutationOperator.CONTEXT_PRUNE, "builder", "build", 0.1)
        stats = selector.get_stats("builder", "build")
        assert len(stats) >= 1
        assert stats[0]["operator"] == "context_prune"

    def test_get_stats_filter_by_agent(self, selector: OperatorSelector) -> None:
        selector.update(MutationOperator.CONTEXT_PRUNE, "builder", "build", 0.1)
        selector.update(MutationOperator.CONTEXT_PRUNE, "oracle", "risk", 0.1)
        builder_stats = selector.get_stats("builder")
        assert all(s["agent_type"] == "builder" for s in builder_stats)

    def test_get_best_operator_none_before_pulls(self, selector: OperatorSelector) -> None:
        result = selector.get_best_operator("builder", "build")
        assert result is None

    def test_get_best_operator_after_pulls(self, selector: OperatorSelector) -> None:
        op = MutationOperator.EXAMPLE_INJECTION
        for _ in range(10):
            selector.update(op, "builder", "build", quality_delta=0.5)
        result = selector.get_best_operator("builder", "build")
        assert result == op


# ── Persistence tests ────────────────────────────────────────────────


class TestOperatorSelectorPersistence:
    """Tests for state persistence."""

    @pytest.fixture
    def tmp_state_path(self, tmp_path: Path) -> Path:
        return tmp_path / "operator_state.json"

    def test_save_and_load_round_trip(self, tmp_state_path: Path) -> None:
        selector1 = OperatorSelector(state_path=tmp_state_path)
        selector1.update(MutationOperator.REASONING_SCAFFOLD, "builder", "build", 0.3)
        selector1.update(MutationOperator.CONTEXT_PRUNE, "oracle", "risk", -0.1)

        # Create new selector from same path — should load persisted state
        selector2 = OperatorSelector(state_path=tmp_state_path)
        stats = selector2.get_stats()
        assert len(stats) == 2

        scaffold_stats = [s for s in stats if s["operator"] == "reasoning_scaffold"]
        assert len(scaffold_stats) == 1
        assert scaffold_stats[0]["pulls"] == 1

    def test_empty_state_file_handled(self, tmp_state_path: Path) -> None:
        tmp_state_path.write_text("{}", encoding="utf-8")
        selector = OperatorSelector(state_path=tmp_state_path)
        op = selector.select_operator("builder", "build")
        assert isinstance(op, MutationOperator)

    def test_corrupt_state_file_handled(self, tmp_state_path: Path) -> None:
        tmp_state_path.write_text("not json", encoding="utf-8")
        selector = OperatorSelector(state_path=tmp_state_path)
        op = selector.select_operator("builder", "build")
        assert isinstance(op, MutationOperator)

    def test_state_file_written_on_update(self, tmp_state_path: Path) -> None:
        selector = OperatorSelector(state_path=tmp_state_path)
        selector.update(MutationOperator.ROLE_REINFORCEMENT, "builder", "build", 0.2)
        assert tmp_state_path.exists()
        data = json.loads(tmp_state_path.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_operator_selector_singleton_path_respects_env_override(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """VETINARI_STATE_DIR env override changes OperatorSelector default state path.

        The singleton path is resolved at construction time via
        _default_state_path(), which reads VETINARI_STATE_DIR from the
        environment.  This proves the env override reaches the path without
        requiring the singleton to be torn down.
        """
        custom_state = tmp_path / "custom_state"
        custom_state.mkdir(parents=True)
        monkeypatch.setenv("VETINARI_STATE_DIR", str(custom_state))

        path = OperatorSelector._default_state_path()

        assert path.parent == custom_state
        assert path.name == "operator_selector_state.json"
