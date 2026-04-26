"""Tests for PromptEvolver wiring to OperatorSelector and ImprovementLog.

Verifies US-504:
- Variant generation uses OperatorSelector + PromptMutator pipeline.
- A/B test completion propagates quality delta to OperatorSelector.
- Each operator application creates an ImprovementRecord in ImprovementLog.
- Existing A/B testing gates (p<0.05, Cohen's d >= 0.2) are preserved.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus
from vetinari.learning.operator_selector import OperatorSelector
from vetinari.learning.prompt_evolver import PromptEvolver
from vetinari.learning.prompt_mutator import MutationOperator, PromptMutator

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def state_dir(tmp_path):
    """Set VETINARI_STATE_DIR to a temp directory for test isolation."""
    state = tmp_path / ".vetinari"
    state.mkdir()
    os.environ["VETINARI_STATE_DIR"] = str(state)
    yield state
    os.environ.pop("VETINARI_STATE_DIR", None)


@pytest.fixture
def improvement_log(tmp_path):
    """Create a fresh ImprovementLog backed by a temp SQLite DB."""
    db_path = tmp_path / "kaizen_test.db"
    return ImprovementLog(db_path=db_path)


@pytest.fixture
def operator_selector(tmp_path):
    """Create a fresh OperatorSelector with temp state."""
    state_path = tmp_path / "operator_state.json"
    return OperatorSelector(state_path=state_path)


@pytest.fixture
def evolver(state_dir, improvement_log, operator_selector):
    """Create a PromptEvolver wired to test instances."""
    ev = PromptEvolver(improvement_log=improvement_log)
    ev._operator_selector = operator_selector
    ev._prompt_mutator = PromptMutator()
    return ev


# ── Variant generation uses operator pipeline ───────────────────────────


class TestVariantGenerationPipeline:
    """Verify that generate_variant uses OperatorSelector + PromptMutator."""

    def test_generate_variant_uses_operator_selector(self, evolver, operator_selector):
        """generate_variant should call OperatorSelector.select_operator."""
        baseline = "You are a helpful assistant. Be clear and concise."
        evolver.register_baseline("planner", baseline)

        with patch.object(operator_selector, "select_operator", wraps=operator_selector.select_operator) as mock_select:
            result = evolver.generate_variant("planner", baseline, mode="default")

        mock_select.assert_called_once_with("planner", "default")
        assert result is not None
        assert result != baseline

    def test_generate_variant_uses_prompt_mutator(self, evolver):
        """generate_variant should call PromptMutator.mutate with the selected operator."""
        baseline = "You are a helpful assistant. Be specific and thorough."
        evolver.register_baseline("builder", baseline)

        mutator = evolver._get_prompt_mutator()
        with patch.object(mutator, "mutate", wraps=mutator.mutate) as mock_mutate:
            result = evolver.generate_variant("builder", baseline, mode="build")

        mock_mutate.assert_called_once()
        call_args = mock_mutate.call_args
        assert call_args[0][0] == baseline
        assert isinstance(call_args[0][1], MutationOperator)
        assert result is not None

    def test_generate_variant_tracks_operator(self, evolver):
        """The operator used should be recorded in _variant_operators."""
        baseline = "You are a research agent. Analyze data carefully."
        evolver.register_baseline("researcher", baseline)

        result = evolver.generate_variant("researcher", baseline, mode="analysis")
        assert result is not None

        # Find the variant that was created
        variants = evolver._variants.get("researcher", [])
        testing_variants = [v for v in variants if v.status == "testing"]
        assert len(testing_variants) == 1

        variant_id = testing_variants[0].variant_id
        assert variant_id in evolver._variant_operators
        op, agent, mode = evolver._variant_operators[variant_id]
        assert isinstance(op, MutationOperator)
        assert agent == "researcher"
        assert mode == "analysis"

    def test_generate_variant_creates_prompt_variant(self, evolver):
        """A new PromptVariant should be appended to the variants list."""
        baseline = "You are a quality agent."
        evolver.register_baseline("quality", baseline)

        result = evolver.generate_variant("quality", baseline)
        assert result is not None

        variants = evolver._variants["quality"]
        # Baseline + one testing variant
        assert len(variants) == 2
        testing = [v for v in variants if v.status == "testing"]
        assert len(testing) == 1
        assert testing[0].prompt_text == result


# ── Operator feedback propagation ───────────────────────────────────────


class TestOperatorFeedbackPropagation:
    """Verify that A/B test results propagate to OperatorSelector.update."""

    def test_promotion_updates_operator_selector(self, evolver, operator_selector):
        """When a variant is promoted, OperatorSelector.update should be called."""
        baseline = "You are a planner agent. Create detailed plans."
        evolver.register_baseline("planner", baseline)

        # Generate a variant
        result = evolver.generate_variant("planner", baseline, mode="planning")
        assert result is not None

        # Find the variant
        testing = [v for v in evolver._variants["planner"] if v.status == "testing"]
        assert len(testing) == 1
        variant = testing[0]
        variant_id = variant.variant_id

        # Get the operator that produced this variant
        op, _, _ = evolver._variant_operators[variant_id]

        with patch.object(operator_selector, "update", wraps=operator_selector.update) as mock_update:
            # Simulate a positive quality delta feedback
            evolver._update_operator_feedback(variant_id, 0.15)

        mock_update.assert_called_once_with(op, "planner", "planning", 0.15)

    def test_deprecation_updates_operator_selector(self, evolver, operator_selector):
        """When a variant is deprecated, negative feedback goes to the selector."""
        baseline = "You are a builder agent."
        evolver.register_baseline("builder", baseline)

        result = evolver.generate_variant("builder", baseline, mode="build")
        assert isinstance(result, str)
        assert len(result) > 0

        testing = [v for v in evolver._variants["builder"] if v.status == "testing"]
        variant_id = testing[0].variant_id
        op, _, _ = evolver._variant_operators[variant_id]

        with patch.object(operator_selector, "update", wraps=operator_selector.update) as mock_update:
            evolver._update_operator_feedback(variant_id, -0.12)

        mock_update.assert_called_once_with(op, "builder", "build", -0.12)

    def test_legacy_variant_skips_operator_update(self, evolver, operator_selector):
        """Variants not produced by operators should not trigger selector update."""
        with patch.object(operator_selector, "update") as mock_update:
            # variant_id not in _variant_operators
            evolver._update_operator_feedback("nonexistent_variant", 0.1)

        mock_update.assert_not_called()


# ── ImprovementLog integration ──────────────────────────────────────────


class TestImprovementLogIntegration:
    """Verify that operator applications create ImprovementRecords."""

    def test_generate_variant_creates_improvement_record(self, evolver, improvement_log):
        """Each operator-generated variant should produce an ImprovementRecord."""
        baseline = "You are a helpful assistant. Be accurate."
        evolver.register_baseline("planner", baseline)

        result = evolver.generate_variant("planner", baseline, mode="default")
        assert result is not None

        # Find the variant
        testing = [v for v in evolver._variants["planner"] if v.status == "testing"]
        variant_id = testing[0].variant_id

        # Check that an improvement was created
        assert variant_id in evolver._variant_improvements
        improvement_id = evolver._variant_improvements[variant_id]

        record = improvement_log.get_improvement(improvement_id)
        assert record is not None
        assert record.metric == "prompt_quality"
        assert record.applied_by == "PromptEvolver"
        assert record.status == ImprovementStatus.ACTIVE
        assert "planner" in record.hypothesis
        assert variant_id in record.rollback_plan

    def test_improvement_records_observation_on_feedback(self, evolver, improvement_log):
        """When operator feedback is given, an observation should be recorded."""
        baseline = "You are a quality reviewer."
        evolver.register_baseline("quality", baseline)

        result = evolver.generate_variant("quality", baseline, mode="review")
        assert result is not None

        testing = [v for v in evolver._variants["quality"] if v.status == "testing"]
        variant_id = testing[0].variant_id
        improvement_id = evolver._variant_improvements[variant_id]

        # Trigger feedback
        evolver._update_operator_feedback(variant_id, 0.1)

        # Check that an observation was recorded
        observations = improvement_log.get_observations(improvement_id)
        assert len(observations) >= 1
        assert observations[0].metric_value > 0

    def test_no_improvement_log_does_not_crash(self, state_dir):
        """PromptEvolver with no ImprovementLog should still work."""
        ev = PromptEvolver(improvement_log=None)
        ev._prompt_mutator = PromptMutator()
        ev._operator_selector = OperatorSelector(state_path=state_dir / "op_state.json")

        baseline = "You are an agent."
        ev.register_baseline("test", baseline)

        result = ev.generate_variant("test", baseline)
        assert result is not None
        assert result != baseline


# ── A/B testing gates preserved ─────────────────────────────────────────


class TestABTestingGatesPreserved:
    """Verify that statistical significance and effect size gates still work."""

    def test_significance_thresholds_optimized(self):
        """The promotion thresholds should match the optimized values."""
        assert PromptEvolver.P_VALUE_THRESHOLD == 0.05
        assert PromptEvolver.MIN_EFFECT_SIZE == 0.3  # Optimized: "medium" effect for reliability
        assert PromptEvolver.MIN_IMPROVEMENT == 0.05
        assert PromptEvolver.MIN_TRIALS == 30  # Optimized: more samples for significance

    def test_insufficient_trials_blocks_promotion(self, evolver):
        """Variants with fewer than MIN_TRIALS should not be promoted."""
        baseline = "You are a planner."
        evolver.register_baseline("planner", baseline)

        result = evolver.generate_variant("planner", baseline)
        assert result is not None

        # Record fewer than MIN_TRIALS results
        testing = [v for v in evolver._variants["planner"] if v.status == "testing"]
        variant = testing[0]

        for _ in range(5):
            evolver.record_result("planner", variant.variant_id, 0.95)

        assert variant.status == "testing"
        assert variant.trials == 5

    def test_significance_test_runs_before_promotion(self, evolver):
        """_test_significance must be called before any promotion."""
        baseline = "You are a planner."
        evolver.register_baseline("planner", baseline)
        evolver.record_result("planner", "planner_baseline", 0.7)

        result = evolver.generate_variant("planner", baseline)
        assert result is not None

        testing = [v for v in evolver._variants["planner"] if v.status == "testing"]
        variant = testing[0]

        with patch.object(PromptEvolver, "_test_significance", return_value=(False, 0.1)) as mock_sig:
            # Record enough trials to exceed MIN_TRIALS (30) with high quality
            for _ in range(35):
                evolver.record_result("planner", "planner_baseline", 0.65)
                evolver.record_result("planner", variant.variant_id, 0.85)

        # _test_significance was called (may be called multiple times as
        # _check_promotion runs on every record_result)
        assert mock_sig.call_count > 0
        # With significance returning False, variant stays in testing
        assert variant.status == "testing"


# ── End-to-end integration ──────────────────────────────────────────────


class TestEndToEndWiring:
    """Full pipeline: generate → A/B test → promote → feedback → improvement."""

    def test_full_pipeline(self, evolver, operator_selector, improvement_log):
        """Walk through the complete lifecycle of a variant."""
        baseline = "You are a helpful assistant. Be concise."
        evolver.register_baseline("agent_x", baseline)

        # Step 1: Generate variant via operator pipeline
        mutated = evolver.generate_variant("agent_x", baseline, mode="default")
        assert mutated is not None
        assert mutated != baseline

        # Step 2: Verify operator tracking
        testing = [v for v in evolver._variants["agent_x"] if v.status == "testing"]
        assert len(testing) == 1
        variant_id = testing[0].variant_id
        assert variant_id in evolver._variant_operators

        # Step 3: Verify improvement record created
        assert variant_id in evolver._variant_improvements
        improvement_id = evolver._variant_improvements[variant_id]
        record = improvement_log.get_improvement(improvement_id)
        assert record is not None
        assert record.status == ImprovementStatus.ACTIVE

        # Step 4: Simulate feedback (positive delta)
        evolver._update_operator_feedback(variant_id, 0.08)

        # Step 5: Verify operator selector was updated
        op, _agent_type, _mode = evolver._variant_operators[variant_id]
        arm_stats = operator_selector.get_stats(agent_type="agent_x", mode="default")
        matching = [s for s in arm_stats if s["operator"] == op.value]
        assert len(matching) == 1
        assert matching[0]["pulls"] >= 1

        # Step 6: Verify improvement observation was recorded
        observations = improvement_log.get_observations(improvement_id)
        assert len(observations) >= 1
