"""Tests for the Kaizen PDCA Feedback Loop Controller."""

from __future__ import annotations

import json

import pytest

from vetinari.exceptions import ExecutionError
from vetinari.kaizen.defect_trends import build_hypothesis, is_valid_category
from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus
from vetinari.kaizen.pdca import (
    PDCAController,
    ThresholdApplicator,
    ThresholdOverride,
)
from vetinari.validation import DefectCategory


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "kaizen_pdca_test.db")


@pytest.fixture
def pdca(improvement_log, tmp_path):
    """Create a PDCAController with a threshold applicator for quality."""
    controller = PDCAController(
        improvement_log=improvement_log,
        overrides_path=tmp_path / "overrides.json",
    )
    applicator = ThresholdApplicator(initial_thresholds={"quality": 0.70})
    controller.register_applicator("quality", applicator)
    return controller


@pytest.fixture
def threshold_applicator():
    """Create a ThresholdApplicator with standard thresholds."""
    return ThresholdApplicator(
        initial_thresholds={
            "quality": 0.70,
            "latency": 200.0,
            "throughput": 50.0,
        },
    )


# ── ThresholdApplicator tests ────────────────────────────────────────────────


class TestThresholdApplicator:
    """Test the built-in threshold adjustment applicator."""

    def test_initial_thresholds(self, threshold_applicator):
        """Initial thresholds are accessible via the thresholds property."""
        thresholds = threshold_applicator.thresholds
        assert thresholds["quality"] == 0.70
        assert thresholds["latency"] == 200.0
        assert thresholds["throughput"] == 50.0

    def test_get_threshold(self, threshold_applicator):
        """get_threshold returns the value for a registered metric."""
        assert threshold_applicator.get_threshold("quality") == 0.70
        assert threshold_applicator.get_threshold("nonexistent") is None

    def test_apply_adjusts_threshold(self, threshold_applicator, improvement_log):
        """Calling the applicator adjusts the threshold to the target value."""
        imp_id = improvement_log.propose(
            hypothesis="Raise quality gate",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert to 0.70",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)

        changes = threshold_applicator(record)

        assert threshold_applicator.get_threshold("quality") == 0.80
        assert changes["metric"] == "quality"
        assert changes["previous"] == 0.70
        assert changes["new"] == 0.80
        assert changes["improvement_id"] == imp_id

    def test_apply_creates_override_record(self, threshold_applicator, improvement_log):
        """Applying a threshold creates a ThresholdOverride in history."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)

        threshold_applicator(record)

        overrides = threshold_applicator.overrides
        assert len(overrides) == 1
        assert overrides[0].metric == "quality"
        assert overrides[0].previous_value == 0.70
        assert overrides[0].new_value == 0.80
        assert not overrides[0].confirmed

    def test_confirm_override(self, threshold_applicator, improvement_log):
        """Confirming an override marks it as permanently applied."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)
        threshold_applicator(record)

        threshold_applicator.confirm_override(imp_id)

        assert threshold_applicator.overrides[0].confirmed is True

    def test_revert_override(self, threshold_applicator, improvement_log):
        """Reverting an override restores the previous threshold value."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)
        threshold_applicator(record)

        assert threshold_applicator.get_threshold("quality") == 0.80

        reverted_to = threshold_applicator.revert_override(imp_id)

        assert reverted_to == 0.70
        assert threshold_applicator.get_threshold("quality") == 0.70

    def test_revert_confirmed_does_nothing(self, threshold_applicator, improvement_log):
        """Reverting a confirmed override returns None (it's permanent)."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)
        threshold_applicator(record)
        threshold_applicator.confirm_override(imp_id)

        reverted_to = threshold_applicator.revert_override(imp_id)

        assert reverted_to is None
        assert threshold_applicator.get_threshold("quality") == 0.80

    def test_apply_unregistered_metric_creates_it(self, threshold_applicator, improvement_log):
        """Applying to an unregistered metric creates the threshold."""
        imp_id = improvement_log.propose(
            hypothesis="Set cost threshold",
            metric="cost",
            baseline=10.0,
            target=8.0,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        record = improvement_log.get_improvement(imp_id)

        threshold_applicator(record)

        assert threshold_applicator.get_threshold("cost") == 8.0


# ── PDCAController tests ────────────────────────────────────────────────────


class TestPDCAControllerActivateAndApply:
    """Test the Do phase: activate and apply improvements."""

    def test_activate_and_apply_adjusts_threshold(self, pdca, improvement_log):
        """activate_and_apply transitions to ACTIVE and applies the change."""
        imp_id = improvement_log.propose(
            hypothesis="Raise quality gate to 0.80",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert to 0.70",
        )

        changes = pdca.activate_and_apply(imp_id)

        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.ACTIVE
        assert changes["metric"] == "quality"
        assert changes["new"] == 0.80

    def test_activate_and_apply_without_applicator(self, improvement_log, tmp_path):
        """Activating a metric with no registered applicator still transitions status."""
        controller = PDCAController(
            improvement_log=improvement_log,
            overrides_path=tmp_path / "overrides.json",
        )
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="unknown_metric",
            baseline=1.0,
            target=2.0,
            applied_by="test",
            rollback_plan="Revert",
        )

        changes = controller.activate_and_apply(imp_id)

        assert changes == {}
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.ACTIVE

    def test_activate_nonexistent_raises(self, pdca):
        """Activating a nonexistent improvement raises ValueError."""
        with pytest.raises(ExecutionError, match="not found"):
            pdca.activate_and_apply("IMP-nonexistent")

    def test_activate_already_active_raises(self, pdca, improvement_log):
        """Activating an already-active improvement raises ValueError."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)

        with pytest.raises(ExecutionError, match="expected 'proposed'"):
            pdca.activate_and_apply(imp_id)


class TestPDCAControllerConfirmAndPersist:
    """Test the Act phase: confirm and persist improvements."""

    def test_confirm_and_persist_writes_overrides_file(self, pdca, improvement_log, tmp_path):
        """Confirmed improvements are persisted to the JSON overrides file."""
        imp_id = improvement_log.propose(
            hypothesis="Raise quality gate",
            metric="quality",
            baseline=0.70,
            target=0.85,
            applied_by="test",
            rollback_plan="Revert to 0.70",
        )
        pdca.activate_and_apply(imp_id)

        # Add observations and evaluate to CONFIRMED
        improvement_log.observe(imp_id, metric_value=0.88, sample_size=10)
        improvement_log.observe(imp_id, metric_value=0.90, sample_size=15)
        result = improvement_log.evaluate(imp_id)
        assert result == ImprovementStatus.CONFIRMED

        pdca.confirm_and_persist(imp_id)

        overrides_path = tmp_path / "overrides.json"
        assert overrides_path.exists()
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        assert len(overrides) == 1
        assert overrides[0]["improvement_id"] == imp_id
        assert overrides[0]["metric"] == "quality"
        assert overrides[0]["target_value"] == 0.85

    def test_confirm_non_confirmed_does_nothing(self, pdca, improvement_log, tmp_path):
        """Persisting an improvement that is not CONFIRMED is a no-op."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)

        pdca.confirm_and_persist(imp_id)

        overrides_path = tmp_path / "overrides.json"
        assert not overrides_path.exists()

    def test_load_persisted_overrides(self, pdca, improvement_log, tmp_path):
        """load_persisted_overrides reads back the persisted data."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.85,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)
        improvement_log.observe(imp_id, metric_value=0.90, sample_size=10)
        improvement_log.evaluate(imp_id)
        pdca.confirm_and_persist(imp_id)

        loaded = pdca.load_persisted_overrides()
        assert len(loaded) == 1
        assert loaded[0]["improvement_id"] == imp_id

    def test_load_empty_overrides(self, improvement_log, tmp_path):
        """Loading overrides from a nonexistent file returns empty list."""
        controller = PDCAController(
            improvement_log=improvement_log,
            overrides_path=tmp_path / "nonexistent.json",
        )
        assert controller.load_persisted_overrides() == []


class TestPDCAControllerTrendProposals:
    """Test the Plan phase: auto-proposing improvements from defect trends."""

    def test_worsening_trend_triggers_proposal(self, pdca, improvement_log):
        """Concerning defect trends auto-propose improvements."""
        weekly_counts = [
            {"hallucination": 5, "prompt": 3},
            {"hallucination": 5, "prompt": 3},
            {"hallucination": 10, "prompt": 3},  # hallucination doubled
        ]

        proposed_ids = pdca.check_trends_and_propose(weekly_counts=weekly_counts)

        assert len(proposed_ids) >= 1
        # Verify the proposed improvements exist in the log
        for imp_id in proposed_ids:
            record = improvement_log.get_improvement(imp_id)
            assert record is not None
            assert record.status == ImprovementStatus.PROPOSED
            assert record.applied_by == "pdca_trend_monitor"

    def test_stable_trends_propose_nothing(self, pdca):
        """Stable trends produce no proposals."""
        weekly_counts = [
            {"hallucination": 5, "prompt": 3},
            {"hallucination": 5, "prompt": 3},
            {"hallucination": 5, "prompt": 3},
        ]

        proposed_ids = pdca.check_trends_and_propose(weekly_counts=weekly_counts)
        assert proposed_ids == []

    def test_insufficient_data_proposes_nothing(self, pdca):
        """With fewer than 2 weeks of data, no proposals are made."""
        weekly_counts = [{"hallucination": 5}]
        proposed_ids = pdca.check_trends_and_propose(weekly_counts=weekly_counts)
        assert proposed_ids == []

    def test_empty_data_proposes_nothing(self, pdca):
        """Empty weekly counts produce no proposals."""
        proposed_ids = pdca.check_trends_and_propose(weekly_counts=[])
        assert proposed_ids == []

    def test_none_data_fetches_from_log(self, pdca):
        """Passing None for weekly_counts fetches from the ImprovementLog."""
        # No defects recorded, so should return empty
        proposed_ids = pdca.check_trends_and_propose(weekly_counts=None)
        assert proposed_ids == []

    def test_multiple_concerning_trends(self, pdca, improvement_log):
        """Multiple concerning categories each produce a proposal."""
        weekly_counts = [
            {"hallucination": 2, "prompt": 2, "bad_spec": 2},
            {"hallucination": 5, "prompt": 5, "bad_spec": 5},  # all increased >100%
        ]

        proposed_ids = pdca.check_trends_and_propose(weekly_counts=weekly_counts)

        # At least hallucination, prompt, and bad_spec should be proposed
        assert len(proposed_ids) >= 3


class TestPDCAControllerRunCheckPhase:
    """Test the full Check-Act bridge: evaluate active and persist confirmed."""

    def test_run_check_confirms_and_persists(self, pdca, improvement_log):
        """run_check_phase evaluates active improvements and persists confirmed ones."""
        imp_id = improvement_log.propose(
            hypothesis="Test improvement",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)
        improvement_log.observe(imp_id, metric_value=0.85, sample_size=10)
        improvement_log.observe(imp_id, metric_value=0.88, sample_size=15)

        confirmed_ids = pdca.run_check_phase()

        assert imp_id in confirmed_ids
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.CONFIRMED

    def test_run_check_reverts_failed(self, pdca, improvement_log):
        """run_check_phase reverts threshold for failed improvements."""
        imp_id = improvement_log.propose(
            hypothesis="Test failing improvement",
            metric="quality",
            baseline=0.70,
            target=0.90,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)
        # Observations worse than baseline * 0.95 = 0.665
        improvement_log.observe(imp_id, metric_value=0.50, sample_size=10)
        improvement_log.observe(imp_id, metric_value=0.55, sample_size=15)

        confirmed_ids = pdca.run_check_phase()

        assert confirmed_ids == []
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.FAILED

    def test_run_check_skips_no_observations(self, pdca, improvement_log):
        """Active improvements with no observations are skipped."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.70,
            target=0.80,
            applied_by="test",
            rollback_plan="Revert",
        )
        pdca.activate_and_apply(imp_id)

        confirmed_ids = pdca.run_check_phase()

        assert confirmed_ids == []
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.ACTIVE


# ── Helper function tests ────────────────────────────────────────────────────


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_is_valid_category_valid(self):
        """Valid DefectCategory values return True."""
        assert is_valid_category("hallucination") is True
        assert is_valid_category("bad_spec") is True
        assert is_valid_category("prompt") is True

    def test_is_valid_category_invalid(self):
        """Invalid category strings return False."""
        assert is_valid_category("not_a_category") is False
        assert is_valid_category("") is False

    @pytest.mark.parametrize(
        ("category", "expected_fragment"),
        [
            (DefectCategory.HALLUCINATION, "Hallucination rate increased"),
            (DefectCategory.BAD_SPEC, "Bad spec rate increased"),
            (DefectCategory.PROMPT_WEAKNESS, "Prompt weakness increased"),
            (DefectCategory.WRONG_MODEL, "Wrong model rate increased"),
            (DefectCategory.INSUFFICIENT_CONTEXT, "Context gaps increased"),
            (DefectCategory.INTEGRATION_ERROR, "Integration errors increased"),
            (DefectCategory.COMPLEXITY_UNDERESTIMATE, "Complexity underestimates increased"),
        ],
    )
    def testbuild_hypothesis_all_categories(self, category, expected_fragment):
        """Each DefectCategory produces a meaningful hypothesis string."""
        hypothesis = build_hypothesis(category, 0.25)
        assert expected_fragment in hypothesis
        assert "25%" in hypothesis
