"""Tests for the Kaizen Improvement Log — PDCA lifecycle tracking."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from vetinari.exceptions import ExecutionError
from vetinari.kaizen.improvement_log import (
    ImprovementLog,
    ImprovementRecord,
    ImprovementStatus,
    KaizenReport,
)


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "kaizen_test.db")


class TestImprovementLogLifecycle:
    """Test the full PDCA lifecycle: propose → activate → observe → evaluate."""

    def test_propose_creates_improvement(self, improvement_log):
        """Proposing an improvement returns an ID and creates a PROPOSED record."""
        imp_id = improvement_log.propose(
            hypothesis="Reducing context by 20% will cut latency 15%",
            metric="latency",
            baseline=200.0,
            target=170.0,
            applied_by="auto_tuner",
            rollback_plan="Revert context window to 4096",
        )
        assert imp_id.startswith("IMP-")

        record = improvement_log.get_improvement(imp_id)
        assert record is not None
        assert record.status == ImprovementStatus.PROPOSED
        assert record.hypothesis == "Reducing context by 20% will cut latency 15%"
        assert record.metric == "latency"
        assert record.baseline_value == 200.0
        assert record.target_value == 170.0
        assert record.applied_by == "auto_tuner"
        assert record.actual_value is None

    def test_activate_transitions_to_active(self, improvement_log):
        """Activating a proposed improvement sets status to ACTIVE and records applied_at."""
        imp_id = improvement_log.propose(
            hypothesis="Test hypothesis",
            metric="quality",
            baseline=0.7,
            target=0.8,
            applied_by="prompt_evolver",
            rollback_plan="Revert to v1",
        )
        improvement_log.activate(imp_id)

        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.ACTIVE
        assert record.applied_at is not None
        assert isinstance(record.applied_at, (str, datetime))

    def test_activate_non_proposed_raises(self, improvement_log):
        """Activating an already-active improvement raises ValueError."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.7,
            target=0.8,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        with pytest.raises(ExecutionError, match="expected 'proposed'"):
            improvement_log.activate(imp_id)

    def test_activate_nonexistent_raises(self, improvement_log):
        """Activating a nonexistent improvement raises ValueError."""
        with pytest.raises(ExecutionError, match="not found"):
            improvement_log.activate("IMP-nonexistent")

    def test_observe_records_observation(self, improvement_log):
        """Observations are recorded and retrievable."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.7,
            target=0.8,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.observe(imp_id, metric_value=0.75, sample_size=10)
        improvement_log.observe(imp_id, metric_value=0.82, sample_size=15)

        observations = improvement_log.get_observations(imp_id)
        assert len(observations) == 2
        assert observations[0].metric_value == 0.75
        assert observations[1].metric_value == 0.82

    def test_observe_nonexistent_raises(self, improvement_log):
        """Observing a nonexistent improvement raises ValueError."""
        with pytest.raises(ExecutionError, match="not found"):
            improvement_log.observe("IMP-nonexistent", 0.5, 10)


class TestEvaluate:
    """Test evaluation logic: CONFIRMED, FAILED, stays ACTIVE."""

    def test_evaluate_confirmed_when_meets_target(self, improvement_log):
        """Improvement is CONFIRMED when actual >= target."""
        imp_id = improvement_log.propose(
            hypothesis="Quality will improve",
            metric="quality",
            baseline=0.7,
            target=0.8,
            applied_by="prompt_evolver",
            rollback_plan="Revert prompt",
        )
        improvement_log.activate(imp_id)
        improvement_log.observe(imp_id, 0.82, 20)
        improvement_log.observe(imp_id, 0.85, 25)

        status = improvement_log.evaluate(imp_id)
        assert status == ImprovementStatus.CONFIRMED

        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.CONFIRMED
        assert record.actual_value is not None
        assert record.actual_value >= 0.8
        assert record.confirmed_at is not None
        assert isinstance(record.confirmed_at, (str, datetime))

    def test_evaluate_failed_when_below_baseline(self, improvement_log):
        """Improvement is FAILED when actual < baseline * 0.95."""
        imp_id = improvement_log.propose(
            hypothesis="Quality will improve with new prompt",
            metric="quality",
            baseline=0.70,
            target=0.85,
            applied_by="prompt_evolver",
            rollback_plan="Revert prompt",
        )
        improvement_log.activate(imp_id)
        # Observations worse than baseline * 0.95 = 0.665
        improvement_log.observe(imp_id, 0.60, 10)
        improvement_log.observe(imp_id, 0.62, 10)

        status = improvement_log.evaluate(imp_id)
        # mean = 0.61, which is < 0.70 * 0.95 = 0.665 → FAILED
        assert status == ImprovementStatus.FAILED

    def test_evaluate_stays_active_in_middle(self, improvement_log):
        """Improvement stays ACTIVE when actual is between baseline*0.95 and target."""
        imp_id = improvement_log.propose(
            hypothesis="Quality will improve",
            metric="quality",
            baseline=0.7,
            target=0.85,
            applied_by="workflow_learner",
            rollback_plan="Revert strategy",
        )
        improvement_log.activate(imp_id)
        # Mean = 0.77, which is >= 0.7*0.95=0.665 but < 0.85 → stays ACTIVE
        improvement_log.observe(imp_id, 0.76, 15)
        improvement_log.observe(imp_id, 0.78, 15)

        status = improvement_log.evaluate(imp_id)
        assert status == ImprovementStatus.ACTIVE

    def test_evaluate_no_observations_raises(self, improvement_log):
        """Evaluating with no observations raises ValueError."""
        imp_id = improvement_log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.7,
            target=0.8,
            applied_by="test",
            rollback_plan="Revert",
        )
        improvement_log.activate(imp_id)
        with pytest.raises(ExecutionError, match="No observations"):
            improvement_log.evaluate(imp_id)


class TestQueryMethods:
    """Test query and reporting methods."""

    def test_get_active_improvements(self, improvement_log):
        """get_active_improvements returns only ACTIVE records."""
        id1 = improvement_log.propose(
            hypothesis="H1",
            metric="q",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.propose(
            hypothesis="H2",
            metric="q",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(id1)
        # id2 stays PROPOSED

        active = improvement_log.get_active_improvements()
        assert len(active) == 1
        assert active[0].id == id1

    def test_get_confirmed_improvements(self, improvement_log):
        """get_confirmed_improvements returns only CONFIRMED records."""
        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="q",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.observe(imp_id, 0.75, 10)
        improvement_log.evaluate(imp_id)

        confirmed = improvement_log.get_confirmed_improvements()
        assert len(confirmed) == 1
        assert confirmed[0].status == ImprovementStatus.CONFIRMED

    def test_get_weekly_report(self, improvement_log):
        """get_weekly_report returns a KaizenReport with correct counts."""
        # Create improvements in various statuses
        id1 = improvement_log.propose(
            hypothesis="H1",
            metric="q",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.propose(
            hypothesis="H2",
            metric="q",
            baseline=0.6,
            target=0.8,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(id1)
        improvement_log.observe(id1, 0.75, 10)
        improvement_log.evaluate(id1)  # CONFIRMED

        report = improvement_log.get_weekly_report()
        assert isinstance(report, KaizenReport)
        assert report.total_confirmed == 1
        assert report.total_proposed == 1
        assert report.avg_improvement_effect > 0  # 0.75 - 0.5 = 0.25

    def test_revert_sets_status(self, improvement_log):
        """Reverting an improvement sets status to REVERTED and regression_detected."""
        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="q",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.revert(imp_id)

        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.REVERTED
        assert record.regression_detected is True
        assert record.reverted_at is not None
        assert isinstance(record.reverted_at, (str, datetime))

    def test_get_nonexistent_improvement_returns_none(self, improvement_log):
        """get_improvement returns None for a nonexistent ID."""
        assert improvement_log.get_improvement("IMP-notreal") is None
