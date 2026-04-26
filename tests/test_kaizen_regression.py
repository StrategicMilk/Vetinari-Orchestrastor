"""Tests for the Regression Detector — monitoring confirmed improvements."""

from __future__ import annotations

import pytest

from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus
from vetinari.kaizen.regression import RegressionAlert, RegressionDetector


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "regression_test.db")


@pytest.fixture
def detector(improvement_log):
    """Create a RegressionDetector instance."""
    return RegressionDetector(improvement_log)


def _create_confirmed_improvement(log, baseline=0.5, actual=0.75):
    """Helper: create a confirmed improvement with the given values."""
    imp_id = log.propose(
        hypothesis="Test improvement",
        metric="quality",
        baseline=baseline,
        target=baseline + 0.2,
        applied_by="test",
        rollback_plan="revert to previous",
    )
    log.activate(imp_id)
    log.observe(imp_id, actual, 20)
    log.evaluate(imp_id)
    return imp_id


class TestRegressionDetector:
    """Test regression detection and auto-revert."""

    def test_no_regression_when_stable(self, improvement_log, detector):
        """No alerts when recent observations match post-improvement level."""
        imp_id = _create_confirmed_improvement(improvement_log, baseline=0.5, actual=0.75)

        # Add recent observations at the same level
        improvement_log.observe(imp_id, 0.74, 10)
        improvement_log.observe(imp_id, 0.76, 10)

        alerts = detector.check_regressions()
        assert len(alerts) == 0

    def test_warning_regression_detected(self, improvement_log, detector):
        """Warning alert when metric degrades 10%+ but still above baseline."""
        imp_id = _create_confirmed_improvement(improvement_log, baseline=0.5, actual=0.75)

        # Add degraded observations (10%+ below 0.75 = below 0.675, but above 0.5)
        improvement_log.observe(imp_id, 0.60, 10)
        improvement_log.observe(imp_id, 0.62, 10)

        alerts = detector.check_regressions()
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].improvement_id == imp_id
        assert alerts[0].degradation_pct > 0.1

        # Should NOT auto-revert for warning
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.CONFIRMED

    def test_critical_regression_auto_reverts(self, improvement_log, detector):
        """Critical regression (below pre-improvement baseline) triggers auto-revert."""
        imp_id = _create_confirmed_improvement(improvement_log, baseline=0.5, actual=0.75)

        # Add many observations well below the original baseline (0.5)
        # The evaluate() left one observation at 0.75 in the DB, so we need enough
        # low observations to bring the mean below 0.5
        for _ in range(5):
            improvement_log.observe(imp_id, 0.30, 10)

        alerts = detector.check_regressions()
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

        # Should auto-revert
        record = improvement_log.get_improvement(imp_id)
        assert record.status == ImprovementStatus.REVERTED
        assert record.regression_detected is True

    def test_no_observations_no_alert(self, improvement_log, detector):
        """No alert when a confirmed improvement has no recent observations."""
        _create_confirmed_improvement(improvement_log)
        # No additional observations added after confirmation
        # The evaluate() observation is the only one, and it's within 7 days
        # so it will be checked — but the average should match actual_value
        alerts = detector.check_regressions()
        assert len(alerts) == 0

    def test_multiple_improvements_checked(self, improvement_log, detector):
        """All confirmed improvements are checked in a single run."""
        id1 = _create_confirmed_improvement(improvement_log, baseline=0.5, actual=0.75)
        id2 = _create_confirmed_improvement(improvement_log, baseline=0.6, actual=0.85)

        # Degrade first, keep second stable
        improvement_log.observe(id1, 0.40, 10)
        improvement_log.observe(id2, 0.84, 10)

        alerts = detector.check_regressions()
        assert len(alerts) == 1
        assert alerts[0].improvement_id == id1
