"""Tests for the Improvement Aggregator — weekly kaizen reports."""

from __future__ import annotations

import pytest

from vetinari.kaizen.aggregator import ImprovementAggregator, KaizenWeeklyReport
from vetinari.kaizen.gemba import AutoGembaWalk
from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "aggregator_test.db")


@pytest.fixture
def aggregator(improvement_log):
    """Create an ImprovementAggregator instance."""
    return ImprovementAggregator(improvement_log)


class TestImprovementAggregator:
    """Test weekly report generation and recommendations."""

    def test_empty_report(self, aggregator):
        """Empty improvement log produces a valid empty report."""
        report = aggregator.generate_weekly_report()
        assert isinstance(report, KaizenWeeklyReport)
        assert report.improvement_velocity == 0.0
        assert len(report.top_improvements) == 0
        assert len(report.regressions) == 0

    def test_report_with_confirmed_improvements(self, improvement_log, aggregator):
        """Report includes confirmed improvements as top improvements."""
        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="prompt_evolver",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)
        improvement_log.observe(imp_id, 0.75, 10)
        improvement_log.evaluate(imp_id)

        report = aggregator.generate_weekly_report()
        assert len(report.top_improvements) == 1
        assert report.improvement_velocity > 0

    def test_report_includes_active_experiments(self, improvement_log, aggregator):
        """Report includes currently active experiments."""
        imp_id = improvement_log.propose(
            hypothesis="H1",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        improvement_log.activate(imp_id)

        report = aggregator.generate_weekly_report()
        assert len(report.active_experiments) == 1

    def test_report_includes_open_hypotheses(self, improvement_log, aggregator):
        """Report includes proposed but not yet activated improvements."""
        improvement_log.propose(
            hypothesis="H1",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )
        report = aggregator.generate_weekly_report()
        assert len(report.open_hypotheses) == 1

    def test_recommendation_no_experiments(self, improvement_log):
        """Recommendation generated when no active experiments or proposals."""
        agg = ImprovementAggregator(improvement_log)
        report = agg.generate_weekly_report()
        assert any("no active" in r.lower() for r in report.recommendations)

    def test_recommendation_more_reverted_than_confirmed(self, improvement_log):
        """Recommendation when more improvements reverted than confirmed."""
        # Create and revert two improvements
        for _ in range(2):
            imp_id = improvement_log.propose(
                hypothesis="H",
                metric="q",
                baseline=0.5,
                target=0.7,
                applied_by="test",
                rollback_plan="revert",
            )
            improvement_log.activate(imp_id)
            improvement_log.revert(imp_id)

        agg = ImprovementAggregator(improvement_log)
        report = agg.generate_weekly_report()
        assert any("reverted" in r.lower() for r in report.recommendations)

    def test_gemba_findings_included(self, improvement_log):
        """Gemba findings from latest walk are included in report."""
        gemba = AutoGembaWalk(improvement_log)
        gemba.run(
            failure_patterns=[
                {"model": "phi-3", "task_type": "coding", "frequency": 5, "avg_score": 0.3},
            ],
        )
        agg = ImprovementAggregator(improvement_log, gemba=gemba)
        report = agg.generate_weekly_report()
        assert len(report.gemba_findings) == 1

    def test_reverted_improvement_with_regression_appears_in_regressions(self, improvement_log):
        """Reverted improvements flagged with regression_detected=True appear in report.regressions.

        Reverted improvements that caused a regression must appear in the weekly
        regressions list even though they are no longer active — they were auto-reverted
        precisely because a regression was detected, so the report must surface them.
        """
        imp_id = improvement_log.propose(
            hypothesis="Reduce learning rate to improve stability",
            metric="quality_score",
            baseline=0.75,
            target=0.85,
            applied_by="auto_tuner",
            rollback_plan="restore lr to 1e-4",
        )
        improvement_log.activate(imp_id)
        improvement_log.revert(imp_id)  # revert() sets regression_detected=True

        agg = ImprovementAggregator(improvement_log)
        report = agg.generate_weekly_report()

        regression_ids = [r.id for r in report.regressions]
        assert imp_id in regression_ids, (
            f"Reverted improvement {imp_id!r} with regression_detected=True must appear "
            f"in report.regressions, but got: {regression_ids!r}"
        )
