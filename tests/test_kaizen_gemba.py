"""Tests for the Auto-Gemba Walk — systematic execution review."""

from __future__ import annotations

import pytest

from vetinari.kaizen.gemba import AutoGembaWalk, GembaFinding, GembaReport
from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus


@pytest.fixture
def improvement_log(tmp_path):
    """Create an ImprovementLog backed by a temporary SQLite database."""
    return ImprovementLog(tmp_path / "gemba_test.db")


@pytest.fixture
def gemba(improvement_log):
    """Create an AutoGembaWalk instance."""
    return AutoGembaWalk(improvement_log)


class TestAutoGembaWalk:
    """Test the gemba walk's ability to surface findings and propose improvements."""

    def test_empty_run_produces_empty_report(self, gemba):
        """Running with no data produces a report with no findings."""
        report = gemba.run()
        assert isinstance(report, GembaReport)
        assert len(report.findings) == 0
        assert report.improvements_proposed == 0

    def test_recurring_failures_detected(self, gemba, improvement_log):
        """Recurring failure patterns generate findings and proposed improvements."""
        patterns = [
            {"model": "phi-3", "task_type": "coding", "frequency": 5, "avg_score": 0.3},
            {"model": "llama-3", "task_type": "review", "frequency": 1, "avg_score": 0.8},
        ]
        report = gemba.run(failure_patterns=patterns)

        assert len(report.findings) == 1
        assert report.findings[0].type == "recurring_failure"
        assert "phi-3" in report.findings[0].detail
        assert report.improvements_proposed == 1

        # Verify improvement was proposed in the log
        proposed = improvement_log.get_improvements_by_status(ImprovementStatus.PROPOSED)
        assert len(proposed) == 1
        assert proposed[0].applied_by == "gemba_walk"

    def test_high_rework_rate_detected(self, gemba):
        """High rework rate (>20%) generates a finding."""
        report = gemba.run(rework_stats={"rework_count": 30, "total_tasks": 100})

        assert len(report.findings) == 1
        assert report.findings[0].type == "high_rework"
        assert "30%" in report.findings[0].detail

    def test_normal_rework_rate_no_finding(self, gemba):
        """Normal rework rate (<=20%) produces no finding."""
        report = gemba.run(rework_stats={"rework_count": 10, "total_tasks": 100})
        assert len(report.findings) == 0

    def test_low_value_add_detected(self, gemba):
        """Low value-add ratio (<50%) generates a finding."""
        report = gemba.run(
            value_stream_stats={
                "value_add_ratio": 0.35,
                "avg_lead_time": 120.0,
                "bottleneck_station": "Builder",
            }
        )
        assert len(report.findings) == 1
        assert report.findings[0].type == "low_value_add"
        assert "Builder" in report.findings[0].proposed_improvement

    def test_dominant_defect_cause_detected(self, gemba):
        """Dominant defect cause (>40%) generates a finding."""
        report = gemba.run(
            defect_distribution={
                "hallucination": 0.50,
                "bad_spec": 0.20,
                "prompt": 0.30,
            }
        )
        assert len(report.findings) == 1
        assert report.findings[0].type == "dominant_defect_cause"
        assert "hallucination" in report.findings[0].detail

    def test_no_dominant_cause_no_finding(self, gemba):
        """Balanced defect distribution produces no finding."""
        report = gemba.run(
            defect_distribution={
                "hallucination": 0.25,
                "bad_spec": 0.25,
                "prompt": 0.25,
                "context": 0.25,
            }
        )
        assert len(report.findings) == 0

    def test_multiple_findings_in_one_walk(self, gemba):
        """Multiple issue types can be found in a single walk."""
        report = gemba.run(
            failure_patterns=[
                {"model": "phi-3", "task_type": "coding", "frequency": 5, "avg_score": 0.3},
            ],
            rework_stats={"rework_count": 30, "total_tasks": 100},
        )
        assert len(report.findings) == 2
        assert report.improvements_proposed == 2

    def test_latest_report_stored(self, gemba):
        """The latest report is accessible via the latest_report attribute."""
        assert gemba.latest_report is None
        report = gemba.run()
        assert gemba.latest_report is report
