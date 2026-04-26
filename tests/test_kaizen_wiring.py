"""Tests for the kaizen wiring module (scheduled entry points)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.kaizen.defect_trends import DefectHotspot, DefectTrendReport
from vetinari.kaizen.improvement_log import ImprovementLog, ImprovementStatus
from vetinari.kaizen.pdca import PDCAController
from vetinari.kaizen.regression import RegressionAlert
from vetinari.kaizen.wiring import (
    _build_hotspots,
    _build_weekly_counts,
    scheduled_pdca_check,
    scheduled_regression_check,
    scheduled_trend_analysis,
    wire_kaizen_subsystem,
)
from vetinari.validation import DefectCategory

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """Temporary SQLite database path for isolation."""
    return str(tmp_path / "kaizen_wiring_test.db")


# ── _build_weekly_counts ──────────────────────────────────────────────────────


class TestBuildWeeklyCounts:
    def test_converts_valid_category_strings(self):
        raw = [{"bad_spec": 3, "prompt": 2}, {"bad_spec": 1, "hallucination": 5}]
        result = _build_weekly_counts(raw)
        assert len(result) == 2
        assert result[0][DefectCategory.BAD_SPEC] == 3
        assert result[0][DefectCategory.PROMPT_WEAKNESS] == 2
        assert result[1][DefectCategory.HALLUCINATION] == 5

    def test_drops_unknown_category_strings(self):
        raw = [{"bad_spec": 2, "totally_unknown_category": 99}]
        result = _build_weekly_counts(raw)
        assert len(result) == 1
        assert DefectCategory.BAD_SPEC in result[0]
        assert len(result[0]) == 1  # unknown category dropped

    def test_empty_input_returns_empty_list(self):
        assert _build_weekly_counts([]) == []

    def test_empty_week_dicts_become_empty_dicts(self):
        result = _build_weekly_counts([{}, {}])
        assert result == [{}, {}]


# ── _build_hotspots ───────────────────────────────────────────────────────────


class TestBuildHotspots:
    def test_converts_valid_hotspot_dict(self):
        raw = [
            {
                "agent_type": "worker",
                "mode": "code",
                "category": "prompt",
                "count": 7,
            }
        ]
        result = _build_hotspots(raw)
        assert len(result) == 1
        hs = result[0]
        assert isinstance(hs, DefectHotspot)
        assert hs.agent_type == "worker"
        assert hs.defect_category == DefectCategory.PROMPT_WEAKNESS
        assert hs.defect_count == 7

    def test_drops_hotspot_with_unknown_category(self):
        raw = [{"agent_type": "worker", "mode": "code", "category": "no_such", "count": 3}]
        result = _build_hotspots(raw)
        assert result == []

    def test_empty_input_returns_empty_list(self):
        assert _build_hotspots([]) == []

    def test_defect_rate_is_one_for_single_occurrence(self):
        raw = [{"agent_type": "foreman", "mode": "plan", "category": "bad_spec", "count": 1}]
        hs = _build_hotspots(raw)[0]
        assert hs.defect_rate == 1.0


# ── scheduled_pdca_check ─────────────────────────────────────────────────────


class TestScheduledPdcaCheck:
    def test_returns_list_of_proposed_ids(self, db_path):
        with patch(
            "vetinari.kaizen.wiring.PDCAController.check_trends_and_propose",
            return_value=["imp-1", "imp-2"],
        ):
            result = scheduled_pdca_check(db_path)
        assert result == ["imp-1", "imp-2"]

    def test_returns_empty_list_when_no_proposals(self, db_path):
        with patch(
            "vetinari.kaizen.wiring.PDCAController.check_trends_and_propose",
            return_value=[],
        ):
            result = scheduled_pdca_check(db_path)
        assert result == []

    def test_worsening_trend_creates_proposed_improvement_in_db(self, db_path):
        """A worsening defect trend must produce a real improvement in the DB.

        This test proves metric semantics: when weekly_counts show a >15%
        increase (hallucination: 10 → 12 = +20%), check_trends_and_propose()
        must write a PROPOSED improvement to the ImprovementLog, not merely
        return an ID that was fabricated by a mock.
        """
        improvement_log = ImprovementLog(db_path)
        controller = PDCAController(improvement_log)

        # Two weeks of data: hallucination increases 20% (10 → 12), above the 15% threshold.
        worsening_counts = [
            {"hallucination": 10},
            {"hallucination": 12},
        ]
        proposed_ids = controller.check_trends_and_propose(weekly_counts=worsening_counts)

        assert len(proposed_ids) >= 1, (
            "Expected at least one improvement proposed for a worsening trend, "
            f"but got: {proposed_ids!r}"
        )
        # Verify each returned ID actually exists in the DB with PROPOSED status.
        for imp_id in proposed_ids:
            record = improvement_log.get_improvement(imp_id)
            assert record is not None, (
                f"Improvement ID {imp_id!r} returned by check_trends_and_propose "
                "does not exist in the ImprovementLog DB"
            )
            assert record.status == ImprovementStatus.PROPOSED, (
                f"Expected PROPOSED status for new improvement {imp_id!r}, "
                f"got {record.status!r}"
            )
            assert record.metric == "defect_count", (
                f"Expected metric='defect_count' for PDCA-proposed improvement, "
                f"got {record.metric!r}"
            )


# ── scheduled_regression_check ───────────────────────────────────────────────


class TestScheduledRegressionCheck:
    def test_returns_list_of_alerts(self, db_path):
        alert = RegressionAlert(
            improvement_id="imp-1",
            metric="quality",
            expected=0.9,
            actual=0.75,
            degradation_pct=0.167,
            severity="warning",
        )
        with patch(
            "vetinari.kaizen.wiring.RegressionDetector.check_regressions",
            return_value=[alert],
        ):
            result = scheduled_regression_check(db_path)
        assert len(result) == 1
        assert result[0].improvement_id == "imp-1"

    def test_returns_empty_list_when_no_regressions(self, db_path):
        with patch(
            "vetinari.kaizen.wiring.RegressionDetector.check_regressions",
            return_value=[],
        ):
            result = scheduled_regression_check(db_path)
        assert result == []

    def test_with_empty_db_returns_no_alerts(self, db_path):
        result = scheduled_regression_check(db_path)
        assert isinstance(result, list)
        assert result == []


# ── scheduled_trend_analysis ─────────────────────────────────────────────────


class TestScheduledTrendAnalysis:
    def test_returns_defect_trend_report(self, db_path):
        report = DefectTrendReport(recommendations=["Fix prompt templates"])
        with patch(
            "vetinari.kaizen.wiring.DefectTrendAnalyzer.analyze_trends",
            return_value=report,
        ):
            result = scheduled_trend_analysis(db_path)
        assert isinstance(result, DefectTrendReport)
        assert result.recommendations == ["Fix prompt templates"]

    def test_with_empty_db_returns_empty_report(self, db_path):
        result = scheduled_trend_analysis(db_path)
        assert isinstance(result, DefectTrendReport)
        assert result.trends == {}
        assert result.recommendations == []


# ── wire_kaizen_subsystem ─────────────────────────────────────────────────────


class TestWireKaizenSubsystem:
    def test_runs_without_error(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="vetinari.kaizen.wiring"):
            wire_kaizen_subsystem()
        # Wiring emits at least one INFO record when it completes successfully.
        assert len(caplog.records) >= 1

    def test_logs_ready_message(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="vetinari.kaizen.wiring"):
            wire_kaizen_subsystem()
        assert any("ready" in record.message for record in caplog.records)
