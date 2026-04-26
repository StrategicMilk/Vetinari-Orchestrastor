"""Tests for the Defect Trend Analyzer — defect category tracking."""

from __future__ import annotations

import pytest

from vetinari.kaizen.defect_trends import (
    DefectHotspot,
    DefectTrend,
    DefectTrendAnalyzer,
    DefectTrendReport,
)
from vetinari.types import AgentType
from vetinari.validation import DefectCategory


@pytest.fixture
def analyzer():
    """Create a DefectTrendAnalyzer instance."""
    return DefectTrendAnalyzer()


class TestDefectTrendAnalyzer:
    """Test defect trend analysis and recommendations."""

    def test_increasing_trend_detected(self, analyzer):
        """A >10% week-over-week increase is classified as 'increasing'."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 5},
            {DefectCategory.HALLUCINATION: 8},  # 60% increase
        ]
        report = analyzer.analyze_trends(weekly_counts)

        trend = report.trends[DefectCategory.HALLUCINATION]
        assert trend.trend == "increasing"
        assert trend.current_count == 8
        assert trend.previous_count == 5
        assert trend.change_pct == pytest.approx(0.6)

    def test_decreasing_trend_detected(self, analyzer):
        """A >10% decrease is classified as 'decreasing'."""
        weekly_counts = [
            {DefectCategory.BAD_SPEC: 10},
            {DefectCategory.BAD_SPEC: 5},  # 50% decrease
        ]
        report = analyzer.analyze_trends(weekly_counts)

        trend = report.trends[DefectCategory.BAD_SPEC]
        assert trend.trend == "decreasing"
        assert trend.change_pct == pytest.approx(-0.5)

    def test_stable_trend_detected(self, analyzer):
        """A change within ±10% is classified as 'stable'."""
        weekly_counts = [
            {DefectCategory.PROMPT_WEAKNESS: 10},
            {DefectCategory.PROMPT_WEAKNESS: 10},  # 0% change
        ]
        report = analyzer.analyze_trends(weekly_counts)

        trend = report.trends[DefectCategory.PROMPT_WEAKNESS]
        assert trend.trend == "stable"
        assert not trend.is_concerning

    def test_concerning_threshold(self, analyzer):
        """A >15% increase is flagged as concerning."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 10},
            {DefectCategory.HALLUCINATION: 12},  # 20% increase > 15%
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert report.trends[DefectCategory.HALLUCINATION].is_concerning is True

    def test_not_concerning_below_threshold(self, analyzer):
        """A <=15% increase is not flagged as concerning."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 10},
            {DefectCategory.HALLUCINATION: 11},  # 10% increase <= 15%
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert report.trends[DefectCategory.HALLUCINATION].is_concerning is False

    def test_top_defect_category(self, analyzer):
        """Top defect category is the one with highest current count."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 5, DefectCategory.BAD_SPEC: 10},
            {DefectCategory.HALLUCINATION: 3, DefectCategory.BAD_SPEC: 12},
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert report.top_defect_category == DefectCategory.BAD_SPEC

    def test_recommendation_for_hallucination(self, analyzer):
        """Concerning hallucination trend generates specific recommendation."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 5},
            {DefectCategory.HALLUCINATION: 10},  # 100% increase
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert any("hallucination" in r.lower() for r in report.recommendations)
        assert any("verification" in r.lower() for r in report.recommendations)

    def test_recommendation_for_bad_spec(self, analyzer):
        """Concerning bad spec trend generates specific recommendation."""
        weekly_counts = [
            {DefectCategory.BAD_SPEC: 5},
            {DefectCategory.BAD_SPEC: 10},
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert any("spec" in r.lower() for r in report.recommendations)

    def test_recommendation_for_prompt_weakness(self, analyzer):
        """Concerning prompt weakness trend generates specific recommendation."""
        weekly_counts = [
            {DefectCategory.PROMPT_WEAKNESS: 5},
            {DefectCategory.PROMPT_WEAKNESS: 10},
        ]
        report = analyzer.analyze_trends(weekly_counts)
        assert any("promptevolver" in r.lower() for r in report.recommendations)

    def test_hotspots_included_in_report(self, analyzer):
        """Pre-computed hotspots are included in the report."""
        hotspots = [
            DefectHotspot(
                agent_type=AgentType.WORKER.value,
                mode="build",
                defect_category=DefectCategory.HALLUCINATION,
                defect_count=15,
                defect_rate=0.3,
            ),
        ]
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 5},
            {DefectCategory.HALLUCINATION: 8},
        ]
        report = analyzer.analyze_trends(weekly_counts, hotspots=hotspots)
        assert len(report.hotspots) == 1
        assert report.hotspots[0].agent_type == AgentType.WORKER.value

    def test_empty_data_produces_empty_report(self, analyzer):
        """Empty weekly counts produce an empty report."""
        report = analyzer.analyze_trends([])
        assert len(report.trends) == 0
        assert report.top_defect_category is None

    def test_single_week_insufficient(self, analyzer):
        """A single week of data is insufficient for trend analysis."""
        weekly_counts = [
            {DefectCategory.HALLUCINATION: 5},
        ]
        report = analyzer.analyze_trends(weekly_counts)
        # Need at least 2 weeks
        assert len(report.trends) == 0
