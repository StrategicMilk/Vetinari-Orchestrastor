"""Improvement Aggregator — Weekly Kaizen Review.

Collects all improvements from all sources and produces a structured
kaizen report. The report includes top improvements, regressions,
improvement velocity, trend analysis, and actionable recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from vetinari.kaizen.gemba import GembaFinding
from vetinari.kaizen.improvement_log import (
    ImprovementLog,
    ImprovementRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class KaizenWeeklyReport:
    """Structured weekly kaizen report.

    Attributes:
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        top_improvements: Top 5 confirmed improvements by effect size.
        regressions: Active improvements with regression detected.
        improvement_velocity: Improvements confirmed per day this week.
        velocity_trend: Direction vs last period ('up', 'down', 'flat').
        open_hypotheses: Proposed but not yet activated improvements.
        active_experiments: Currently active improvements being observed.
        recommendations: Actionable suggestions based on patterns.
        gemba_findings: Latest gemba walk findings.
        generated_at: When this report was generated.
    """

    period_start: datetime | None = None
    period_end: datetime | None = None
    top_improvements: list[ImprovementRecord] = field(default_factory=list)
    regressions: list[ImprovementRecord] = field(default_factory=list)
    improvement_velocity: float = 0.0
    velocity_trend: str = "flat"
    open_hypotheses: list[ImprovementRecord] = field(default_factory=list)
    active_experiments: list[ImprovementRecord] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    gemba_findings: list[GembaFinding] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"KaizenWeeklyReport(velocity={self.improvement_velocity!r}, "
            f"velocity_trend={self.velocity_trend!r}, "
            f"top_improvements={len(self.top_improvements)}, "
            f"open_hypotheses={len(self.open_hypotheses)})"
        )


class ImprovementAggregator:
    """Produces weekly kaizen reports from all improvement sources.

    Aggregates data from the ImprovementLog and optional gemba walk
    to produce a structured weekly report with recommendations.

    Args:
        improvement_log: The ImprovementLog to aggregate from.
        gemba: Optional AutoGembaWalk instance for latest findings.
    """

    def __init__(
        self,
        improvement_log: ImprovementLog,
        gemba: object | None = None,
    ) -> None:
        self._log = improvement_log
        self._gemba = gemba

    def generate_weekly_report(self) -> KaizenWeeklyReport:
        """Generate a weekly kaizen report.

        Returns:
            A KaizenWeeklyReport with aggregated data and recommendations.
        """
        confirmed = self._log.get_confirmed_this_week()
        reverted = self._log.get_reverted_this_week()
        proposed = self._log.get_proposed_this_week()
        active = self._log.get_active_improvements()

        # Top improvements by effect size (absolute change from baseline)
        top = sorted(
            confirmed,
            key=lambda i: abs((i.actual_value or 0.0) - i.baseline_value),
            reverse=True,
        )[:5]

        # Regressions: active OR reverted improvements where regression_detected=True.
        # Reverted improvements must be included so the weekly report surfaces
        # regressions that have already been auto-reverted (not just open ones).
        regressions = [i for i in active if i.regression_detected] + [
            i for i in reverted if i.regression_detected
        ]

        # Velocity: confirmed per day
        velocity = len(confirmed) / 7.0

        # Gemba findings
        gemba_findings: list[GembaFinding] = []
        if self._gemba and hasattr(self._gemba, "latest_report"):
            latest = self._gemba.latest_report
            if latest:
                gemba_findings = latest.findings

        recommendations = self._generate_recommendations(
            active,
            confirmed,
            reverted,
            proposed,
        )

        now = datetime.now(timezone.utc)
        return KaizenWeeklyReport(
            period_end=now,
            top_improvements=top,
            regressions=regressions,
            improvement_velocity=velocity,
            velocity_trend="flat",
            open_hypotheses=proposed,
            active_experiments=active,
            recommendations=recommendations,
            gemba_findings=gemba_findings,
        )

    def _generate_recommendations(
        self,
        active: list[ImprovementRecord],
        confirmed: list[ImprovementRecord],
        reverted: list[ImprovementRecord],
        proposed: list[ImprovementRecord],
    ) -> list[str]:
        """Generate next-cycle recommendations based on patterns.

        Args:
            active: Currently active improvements.
            confirmed: Confirmed improvements this week.
            reverted: Reverted improvements this week.
            proposed: Proposed improvements this week.

        Returns:
            List of actionable recommendation strings.
        """
        recs: list[str] = []

        if len(reverted) > len(confirmed):
            recs.append(
                "More improvements being reverted than confirmed — hypotheses may be too aggressive",
            )

        if not active and not proposed:
            recs.append(
                "No active experiments or proposals — consider running gemba walk to find opportunities",
            )

        if len(active) > 10:
            recs.append(
                f"{len(active)} active experiments — consider focusing on fewer improvements for clearer signal",
            )

        if confirmed:
            # Check for single-source dominance
            sources = [i.applied_by for i in confirmed]
            if len(set(sources)) == 1 and len(confirmed) > 2:
                recs.append(
                    f"All confirmed improvements from {sources[0]} — other subsystems may need attention",
                )

        if not confirmed and not reverted and active:
            recs.append(
                "No improvements confirmed or reverted this week — check if observation windows are too long",
            )

        return recs
