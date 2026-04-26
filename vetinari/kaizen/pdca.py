"""PDCA Feedback Loop Controller — Wires Kaizen improvements to real actions.

Closes the Plan-Do-Check-Act loop by:
- **Applying** improvements when activated (Do phase)
- **Persisting** improvements when confirmed (Act phase)
- **Auto-proposing** improvements when defect trends worsen (Plan phase)

Without this module, improvements are tracked in SQLite but never actually
change system behavior.  The PDCAController registers ``ImprovementApplicator``
callables keyed by metric name; when an improvement targeting that metric is
activated, the applicator is invoked.  On confirmation the applied change is
written to a durable JSON overrides file so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetinari.kaizen.knowledge_lint import KnowledgeLintReport

from vetinari.constants import get_user_dir
from vetinari.exceptions import ExecutionError
from vetinari.kaizen.defect_trends import (
    DefectHotspot,
    DefectTrendAnalyzer,
    build_hypothesis,
    is_valid_category,
)
from vetinari.kaizen.improvement_log import (
    ImprovementLog,
    ImprovementRecord,
    ImprovementStatus,
)
from vetinari.validation import DefectCategory

logger = logging.getLogger(__name__)


def _get_default_overrides_path() -> Path:
    """Return the default overrides path, using get_user_dir() for testability."""
    return get_user_dir() / "kaizen_overrides.json"


ImprovementApplicator = Callable[[ImprovementRecord], dict[str, Any]]  # applies an improvement, returns changes

# ── Built-in: Threshold applicator ───────────────────────────────────────────


@dataclass
class ThresholdOverride:
    """A runtime threshold override applied by the PDCA loop."""

    metric: str
    previous_value: float
    new_value: float
    improvement_id: str
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed: bool = False

    def __repr__(self) -> str:
        """Compact representation showing metric, value change, and confirmation state."""
        return (
            f"ThresholdOverride(metric={self.metric!r}, "
            f"{self.previous_value}->{self.new_value}, "
            f"id={self.improvement_id!r}, confirmed={self.confirmed})"
        )


class ThresholdApplicator:
    """Applies improvements by adjusting runtime threshold values.

    Manages a registry of named thresholds (e.g. ``quality``, ``latency``,
    ``throughput``) with current values.  When an improvement targeting one
    of these metrics is activated, the threshold is adjusted toward the
    target value.

    Args:
        initial_thresholds: Starting threshold values keyed by metric name.
    """

    def __init__(self, initial_thresholds: dict[str, float] | None = None) -> None:
        self._thresholds: dict[str, float] = dict(initial_thresholds or {})  # noqa: VET112 - empty fallback preserves optional request metadata contract
        self._overrides: list[ThresholdOverride] = []
        self._lock = threading.Lock()

    @property
    def thresholds(self) -> dict[str, float]:
        """Current threshold values (read-only copy)."""
        with self._lock:
            return dict(self._thresholds)

    @property
    def overrides(self) -> list[ThresholdOverride]:
        """History of all threshold overrides applied."""
        return list(self._overrides)

    def get_threshold(self, metric: str) -> float | None:
        """Get the current value for a named threshold.

        Args:
            metric: The threshold metric name.

        Returns:
            Current value, or None if the metric is not registered.
        """
        with self._lock:
            return self._thresholds.get(metric)

    def __call__(self, record: ImprovementRecord) -> dict[str, Any]:
        """Apply a threshold adjustment based on the improvement's target.

        Moves the threshold for ``record.metric`` to ``record.target_value``.
        If the metric is not registered, the override is still recorded
        (the metric is created).

        Args:
            record: The improvement being activated.

        Returns:
            Dict describing the change: metric, previous, new, improvement_id.
        """
        with self._lock:
            previous = self._thresholds.get(record.metric, record.baseline_value)
            self._thresholds[record.metric] = record.target_value

            override = ThresholdOverride(
                metric=record.metric,
                previous_value=previous,
                new_value=record.target_value,
                improvement_id=record.id,
            )
            self._overrides.append(override)

        logger.info(
            "Threshold applied: metric=%s, %s -> %s (improvement=%s)",
            record.metric,
            previous,
            record.target_value,
            record.id,
        )
        return {
            "metric": record.metric,
            "previous": previous,
            "new": record.target_value,
            "improvement_id": record.id,
        }

    def confirm_override(self, improvement_id: str) -> None:
        """Mark an override as confirmed (permanently applied).

        Args:
            improvement_id: The improvement whose override to confirm.
        """
        with self._lock:
            for override in self._overrides:
                if override.improvement_id == improvement_id:
                    override.confirmed = True

    def revert_override(self, improvement_id: str) -> float | None:
        """Revert a threshold to its pre-override value.

        Args:
            improvement_id: The improvement whose override to revert.

        Returns:
            The reverted-to value, or None if no override was found.
        """
        with self._lock:
            for override in reversed(self._overrides):
                if override.improvement_id == improvement_id and not override.confirmed:
                    self._thresholds[override.metric] = override.previous_value
                    logger.info(
                        "Threshold reverted: metric=%s, %s -> %s (improvement=%s)",
                        override.metric,
                        override.new_value,
                        override.previous_value,
                        improvement_id,
                    )
                    return override.previous_value
        return None


# ── PDCA Controller ──────────────────────────────────────────────────────────


class PDCAController:
    """Orchestrates the full PDCA feedback loop for kaizen improvements.

    Bridges the gap between ImprovementLog (data store) and real system
    changes.  Registers applicators per metric, invokes them on activation,
    persists changes on confirmation, and auto-proposes improvements when
    defect trends worsen.

    Args:
        improvement_log: The kaizen ImprovementLog instance.
        overrides_path: Path to the JSON file for persisting confirmed overrides.
    """

    def __init__(
        self,
        improvement_log: ImprovementLog,
        overrides_path: Path | str | None = None,
    ) -> None:
        self._log = improvement_log
        self._overrides_path = Path(overrides_path) if overrides_path else _get_default_overrides_path()
        self._applicators: dict[str, ImprovementApplicator] = {}
        self._applied: dict[str, dict[str, Any]] = {}
        self._trend_analyzer = DefectTrendAnalyzer()

    def register_applicator(self, metric: str, applicator: ImprovementApplicator) -> None:
        """Register an applicator for a given metric name.

        Args:
            metric: The metric name (e.g. 'quality', 'latency', 'throughput').
            applicator: Callable that applies improvements for this metric.
        """
        self._applicators[metric] = applicator
        logger.info("Registered improvement applicator for metric=%s", metric)

    # ── Do phase: activate and apply ─────────────────────────────────────

    def activate_and_apply(self, improvement_id: str) -> dict[str, Any]:
        """Activate an improvement and apply it to the running system.

        Calls ``ImprovementLog.activate()`` to transition the status, then
        looks up the registered applicator for the improvement's metric and
        invokes it.  If no applicator is registered, the improvement is
        still activated but nothing is applied.

        Args:
            improvement_id: The improvement to activate.

        Returns:
            Dict describing what was applied (empty if no applicator matched).

        Raises:
            ValueError: If the improvement does not exist or is not in
                PROPOSED status.
        """
        # Transition status in the log — ImprovementLog.activate() emits
        # KaizenImprovementActive internally, so we must NOT emit again here.
        self._log.activate(improvement_id)

        record = self._log.get_improvement(improvement_id)
        if record is None:
            raise ExecutionError(f"Improvement not found after activation: {improvement_id}")

        applicator = self._applicators.get(record.metric)
        if applicator is None:
            logger.info(
                "No applicator registered for metric=%s; improvement %s activated but not applied",
                record.metric,
                improvement_id,
            )
            return {}

        try:
            changes = applicator(record)
        except Exception:
            # Roll back to PROPOSED if the applicator fails — leaving the
            # improvement ACTIVE with no actual change applied would make the
            # observation window meaningless and could corrupt the baseline.
            logger.error(
                "Applicator for metric=%s raised while applying improvement %s — "
                "reverting to PROPOSED so the improvement can be retried",
                record.metric,
                improvement_id,
                exc_info=True,
            )
            self._log.revert_to_proposed(improvement_id)
            raise

        self._applied[improvement_id] = changes
        logger.info(
            "Improvement applied: id=%s, metric=%s, changes=%s",
            improvement_id,
            record.metric,
            changes,
        )
        return changes

    # ── Act phase: confirm and persist ───────────────────────────────────

    def confirm_and_persist(self, improvement_id: str) -> None:
        """Persist an improvement's changes after successful evaluation.

        Writes the applied changes to the overrides JSON file so they
        survive restarts.  Should be called after ``ImprovementLog.evaluate()``
        returns CONFIRMED.

        Args:
            improvement_id: The improvement to persist.
        """
        record = self._log.get_improvement(improvement_id)
        if record is None:
            logger.warning("Cannot persist unknown improvement: %s", improvement_id)
            return

        if record.status != ImprovementStatus.CONFIRMED:
            logger.warning(
                "Cannot persist improvement %s — status is %s, expected CONFIRMED",
                improvement_id,
                record.status.value,
            )
            return

        changes = self._applied.get(improvement_id, {})
        self._write_override(improvement_id, record, changes)

        # Mark the applicator override as confirmed
        applicator = self._applicators.get(record.metric)
        if isinstance(applicator, ThresholdApplicator):
            applicator.confirm_override(improvement_id)

        logger.info(
            "Improvement persisted: id=%s, metric=%s",
            improvement_id,
            record.metric,
        )

    def _write_override(
        self,
        improvement_id: str,
        record: ImprovementRecord,
        changes: dict[str, Any],
    ) -> None:
        """Append an override entry to the JSON overrides file.

        Args:
            improvement_id: The improvement ID.
            record: The improvement record.
            changes: The changes dict returned by the applicator.
        """
        self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
        overrides: list[dict[str, Any]] = []
        if self._overrides_path.exists():
            try:
                raw = self._overrides_path.read_text(encoding="utf-8")
                overrides = json.loads(raw) if raw.strip() else []
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Could not read overrides file %s — starting fresh",
                    self._overrides_path,
                )

        overrides.append({
            "improvement_id": improvement_id,
            "metric": record.metric,
            "hypothesis": record.hypothesis,
            "baseline_value": record.baseline_value,
            "target_value": record.target_value,
            "actual_value": record.actual_value,
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
            "changes": changes,
        })
        self._overrides_path.write_text(
            json.dumps(overrides, indent=2),
            encoding="utf-8",
        )

    def load_persisted_overrides(self) -> list[dict[str, Any]]:
        """Load previously persisted overrides from the JSON file.

        Returns:
            List of override dicts, or empty list if file doesn't exist.
        """
        if not self._overrides_path.exists():
            return []
        try:
            raw = self._overrides_path.read_text(encoding="utf-8")
            return json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read overrides file: %s", self._overrides_path)
            return []

    # ── Plan phase: trend analysis → auto-propose ────────────────────────

    def check_trends_and_propose(
        self,
        weekly_counts: list[dict[str, int]] | None = None,
        hotspot_data: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Analyze defect trends and auto-propose improvements for worsening metrics.

        When no ``weekly_counts`` are provided, fetches them from the
        ImprovementLog's defect tracking tables.  Concerning trends
        (>15% week-over-week increase) generate automatic improvement
        proposals.

        Args:
            weekly_counts: Optional pre-computed weekly defect counts.
                Each dict maps category string to count.
            hotspot_data: Optional pre-computed hotspot data from
                ``ImprovementLog.get_defect_hotspots()``.

        Returns:
            List of improvement IDs that were proposed.
        """
        if weekly_counts is None:
            weekly_counts = self._log.get_weekly_defect_counts(weeks=4)

        if not weekly_counts or len(weekly_counts) < 2:
            logger.info("Insufficient defect data for trend analysis (need >= 2 weeks)")
            return []

        typed_counts: list[dict[DefectCategory, int]] = []
        for week in weekly_counts:
            typed_week: dict[DefectCategory, int] = {}
            for cat_str, count in week.items():
                try:
                    typed_week[DefectCategory(cat_str)] = count
                except ValueError:
                    logger.warning("Skipping unknown defect category: %s", cat_str)
            typed_counts.append(typed_week)

        hotspots: list[DefectHotspot] | None = None
        if hotspot_data:
            hotspots = [
                DefectHotspot(
                    agent_type=h["agent_type"],
                    mode=h["mode"],
                    defect_category=DefectCategory(h["category"]),
                    defect_count=h["count"],
                    defect_rate=h.get("defect_rate", 0.0),
                )
                for h in hotspot_data
                if is_valid_category(h.get("category", ""))
            ]

        report = self._trend_analyzer.analyze_trends(typed_counts, hotspots)

        proposed_ids: list[str] = []
        for trend in report.trends.values():
            if not trend.is_concerning:
                continue

            hypothesis = build_hypothesis(trend.category, trend.change_pct)
            imp_id = self._log.propose(
                hypothesis=hypothesis,
                metric="defect_count",
                baseline=float(trend.current_count),
                target=float(max(trend.previous_count - 1, 0)),
                applied_by="pdca_trend_monitor",
                rollback_plan="Revert to previous configuration for this defect category",
            )
            proposed_ids.append(imp_id)
            logger.info(
                "Auto-proposed improvement %s for worsening %s trend (+%.0f%%)",
                imp_id,
                trend.category.value,
                trend.change_pct * 100,
            )

        logger.info(
            "Trend analysis complete: %d improvement(s) proposed",
            len(proposed_ids),
        )
        return proposed_ids

    # ── Knowledge lint ───────────────────────────────────────────────────

    def knowledge_lint(self) -> KnowledgeLintReport:
        """Run knowledge lint checks on all memory entries.

        Returns:
            KnowledgeLintReport with findings from contradiction, stale,
            orphaned, and vocabulary drift checks.

        Raises:
            Exception: If memory store is unavailable or linter fails.
        """
        from vetinari.kaizen.knowledge_lint import KnowledgeLinter, propose_lint_findings
        from vetinari.memory.unified import UnifiedMemoryStore

        try:
            entries = UnifiedMemoryStore().search("", limit=10_000)
        except Exception:
            logger.error(
                "Knowledge lint: backing store unavailable — cannot run lint checks",
                exc_info=True,
            )
            raise

        # Linter failures propagate — callers must not silently receive an
        # empty report when the linter itself is broken.
        report = KnowledgeLinter().lint_all(entries)

        try:
            propose_lint_findings(self._log, report)
        except Exception:
            logger.warning(
                "Knowledge lint: propose_lint_findings failed — findings not proposed to improvement log",
                exc_info=True,
            )

        return report

    # ── Full PDCA cycle convenience ──────────────────────────────────────

    def run_check_phase(self) -> list[str]:
        """Run the Check phase: evaluate active improvements and handle results.

        Evaluates all active improvements.  Confirmed ones are persisted;
        failed ones are logged.  This is the automated Check-Act bridge.

        Returns:
            List of improvement IDs that were confirmed and persisted.
        """
        active = self._log.get_active_improvements()
        confirmed_ids: list[str] = []

        now_utc = datetime.now(timezone.utc)
        for improvement in active:
            observations = self._log.get_observations(improvement.id)
            if not observations:
                # Skip improvements that are still within their observation window.
                # If the window has expired with no observations, revert to PROPOSED
                # so the improvement does not remain stuck in ACTIVE indefinitely.
                if improvement.applied_at is not None:
                    window_expires = improvement.applied_at + improvement.observation_window
                    if now_utc > window_expires:
                        logger.warning(
                            "Improvement %s stuck in ACTIVE: observation window expired "
                            "with no observations — reverting to PROPOSED for retry",
                            improvement.id,
                        )
                        try:
                            self._log.revert_to_proposed(improvement.id)
                        except Exception:
                            logger.error(
                                "Failed to revert stuck improvement %s to PROPOSED",
                                improvement.id,
                                exc_info=True,
                            )
                continue
            result = self._log.evaluate(improvement.id)
            if result == ImprovementStatus.CONFIRMED:
                self.confirm_and_persist(improvement.id)
                confirmed_ids.append(improvement.id)
            elif result == ImprovementStatus.FAILED:
                # Revert the applicator's changes
                applicator = self._applicators.get(improvement.metric)
                if isinstance(applicator, ThresholdApplicator):
                    applicator.revert_override(improvement.id)
                logger.info(
                    "Improvement %s failed evaluation — changes reverted",
                    improvement.id,
                )

        from vetinari.kaizen.knowledge_compactor import run_compaction_step

        run_compaction_step()

        return confirmed_ids
