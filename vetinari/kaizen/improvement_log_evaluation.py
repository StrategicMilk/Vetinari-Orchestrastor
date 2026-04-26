"""Kaizen improvement evaluation and revert helpers.

Extracted from ImprovementLog to keep that class within the 550-line limit.
Functions accept a ``log: ImprovementLog`` as their first argument and call its
``_connect()``, ``_lock``, and event emission methods directly.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetinari.kaizen.improvement_log import ImprovementLog

from vetinari.exceptions import ExecutionError
from vetinari.kaizen.improvement_log import REGRESSION_THRESHOLD, ImprovementStatus

logger = logging.getLogger(__name__)

# Metrics where a lower value is better (latency, cost, error rate, rework).
# For these metrics the CONFIRMED and FAILED thresholds are inverted relative
# to higher-is-better metrics like quality score or throughput.
_LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    {
        "latency",
        "latency_ms",
        "cost",
        "cost_per_task",
        "error_rate",
        "rework_rate",
        "failure_rate",
        "p95_latency",
        "p99_latency",
    }
)

__all__ = [
    "evaluate",
    "is_lower_is_better",
    "revert",
]


def is_lower_is_better(metric: str) -> bool:
    """Return True if a lower metric value indicates improvement.

    Args:
        metric: The metric name (e.g. 'latency', 'cost', 'error_rate').

    Returns:
        True for latency, cost, and rework-rate style metrics; False for
        quality, throughput, and others.
    """
    return metric in _LOWER_IS_BETTER_METRICS


def evaluate(log: ImprovementLog, improvement_id: str) -> ImprovementStatus:
    """Evaluate an improvement after its observation window.

    Compares mean observation value against target and baseline:
    - actual >= target → CONFIRMED
    - actual < baseline * 0.95 → FAILED
    - otherwise → stays ACTIVE (keep observing)

    Args:
        log: ImprovementLog instance providing _connect, _lock, and emit helpers.
        improvement_id: The improvement to evaluate.

    Returns:
        The resulting ImprovementStatus after evaluation.

    Raises:
        ExecutionError: If the improvement does not exist or has no observations.
    """
    with log._lock, log._connect() as conn:
        row = conn.execute(
            "SELECT * FROM improvements WHERE id = ?",
            (improvement_id,),
        ).fetchone()
        if row is None:
            raise ExecutionError(f"Improvement not found: {improvement_id}")

        observations = conn.execute(
            """SELECT metric_value FROM improvement_observations
               WHERE improvement_id = ? ORDER BY observed_at""",
            (improvement_id,),
        ).fetchall()

        if not observations:
            raise ExecutionError(
                f"No observations for improvement {improvement_id} — cannot evaluate",
            )

        actual = statistics.mean([o["metric_value"] for o in observations])
        now = datetime.now(timezone.utc).isoformat()

        baseline = row["baseline_value"]
        target = row["target_value"]
        metric = row["metric"]
        lower_better = is_lower_is_better(metric)

        # Direction-aware thresholds:
        #   Higher-is-better: confirmed when actual >= target; failed when actual < baseline*0.95
        #   Lower-is-better:  confirmed when actual <= target; failed when actual > baseline*1.05
        if lower_better:
            # For latency/cost/rework, improvement means going DOWN.
            regression_ceil = baseline * (2.0 - REGRESSION_THRESHOLD)  # e.g. baseline*1.05
            is_confirmed = actual <= target
            is_failed = actual > regression_ceil
        else:
            is_confirmed = actual >= target
            is_failed = actual < baseline * REGRESSION_THRESHOLD

        if is_confirmed:
            # Verify: can only confirm if in ACTIVE state with applied_at set
            if row["status"] != ImprovementStatus.ACTIVE.value or row["applied_at"] is None:
                raise ExecutionError(
                    f"Cannot confirm improvement {improvement_id}: "
                    f"must be in ACTIVE state with applied_at set. "
                    f"Current state={row['status']}, applied_at={row['applied_at']}"
                )
            new_status = ImprovementStatus.CONFIRMED
            conn.execute(
                "UPDATE improvements SET status = ?, actual_value = ?, confirmed_at = ? WHERE id = ?",
                (new_status.value, actual, now, improvement_id),
            )
            logger.info(
                "Improvement CONFIRMED: %s metric=%s (actual=%.3f, target=%.3f)",
                improvement_id,
                metric,
                actual,
                target,
            )
        elif is_failed:
            new_status = ImprovementStatus.FAILED
            conn.execute(
                "UPDATE improvements SET status = ?, actual_value = ? WHERE id = ?",
                (new_status.value, actual, improvement_id),
            )
            logger.warning(
                "Improvement FAILED: %s metric=%s (actual=%.3f regressed from baseline=%.3f)",
                improvement_id,
                metric,
                actual,
                baseline,
            )
        else:
            new_status = ImprovementStatus.ACTIVE
            conn.execute(
                "UPDATE improvements SET actual_value = ? WHERE id = ?",
                (actual, improvement_id),
            )
            logger.info(
                "Improvement still ACTIVE: %s metric=%s (actual=%.3f, target=%.3f, baseline=%.3f)",
                improvement_id,
                metric,
                actual,
                target,
                baseline,
            )

    if new_status == ImprovementStatus.CONFIRMED:
        log._emit_confirmed(
            improvement_id,
            row["metric"],
            baseline,
            actual,
            row["applied_by"],
        )

    return new_status


def revert(log: ImprovementLog, improvement_id: str) -> None:
    """Mark an improvement as reverted.

    Only ACTIVE or CONFIRMED improvements may be reverted.

    Args:
        log: ImprovementLog instance providing _connect, _lock, and emit helpers.
        improvement_id: The improvement to revert.

    Raises:
        ExecutionError: If the improvement does not exist or cannot be reverted.
    """
    now = datetime.now(timezone.utc).isoformat()
    revertible = {
        ImprovementStatus.ACTIVE.value,
        ImprovementStatus.CONFIRMED.value,
    }
    with log._lock, log._connect() as conn:
        row = conn.execute(
            "SELECT id, status, metric FROM improvements WHERE id = ?",
            (improvement_id,),
        ).fetchone()
        if row is None:
            raise ExecutionError(f"Improvement not found: {improvement_id}")
        if row["status"] not in revertible:
            raise ExecutionError(
                f"Cannot revert improvement in state '{row['status']}' (must be active or confirmed): {improvement_id}",
            )
        conn.execute(
            "UPDATE improvements SET status = 'reverted', reverted_at = ?, regression_detected = 1 WHERE id = ?",
            (now, improvement_id),
        )
        metric = row["metric"]
    logger.info("Improvement reverted: %s", improvement_id)
    log._emit_reverted(improvement_id, metric, "regression_detected")
