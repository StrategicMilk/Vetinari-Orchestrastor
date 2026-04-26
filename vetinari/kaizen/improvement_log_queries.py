"""Kaizen improvement log query helpers.

Read-only query operations extracted from ImprovementLog to keep that class
within the 550-line limit.  Functions accept a ``log: ImprovementLog`` as their
first argument and call its ``_connect()`` and ``_row_to_record()`` directly.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetinari.kaizen.improvement_log import ImprovementLog

from vetinari.kaizen.improvement_log import ImprovementStatus, KaizenReport, Observation
from vetinari.types import StatusEnum

__all__ = [
    "get_active_improvements",
    "get_confirmed_improvements",
    "get_confirmed_this_week",
    "get_improvement",
    "get_improvements_by_status",
    "get_observations",
    "get_proposed_this_week",
    "get_reverted_this_week",
    "get_weekly_report",
]


def get_improvement(log: ImprovementLog, improvement_id: str):
    """Retrieve a single improvement by ID.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.
        improvement_id: The improvement to retrieve.

    Returns:
        The ImprovementRecord, or None if not found.
    """
    with log._connect() as conn:
        row = conn.execute(
            "SELECT * FROM improvements WHERE id = ?",
            (improvement_id,),
        ).fetchone()
    if row is None:
        return None
    return log._row_to_record(row)


def get_improvements_by_status(log: ImprovementLog, status: ImprovementStatus) -> list:
    """Retrieve all improvements with a given status.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.
        status: The status to filter by.

    Returns:
        List of matching ImprovementRecord instances.
    """
    with log._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM improvements WHERE status = ? ORDER BY applied_at DESC",
            (status.value,),
        ).fetchall()
    return [log._row_to_record(r) for r in rows]


def get_active_improvements(log: ImprovementLog) -> list:
    """Return all improvements in ACTIVE status.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.

    Returns:
        List of active ImprovementRecord instances.
    """
    return get_improvements_by_status(log, ImprovementStatus.ACTIVE)


def get_confirmed_improvements(log: ImprovementLog) -> list:
    """Return all improvements in CONFIRMED status.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.

    Returns:
        List of confirmed ImprovementRecord instances.
    """
    return get_improvements_by_status(log, ImprovementStatus.CONFIRMED)


def get_observations(
    log: ImprovementLog,
    improvement_id: str,
    days: int | None = None,
) -> list[Observation]:
    """Retrieve observations for an improvement, optionally filtered by recency.

    Args:
        log: ImprovementLog instance providing _connect.
        improvement_id: The improvement whose observations to retrieve.
        days: If provided, only return observations from the last N days.

    Returns:
        List of Observation instances in chronological order.
    """
    with log._connect() as conn:
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            rows = conn.execute(
                """SELECT * FROM improvement_observations
                   WHERE improvement_id = ? AND observed_at >= ?
                   ORDER BY observed_at""",
                (improvement_id, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM improvement_observations
                   WHERE improvement_id = ? ORDER BY observed_at""",
                (improvement_id,),
            ).fetchall()
    return [
        Observation(
            observation_id=r["observation_id"],
            improvement_id=r["improvement_id"],
            observed_at=datetime.fromisoformat(r["observed_at"]),
            metric_value=r["metric_value"],
            sample_size=r["sample_size"],
        )
        for r in rows
    ]


def get_weekly_report(log: ImprovementLog) -> KaizenReport:
    """Generate a summary report of the kaizen system's current state.

    Args:
        log: ImprovementLog instance providing _connect.

    Returns:
        A KaizenReport with counts per status and average improvement effect.
    """
    with log._connect() as conn:
        counts: dict[str, int] = {}
        for row in conn.execute(
            "SELECT status, COUNT(*) as cnt FROM improvements GROUP BY status",
        ):
            counts[row["status"]] = row["cnt"]

        confirmed_rows = conn.execute(
            """SELECT actual_value, baseline_value FROM improvements
               WHERE status = 'confirmed' AND actual_value IS NOT NULL""",
        ).fetchall()

    avg_effect = 0.0
    if confirmed_rows:
        effects = [r["actual_value"] - r["baseline_value"] for r in confirmed_rows]
        avg_effect = statistics.mean(effects)

    return KaizenReport(
        total_proposed=counts.get("proposed", 0),
        total_active=counts.get("active", 0),
        total_confirmed=counts.get("confirmed", 0),
        total_failed=counts.get(StatusEnum.FAILED.value, 0),
        total_reverted=counts.get("reverted", 0),
        avg_improvement_effect=avg_effect,
    )


def get_confirmed_this_week(log: ImprovementLog) -> list:
    """Return improvements confirmed in the last 7 days.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.

    Returns:
        List of recently confirmed ImprovementRecord instances.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    with log._connect() as conn:
        rows = conn.execute(
            """SELECT * FROM improvements
               WHERE status = 'confirmed' AND confirmed_at >= ?
               ORDER BY confirmed_at DESC""",
            (cutoff,),
        ).fetchall()
    return [log._row_to_record(r) for r in rows]


def get_reverted_this_week(log: ImprovementLog) -> list:
    """Return improvements reverted in the last 7 days.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.

    Returns:
        List of recently reverted ImprovementRecord instances.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    with log._connect() as conn:
        rows = conn.execute(
            """SELECT * FROM improvements
               WHERE status = 'reverted' AND reverted_at >= ?
               ORDER BY reverted_at DESC""",
            (cutoff,),
        ).fetchall()
    return [log._row_to_record(r) for r in rows]


def get_proposed_this_week(log: ImprovementLog) -> list:
    """Return improvements proposed within the current ISO week (Monday-Sunday).

    Uses the ISO week start (Monday at 00:00 UTC) as the cutoff rather than a
    rolling 7-day window, so the count stays consistent throughout the week and
    does not include improvements from the previous week late on Sunday.

    Args:
        log: ImprovementLog instance providing _connect and _row_to_record.

    Returns:
        List of ImprovementRecord instances proposed this ISO week.
    """
    now = datetime.now(timezone.utc)
    # ISO weekday: Monday=1, Sunday=7.  Subtract (weekday - 1) days to reach Monday.
    monday = now - timedelta(days=now.isoweekday() - 1)
    week_start = monday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    with log._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM improvements WHERE status = 'proposed' AND created_at >= ? ORDER BY rowid DESC",
            (week_start,),
        ).fetchall()
    return [log._row_to_record(r) for r in rows]
