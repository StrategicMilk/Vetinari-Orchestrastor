"""Pipeline checkpoint store — per-stage observability snapshots for trace replay and cost analysis.

Records each pipeline stage to SQLite after it completes, enabling:
- Post-hoc trace replay via /api/v1/replay
- Cost attribution analysis via /api/v1/cost-analysis
- Debugging of production pipeline execution history

This module is the write side of the observability loop; the replay and cost-analysis
endpoints are the read side.  Every save is non-fatal — checkpoint failures must never
block the pipeline itself.

Pipeline role: Execute → **Checkpoint** → Replay/Analysis.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# -- Module-level singleton (double-checked locking) --
_store: PipelineCheckpointStore | None = None
_store_lock = threading.Lock()

# Snapshot size cap: 64 KB per input/output to bound database growth.
# Larger payloads are truncated with a preview to preserve debuggability.
_MAX_SNAPSHOT_BYTES = 65_536


@dataclass
class PipelineCheckpoint:
    """A single per-stage pipeline execution snapshot.

    Attributes:
        trace_id: Pipeline-level trace identifier (from CorrelationContext).
        execution_id: Request queue execution identifier (``_exec_id`` in context).
        step_name: Stage name (e.g. ``"intake"``, ``"planning"``, ``"worker"``).
        step_index: Zero-based stage ordering within the trace.
        status: ``"completed"``, ``"failed"``, or ``"skipped"``.
        input_snapshot: Stage input data; truncated to 64 KB.
        output_snapshot: Stage output data; truncated to 64 KB.
        tokens_used: Tokens consumed in this stage (0 for non-inference stages).
        latency_ms: Wall-clock duration of the stage in milliseconds.
        model_id: Model used in this stage; empty for non-LLM stages.
        quality_score: Inspector quality score; ``None`` if not yet scored.
        created_at: ISO-8601 UTC timestamp when the checkpoint was saved.
    """

    trace_id: str
    execution_id: str
    step_name: str
    step_index: int = 0
    status: str = "completed"  # completed | failed | skipped
    input_snapshot: dict[str, Any] = field(default_factory=dict)
    output_snapshot: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    quality_score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        """Show key fields for debugging."""
        return (
            f"PipelineCheckpoint(trace_id={self.trace_id!r}, step={self.step_name!r}, "
            f"status={self.status!r}, tokens={self.tokens_used}, latency_ms={self.latency_ms:.1f})"
        )


# -- CheckpointStore ----------------------------------------------------------


class PipelineCheckpointStore:
    """SQLite-backed store for pipeline stage checkpoints.

    Reads and writes the ``pipeline_traces`` table via the shared
    ``vetinari.database`` connection.  All public methods are non-fatal:
    failures are logged at WARNING and the caller continues normally.
    """

    def save_checkpoint(self, checkpoint: PipelineCheckpoint) -> None:
        """Persist a stage checkpoint to the ``pipeline_traces`` table.

        Snapshots larger than 64 KB are truncated to a preview dict to bound
        storage growth while preserving enough context for debugging.

        Args:
            checkpoint: The checkpoint to persist.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            input_json = json.dumps(checkpoint.input_snapshot, default=str)
            output_json = json.dumps(checkpoint.output_snapshot, default=str)
            # Truncate oversized snapshots rather than failing
            if len(input_json) > _MAX_SNAPSHOT_BYTES:
                input_json = json.dumps({"_truncated": True, "preview": input_json[:500]})
            if len(output_json) > _MAX_SNAPSHOT_BYTES:
                output_json = json.dumps({"_truncated": True, "preview": output_json[:500]})
            conn.execute(
                """
                INSERT INTO pipeline_traces (
                    trace_id, execution_id, step_name, step_index, status,
                    input_snapshot_json, output_snapshot_json,
                    tokens_used, latency_ms, model_id, quality_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.trace_id,
                    checkpoint.execution_id,
                    checkpoint.step_name,
                    checkpoint.step_index,
                    checkpoint.status,
                    input_json,
                    output_json,
                    checkpoint.tokens_used,
                    checkpoint.latency_ms,
                    checkpoint.model_id,
                    checkpoint.quality_score,
                    checkpoint.created_at,
                ),
            )
            conn.commit()
            logger.debug(
                "Saved checkpoint trace=%s step=%s status=%s",
                checkpoint.trace_id,
                checkpoint.step_name,
                checkpoint.status,
            )
        except Exception:
            logger.warning(
                "Failed to save checkpoint for trace=%s step=%s — observability data lost, pipeline continues normally",
                checkpoint.trace_id,
                checkpoint.step_name,
                exc_info=True,
            )

    def load_checkpoint(self, trace_id: str, step_name: str) -> PipelineCheckpoint | None:
        """Load the most recent checkpoint for a given trace and step.

        Args:
            trace_id: The pipeline trace identifier.
            step_name: The stage name to load (e.g. ``"planning"``).

        Returns:
            The most recent matching checkpoint, or ``None`` if not found.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            row = conn.execute(
                """
                SELECT trace_id, execution_id, step_name, step_index, status,
                       input_snapshot_json, output_snapshot_json,
                       tokens_used, latency_ms, model_id, quality_score, created_at
                FROM pipeline_traces
                WHERE trace_id = ? AND step_name = ?
                ORDER BY id DESC LIMIT 1
                """,
                (trace_id, step_name),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_checkpoint(row)
        except Exception:
            logger.warning(
                "Failed to load checkpoint for trace=%s step=%s",
                trace_id,
                step_name,
                exc_info=True,
            )
            return None

    def list_checkpoints(self, trace_id: str) -> list[PipelineCheckpoint]:
        """List all checkpoints for a trace in step_index order.

        Args:
            trace_id: The pipeline trace identifier.

        Returns:
            All checkpoints for this trace, ordered by ``step_index`` ascending.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                """
                SELECT trace_id, execution_id, step_name, step_index, status,
                       input_snapshot_json, output_snapshot_json,
                       tokens_used, latency_ms, model_id, quality_score, created_at
                FROM pipeline_traces
                WHERE trace_id = ?
                ORDER BY step_index ASC, id ASC
                """,
                (trace_id,),
            ).fetchall()
            return [self._row_to_checkpoint(r) for r in rows]
        except Exception:
            logger.warning(
                "Failed to list checkpoints for trace=%s",
                trace_id,
                exc_info=True,
            )
            return []

    def list_traces(self, since: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List distinct traces with summary statistics.

        Args:
            since: ISO-8601 datetime string; only traces created at or after
                this time are returned.  ``None`` returns the most recent traces.
            limit: Maximum number of distinct traces to return.

        Returns:
            List of dicts with keys: ``trace_id``, ``execution_id``,
            ``step_count``, ``total_tokens``, ``total_latency_ms``, ``created_at``.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            if since:
                rows = conn.execute(
                    """
                    SELECT trace_id, execution_id,
                           COUNT(*) AS step_count,
                           COALESCE(SUM(tokens_used), 0) AS total_tokens,
                           COALESCE(SUM(latency_ms), 0.0) AS total_latency_ms,
                           MAX(created_at) AS created_at
                    FROM pipeline_traces
                    GROUP BY trace_id
                    HAVING MAX(created_at) >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT trace_id, execution_id,
                           COUNT(*) AS step_count,
                           COALESCE(SUM(tokens_used), 0) AS total_tokens,
                           COALESCE(SUM(latency_ms), 0.0) AS total_latency_ms,
                           MAX(created_at) AS created_at
                    FROM pipeline_traces
                    GROUP BY trace_id
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.warning("Failed to list traces", exc_info=True)
            return []

    def list_all_checkpoints(
        self,
        since: str | None = None,
        limit: int = 1_000,
    ) -> list[PipelineCheckpoint]:
        """Return individual checkpoint records for cost analysis.

        Unlike ``list_traces`` (which returns one summary row per trace),
        this returns one ``PipelineCheckpoint`` per pipeline stage row so that
        callers can aggregate ``tokens_used`` and ``quality_score`` at any
        granularity (per model, per stage, per trace).

        Args:
            since: ISO-8601 datetime string; only rows created at or after
                this time are returned.  ``None`` returns the most recent rows.
            limit: Maximum number of individual checkpoint rows to return.
                Capped at the caller's discretion; large values scan the full
                table.

        Returns:
            List of ``PipelineCheckpoint`` objects ordered by ``created_at``
            descending, newest first.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            if since:
                rows = conn.execute(
                    """
                    SELECT trace_id, execution_id, step_name, step_index, status,
                           input_snapshot_json, output_snapshot_json,
                           tokens_used, latency_ms, model_id, quality_score, created_at
                    FROM pipeline_traces
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT trace_id, execution_id, step_name, step_index, status,
                           input_snapshot_json, output_snapshot_json,
                           tokens_used, latency_ms, model_id, quality_score, created_at
                    FROM pipeline_traces
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [self._row_to_checkpoint(r) for r in rows]
        except Exception:
            logger.warning("Failed to list all checkpoints for cost analysis", exc_info=True)
            return []

    def _row_to_checkpoint(self, row: Any) -> PipelineCheckpoint:
        """Convert a ``sqlite3.Row`` to a ``PipelineCheckpoint``.

        Args:
            row: A row from the ``pipeline_traces`` table.

        Returns:
            Populated ``PipelineCheckpoint`` instance.
        """
        try:
            input_snap = json.loads(row["input_snapshot_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            input_snap = {}
        try:
            output_snap = json.loads(row["output_snapshot_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            output_snap = {}
        return PipelineCheckpoint(
            trace_id=row["trace_id"],
            execution_id=row["execution_id"] or "",
            step_name=row["step_name"],
            step_index=row["step_index"] or 0,
            status=row["status"] or "completed",
            input_snapshot=input_snap,
            output_snapshot=output_snap,
            tokens_used=row["tokens_used"] or 0,
            latency_ms=float(row["latency_ms"] or 0.0),
            model_id=row["model_id"] or "",
            quality_score=row["quality_score"],
            created_at=row["created_at"] or "",
        )


# -- Singleton factory --------------------------------------------------------


def get_checkpoint_store() -> PipelineCheckpointStore:
    """Return the module-level singleton ``PipelineCheckpointStore``, creating it if needed.

    Uses double-checked locking so repeated calls are lock-free after first init.

    Returns:
        The singleton ``PipelineCheckpointStore`` instance.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PipelineCheckpointStore()
    return _store


def reset_checkpoint_store() -> None:
    """Reset the singleton for test isolation.

    After this call, the next ``get_checkpoint_store()`` creates a fresh instance.
    """
    global _store
    with _store_lock:
        _store = None
