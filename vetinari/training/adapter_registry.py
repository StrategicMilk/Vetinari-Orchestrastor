"""SQLite-backed registry for trained LoRA adapters.

Tracks every adapter produced by the training pipeline with full metadata:
base model, training date, data statistics, evaluation results, and
deployment status. This is the authoritative record of what was trained,
when, how well it performed, and whether it is currently deployed.

The ``LoRAAdapterManager`` in ``continual_learning.py`` manages the
runtime mapping of task_type → adapter path for inference. This registry
is the persistent historical record that survives across restarts and
supports dashboard queries, audit, and rollback decisions.

This module is step 7 of the training pipeline:
Data Curation → Training → Quality Gate → GGUF → Deploy → **Registry** → Inference.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton, protected by double-checked locking
_registry: TrainingAdapterRegistry | None = None
_registry_lock = threading.Lock()


@dataclass(frozen=True)
class AdapterRecord:
    """Metadata for a single trained adapter.

    Attributes:
        adapter_id: Unique identifier (typically the training run_id).
        base_model: HuggingFace model ID used as the training base.
        task_type: Task domain the adapter was trained for (e.g. "coding").
        adapter_path: Filesystem path to the saved LoRA adapter directory.
        training_date: ISO-8601 UTC timestamp of when training completed.
        training_examples: Number of examples used for training.
        epochs: Number of training epochs completed.
        eval_score: Post-training evaluation score (0.0-1.0).
        baseline_score: Pre-training baseline score for comparison.
        deployment_status: One of "deployed", "pending", "rolled_back", "rejected".
        gguf_path: Path to the converted GGUF model file (empty if not converted).
        data_stats: JSON-serialised training data statistics.
        eval_details: JSON-serialised evaluation breakdown.
    """

    adapter_id: str
    base_model: str
    task_type: str
    adapter_path: str
    training_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    training_examples: int = 0
    epochs: int = 0
    eval_score: float = 0.0
    baseline_score: float = 0.0
    deployment_status: str = "pending"
    gguf_path: str = ""
    data_stats: str = "{}"
    eval_details: str = "{}"

    def __repr__(self) -> str:
        return (
            f"AdapterRecord(adapter_id={self.adapter_id!r}, "
            f"base_model={self.base_model!r}, task_type={self.task_type!r}, "
            f"status={self.deployment_status!r}, eval={self.eval_score:.3f})"
        )


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS adapter_registry (
    adapter_id       TEXT PRIMARY KEY,
    base_model       TEXT NOT NULL,
    task_type        TEXT NOT NULL,
    adapter_path     TEXT NOT NULL,
    training_date    TEXT NOT NULL,
    training_examples INTEGER DEFAULT 0,
    epochs           INTEGER DEFAULT 0,
    eval_score       REAL DEFAULT 0.0,
    baseline_score   REAL DEFAULT 0.0,
    deployment_status TEXT DEFAULT 'pending',
    gguf_path        TEXT DEFAULT '',
    data_stats       TEXT DEFAULT '{}',
    eval_details     TEXT DEFAULT '{}'
)
"""


class TrainingAdapterRegistry:
    """SQLite-backed registry tracking all trained adapters with full metadata.

    Stores one row per training run, recording the base model, data stats,
    evaluation results, and deployment status. Supports querying by task
    type, status, and date range for dashboard display and rollback decisions.

    Thread-safe: all database writes are serialised by an internal lock.
    Reads use individual connections (SQLite supports concurrent readers).
    """

    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the adapter_registry table if it does not exist."""
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        except Exception:
            logger.warning(
                "AdapterRegistry: could not create table — "
                "adapter tracking will be unavailable until database is accessible",
                exc_info=True,
            )

    def register(self, record: AdapterRecord) -> None:
        """Insert or update an adapter record in the registry.

        Uses INSERT OR REPLACE so re-registering the same adapter_id
        updates the existing row rather than failing.

        Args:
            record: The adapter metadata to persist.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            with self._write_lock:
                conn.execute(
                    """INSERT OR REPLACE INTO adapter_registry
                       (adapter_id, base_model, task_type, adapter_path,
                        training_date, training_examples, epochs, eval_score,
                        baseline_score, deployment_status, gguf_path,
                        data_stats, eval_details)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.adapter_id,
                        record.base_model,
                        record.task_type,
                        record.adapter_path,
                        record.training_date,
                        record.training_examples,
                        record.epochs,
                        record.eval_score,
                        record.baseline_score,
                        record.deployment_status,
                        record.gguf_path,
                        record.data_stats,
                        record.eval_details,
                    ),
                )
                conn.commit()
            logger.info(
                "AdapterRegistry: registered adapter %s (status=%s, eval=%.3f)",
                record.adapter_id,
                record.deployment_status,
                record.eval_score,
            )
        except Exception:
            logger.warning(
                "AdapterRegistry: failed to register adapter %s",
                record.adapter_id,
                exc_info=True,
            )

    def update_status(self, adapter_id: str, status: str) -> bool:
        """Update the deployment status of an adapter.

        Args:
            adapter_id: The adapter to update.
            status: New status — one of "deployed", "pending", "rolled_back", "rejected".

        Returns:
            True if the row was found and updated, False otherwise.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            with self._write_lock:
                cursor = conn.execute(
                    "UPDATE adapter_registry SET deployment_status = ? WHERE adapter_id = ?",
                    (status, adapter_id),
                )
                conn.commit()
                updated = cursor.rowcount > 0
            if updated:
                logger.info("AdapterRegistry: %s status -> %s", adapter_id, status)
            else:
                logger.warning("AdapterRegistry: adapter %s not found for status update", adapter_id)
            return updated
        except Exception:
            logger.warning(
                "AdapterRegistry: failed to update status for %s",
                adapter_id,
                exc_info=True,
            )
            return False

    def get(self, adapter_id: str) -> AdapterRecord | None:
        """Retrieve a single adapter record by ID.

        Args:
            adapter_id: The adapter to look up.

        Returns:
            AdapterRecord if found, None otherwise.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            row = conn.execute(
                "SELECT * FROM adapter_registry WHERE adapter_id = ?",
                (adapter_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except Exception:
            logger.warning(
                "AdapterRegistry: failed to get adapter %s",
                adapter_id,
                exc_info=True,
            )
            return None

    def list_by_task_type(self, task_type: str) -> list[AdapterRecord]:
        """List all adapters trained for a specific task type, newest first.

        Args:
            task_type: The task domain to filter by.

        Returns:
            List of AdapterRecord sorted by training_date descending.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM adapter_registry WHERE task_type = ? ORDER BY training_date DESC",
                (task_type,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception:
            logger.warning(
                "AdapterRegistry: failed to list adapters for task_type=%s",
                task_type,
                exc_info=True,
            )
            return []

    def list_deployed(self) -> list[AdapterRecord]:
        """List all currently deployed adapters.

        Returns:
            List of AdapterRecord with deployment_status == "deployed".
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM adapter_registry WHERE deployment_status = 'deployed' ORDER BY training_date DESC",
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception:
            logger.warning("AdapterRegistry: failed to list deployed adapters", exc_info=True)
            return []

    def list_all(self) -> list[AdapterRecord]:
        """List all adapter records, newest first.

        Returns:
            Complete list of AdapterRecord sorted by training_date descending.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM adapter_registry ORDER BY training_date DESC",
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception:
            logger.warning("AdapterRegistry: failed to list all adapters", exc_info=True)
            return []

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about registered adapters.

        Returns:
            Dictionary with total count, deployed count, task types,
            and average eval score.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            total = conn.execute("SELECT COUNT(*) FROM adapter_registry").fetchone()[0]
            deployed = conn.execute(
                "SELECT COUNT(*) FROM adapter_registry WHERE deployment_status = 'deployed'",
            ).fetchone()[0]
            avg_eval = conn.execute(
                "SELECT AVG(eval_score) FROM adapter_registry WHERE eval_score > 0",
            ).fetchone()[0]
            if avg_eval is None:
                avg_eval = 0.0
            task_types = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT task_type FROM adapter_registry ORDER BY task_type",
                ).fetchall()
            ]
            return {
                "total": total,
                "deployed": deployed,
                "avg_eval_score": round(avg_eval, 4),
                "task_types": task_types,
            }
        except Exception:
            logger.warning("AdapterRegistry: failed to get stats", exc_info=True)
            return {"total": 0, "deployed": 0, "avg_eval_score": 0.0, "task_types": []}

    @staticmethod
    def _row_to_record(row: Any) -> AdapterRecord:
        """Convert a SQLite row to an AdapterRecord.

        Args:
            row: A sqlite3.Row or tuple from a SELECT query.

        Returns:
            Populated AdapterRecord dataclass.
        """
        if hasattr(row, "keys"):
            # sqlite3.Row — access by column name
            return AdapterRecord(
                adapter_id=row["adapter_id"],
                base_model=row["base_model"],
                task_type=row["task_type"],
                adapter_path=row["adapter_path"],
                training_date=row["training_date"],
                training_examples=row["training_examples"],
                epochs=row["epochs"],
                eval_score=row["eval_score"],
                baseline_score=row["baseline_score"],
                deployment_status=row["deployment_status"],
                gguf_path=row["gguf_path"],
                data_stats=row["data_stats"],
                eval_details=row["eval_details"],
            )
        # Tuple — access by index (must match CREATE TABLE column order)
        return AdapterRecord(
            adapter_id=row[0],
            base_model=row[1],
            task_type=row[2],
            adapter_path=row[3],
            training_date=row[4],
            training_examples=row[5],
            epochs=row[6],
            eval_score=row[7],
            baseline_score=row[8],
            deployment_status=row[9],
            gguf_path=row[10],
            data_stats=row[11],
            eval_details=row[12],
        )


def get_adapter_registry() -> TrainingAdapterRegistry:
    """Return the singleton AdapterRegistry instance.

    Uses double-checked locking for thread safety.

    Returns:
        The shared AdapterRegistry instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TrainingAdapterRegistry()
    return _registry


def list_adapters_by_task_type(task_type: str) -> list[AdapterRecord]:
    """List all adapters trained for a specific task type, newest first.

    Convenience wrapper around ``get_adapter_registry().list_by_task_type()``.

    Args:
        task_type: The task domain to filter by (e.g. ``"coding"``, ``"analysis"``).

    Returns:
        List of AdapterRecord sorted by training_date descending.
    """
    return get_adapter_registry().list_by_task_type(task_type)


def list_deployed_adapters() -> list[AdapterRecord]:
    """List all currently deployed adapters.

    Convenience wrapper around ``get_adapter_registry().list_deployed()``.

    Returns:
        List of AdapterRecord with deployment_status == ``"deployed"``.
    """
    return get_adapter_registry().list_deployed()
