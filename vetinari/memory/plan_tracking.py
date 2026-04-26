"""Plan execution tracking store.

Tracks plan execution history, subtask outcomes, and model performance
metrics in SQLite with a JSON fallback.  Used by ``FeedbackLoop`` and
``PlanModeEngine``.

Note: This is *not* the same system as ``UnifiedMemoryStore`` (agent
episodic/semantic memory).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import _PROJECT_ROOT
from vetinari.database import get_connection
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)

PLAN_RETENTION_DAYS = int(os.environ.get("PLAN_RETENTION_DAYS", "90"))
PLAN_ADMIN_TOKEN = os.environ.get("PLAN_ADMIN_TOKEN", "")


class MemoryStore:
    """Plan execution tracking store (PlanHistory, SubtaskMemory, ModelPerformance).

    Note: This is *not* the same system as DualMemoryStore (agent episodic/semantic
    memory).  MemoryStore tracks plan execution history, subtask outcomes, and
    model performance metrics in SQLite/JSON.  Used by FeedbackLoop and PlanModeEngine.
    """

    def __init__(self, db_path: str | None = None, use_json_fallback: bool = False):
        self.use_json_fallback = use_json_fallback
        self._lock = threading.Lock()
        # JSON fallback path: alongside the unified DB or a custom path if provided
        if db_path:
            self._json_path = db_path.replace(".db", ".json")
        else:
            self._json_path = str(_PROJECT_ROOT / ".vetinari" / "vetinari_memory.json")

        if use_json_fallback:
            self._init_json_store()
        else:
            self._init_sqlite()

    def _init_sqlite(self):
        try:
            conn = get_connection()

            conn.execute("""
                CREATE TABLE IF NOT EXISTS PlanHistory (
                    plan_id TEXT PRIMARY KEY,
                    plan_version INTEGER DEFAULT 1,
                    goal TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'draft',
                    plan_json TEXT,
                    plan_explanation_json TEXT,
                    chosen_plan_id TEXT,
                    plan_justification TEXT,
                    risk_score REAL DEFAULT 0.0,
                    dry_run BOOLEAN DEFAULT 0,
                    auto_approved BOOLEAN DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS SubtaskMemory (
                    subtask_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    parent_subtask_id TEXT,
                    description TEXT NOT NULL,
                    depth INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    assigned_model_id TEXT,
                    outcome TEXT,
                    duration_seconds REAL,
                    cost_estimate REAL,
                    rationale TEXT,
                    subtask_explanation_json TEXT,
                    domain TEXT,
                    quality_score REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (plan_id) REFERENCES PlanHistory(plan_id)
                )
            """)

            # Migration: add quality_score column to existing databases
            plan_columns = {row[1] for row in conn.execute("PRAGMA table_info(PlanHistory)").fetchall()}
            if "plan_explanation_json" not in plan_columns:
                conn.execute("ALTER TABLE PlanHistory ADD COLUMN plan_explanation_json TEXT")

            subtask_columns = {row[1] for row in conn.execute("PRAGMA table_info(SubtaskMemory)").fetchall()}
            if "subtask_explanation_json" not in subtask_columns:
                conn.execute("ALTER TABLE SubtaskMemory ADD COLUMN subtask_explanation_json TEXT")
            if "quality_score" not in subtask_columns:
                conn.execute("ALTER TABLE SubtaskMemory ADD COLUMN quality_score REAL DEFAULT 0.0")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS ModelPerformance (
                    model_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    avg_latency REAL DEFAULT 0.0,
                    total_uses INTEGER DEFAULT 0,
                    last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_id, task_type)
                )
            """)

            conn.commit()
            logger.info("Memory store initialized (unified database)")

        except sqlite3.Error as e:
            logger.warning("SQLite initialization failed: %s. Falling back to JSON.", e)
            self._init_json_store()

    def _init_json_store(self):
        self.use_json_fallback = True
        if not Path(self._json_path).exists():
            self._json_data = {"plans": {}, "subtasks": {}, "model_performance": {}}
            self._save_json()
        else:
            try:
                with Path(self._json_path).open(encoding="utf-8") as f:
                    loaded = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "JSON fallback memory store at %s is unreadable or corrupt; starting with empty in-memory data",
                    self._json_path,
                )
                logger.debug("JSON fallback load failure: %s", exc)
                loaded = {}
            self._json_data = {
                "plans": dict(loaded.get("plans", {})) if isinstance(loaded, dict) else {},
                "subtasks": dict(loaded.get("subtasks", {})) if isinstance(loaded, dict) else {},
                "model_performance": dict(loaded.get("model_performance", {})) if isinstance(loaded, dict) else {},
            }
        logger.info("JSON fallback memory store initialized at %s", self._json_path)

    def _save_json(self):
        path = Path(self._json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)

    def write_plan_history(self, plan_data: dict[str, Any]) -> bool:
        """Write plan history.

        Returns:
            True if successful, False otherwise.
        """
        if self.use_json_fallback:
            return self._write_plan_json(plan_data)

        try:
            conn = get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO PlanHistory
                (plan_id, plan_version, goal, updated_at, status, plan_json, plan_explanation_json,
                 chosen_plan_id, plan_justification, risk_score, dry_run, auto_approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plan_data.get("plan_id"),
                    plan_data.get("plan_version", 1),
                    plan_data.get("goal"),
                    datetime.now(timezone.utc).isoformat(),
                    plan_data.get("status", "draft"),
                    json.dumps(plan_data.get("plan_json", {})),
                    plan_data.get("plan_explanation_json", ""),
                    plan_data.get("chosen_plan_id"),
                    plan_data.get("plan_justification"),
                    plan_data.get("risk_score", 0.0),
                    plan_data.get("dry_run", False),
                    plan_data.get("auto_approved", False),
                ),
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error("Failed to write plan: %s", e)
            return False

    def _write_plan_json(self, plan_data: dict[str, Any]) -> bool:
        plan_id = plan_data.get("plan_id")
        self._json_data["plans"][plan_id] = {**plan_data, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._save_json()
        return True

    def write_subtask_memory(self, subtask_data: dict[str, Any]) -> bool:
        """Write subtask memory.

        Returns:
            True if successful, False otherwise.
        """
        if self.use_json_fallback:
            return self._write_subtask_json(subtask_data)

        try:
            conn = get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO SubtaskMemory
                (subtask_id, plan_id, parent_subtask_id, description, depth,
                 status, assigned_model_id, outcome, duration_seconds,
                 cost_estimate, rationale, subtask_explanation_json, domain,
                 quality_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    subtask_data.get("subtask_id"),
                    subtask_data.get("plan_id"),
                    subtask_data.get("parent_subtask_id"),
                    subtask_data.get("description"),
                    subtask_data.get("depth", 0),
                    subtask_data.get("status", StatusEnum.PENDING.value),
                    subtask_data.get("assigned_model_id"),
                    subtask_data.get("outcome"),
                    subtask_data.get("duration_seconds"),
                    subtask_data.get("cost_estimate"),
                    subtask_data.get("rationale"),
                    subtask_data.get("subtask_explanation_json", ""),
                    subtask_data.get("domain"),
                    subtask_data.get("quality_score"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error("Failed to write subtask: %s", e)
            return False

    def _write_subtask_json(self, subtask_data: dict[str, Any]) -> bool:
        subtask_id = subtask_data.get("subtask_id")
        self._json_data["subtasks"][subtask_id] = {**subtask_data, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._save_json()
        return True

    def query_plan_history(
        self,
        plan_id: str | None = None,
        goal_contains: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query plan history.

        Args:
            plan_id: The plan id.
            goal_contains: The goal contains.
            limit: The limit.

        Returns:
            List of matching plan history records.
        """
        if self.use_json_fallback:
            return self._query_plan_json(plan_id, goal_contains, limit)

        try:
            conn = get_connection()

            if plan_id:
                rows = conn.execute(
                    """
                    SELECT * FROM PlanHistory
                    WHERE plan_id = ?
                    ORDER BY created_at DESC
                """,
                    (plan_id,),
                ).fetchall()
            elif goal_contains:
                rows = conn.execute(
                    """
                    SELECT * FROM PlanHistory
                    WHERE goal LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (f"%{goal_contains}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM PlanHistory
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()

            return [dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error("Failed to query plans: %s", e)
            return []

    def _query_plan_json(self, plan_id: str | None, goal_contains: str | None, limit: int) -> list[dict]:
        plans = list(self._json_data["plans"].values())

        if plan_id:
            plans = [p for p in plans if p.get("plan_id") == plan_id]
        elif goal_contains:
            plans = [p for p in plans if goal_contains.lower() in p.get("goal", "").lower()]

        plans.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return plans[:limit]

    def query_subtasks(
        self,
        plan_id: str | None = None,
        subtask_id: str | None = None,
        depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query subtasks.

        Args:
            plan_id: The plan id.
            subtask_id: The subtask id.
            depth: The depth.

        Returns:
            List of matching subtask records.
        """
        if self.use_json_fallback:
            return self._query_subtasks_json(plan_id, subtask_id, depth)

        try:
            conn = get_connection()

            if subtask_id:
                rows = conn.execute(
                    """
                    SELECT * FROM SubtaskMemory
                    WHERE subtask_id = ?
                """,
                    (subtask_id,),
                ).fetchall()
            elif plan_id:
                if depth is not None:
                    rows = conn.execute(
                        """
                        SELECT * FROM SubtaskMemory
                        WHERE plan_id = ? AND depth = ?
                        ORDER BY depth, subtask_id
                    """,
                        (plan_id, depth),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM SubtaskMemory
                        WHERE plan_id = ?
                        ORDER BY depth, subtask_id
                    """,
                        (plan_id,),
                    ).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM SubtaskMemory
                    ORDER BY created_at DESC
                    LIMIT 100
                """).fetchall()

            return [dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error("Failed to query subtasks: %s", e)
            return []

    def _query_subtasks_json(self, plan_id: str | None, subtask_id: str | None, depth: int | None) -> list[dict]:
        subtasks = list(self._json_data["subtasks"].values())

        if subtask_id:
            subtasks = [s for s in subtasks if s.get("subtask_id") == subtask_id]
        elif plan_id:
            subtasks = [s for s in subtasks if s.get("plan_id") == plan_id]
            if depth is not None:
                subtasks = [s for s in subtasks if s.get("depth") == depth]

        return subtasks

    def get_model_performance(self, model_id: str, task_type: str) -> dict[str, Any] | None:
        """Retrieve model performance record for a given model and task type.

        Args:
            model_id: The model id.
            task_type: The task type.

        Returns:
            Performance record dict, or None if not found.
        """
        if self.use_json_fallback:
            key = f"{model_id}:{task_type}"
            return self._json_data.get("model_performance", {}).get(key)
        try:
            conn = get_connection()
            row = conn.execute(
                "SELECT * FROM ModelPerformance WHERE model_id = ? AND task_type = ?",
                (model_id, task_type),
            ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.warning("get_model_performance failed: %s", e)
            return None

    def update_subtask_quality(self, subtask_id: str, quality_score: float = 0.0, succeeded: bool = True) -> bool:
        """Annotate a SubtaskMemory record with a quality score and outcome.

        Args:
            subtask_id: The subtask id.
            quality_score: The quality score.
            succeeded: Whether the subtask succeeded.

        Returns:
            True if successful, False otherwise.
        """
        if self.use_json_fallback:
            subtask = self._json_data.get("subtasks", {}).get(subtask_id, {})
            if subtask:
                subtask["quality_score"] = quality_score
                subtask["outcome"] = StatusEnum.COMPLETED.value if succeeded else StatusEnum.FAILED.value
                subtask["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_json()
                return True
            logger.warning(
                "update_subtask_quality: subtask %s not found in JSON store — quality update skipped",
                subtask_id,
            )
            return False
        try:
            conn = get_connection()
            cursor = conn.execute(
                """UPDATE SubtaskMemory
                   SET outcome = ?, quality_score = ?, updated_at = ?
                   WHERE subtask_id = ?""",
                (
                    StatusEnum.COMPLETED.value if succeeded else StatusEnum.FAILED.value,
                    quality_score,
                    datetime.now(timezone.utc).isoformat(),
                    subtask_id,
                ),
            )
            conn.commit()
            if cursor.rowcount == 0:
                logger.warning(
                    "update_subtask_quality: subtask %s not found in SubtaskMemory — quality update skipped",
                    subtask_id,
                )
                return False
            return True
        except sqlite3.Error as e:
            logger.warning("update_subtask_quality failed: %s", e)
            return False

    def update_model_performance(
        self,
        model_id: str,
        task_type: str,
        success_or_dict=None,
        latency: float = 0.0,
    ) -> bool:
        """Update model performance metrics.

        Accepts two call signatures:
          - Alternative: update_model_performance(model_id, task_type, success: bool, latency: float)
          - New:         update_model_performance(model_id, task_type, data: dict)

        Args:
            model_id: The model id.
            task_type: The task type.
            success_or_dict: Boolean success flag or dict with performance data.
            latency: The latency in seconds.

        Returns:
            True if successful, False otherwise.
        """
        if isinstance(success_or_dict, dict):
            data = success_or_dict
            success = data.get("success_rate", 1.0) >= 0.5
            latency = float(data.get("avg_latency", latency))
        elif success_or_dict is None:
            success = True
        else:
            success = bool(success_or_dict)

        if self.use_json_fallback:
            return self._update_model_perf_json(model_id, task_type, success, latency)

        try:
            conn = get_connection()

            row = conn.execute(
                """
                SELECT * FROM ModelPerformance
                WHERE model_id = ? AND task_type = ?
            """,
                (model_id, task_type),
            ).fetchone()

            if row:
                new_success_rate = (row["success_rate"] * row["total_uses"] + (1 if success else 0)) / (
                    row["total_uses"] + 1
                )
                new_latency = (row["avg_latency"] * row["total_uses"] + latency) / (row["total_uses"] + 1)
                new_uses = row["total_uses"] + 1

                conn.execute(
                    """
                    UPDATE ModelPerformance
                    SET success_rate = ?, avg_latency = ?,
                        total_uses = ?, last_used_at = ?
                    WHERE model_id = ? AND task_type = ?
                """,
                    (
                        new_success_rate,
                        new_latency,
                        new_uses,
                        datetime.now(timezone.utc).isoformat(),
                        model_id,
                        task_type,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO ModelPerformance
                    (model_id, task_type, success_rate, avg_latency, total_uses, last_used_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                """,
                    (model_id, task_type, 1.0 if success else 0.0, latency, datetime.now(timezone.utc).isoformat()),
                )

            conn.commit()
            return True

        except sqlite3.Error as e:
            logger.error("Failed to update model performance: %s", e)
            return False

    def _update_model_perf_json(self, model_id: str, task_type: str, success: bool, latency: float) -> bool:
        key = f"{model_id}:{task_type}"
        if key not in self._json_data["model_performance"]:
            self._json_data["model_performance"][key] = {
                "model_id": model_id,
                "task_type": task_type,
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "total_uses": 0,
            }

        perf = self._json_data["model_performance"][key]
        total = perf["total_uses"] + 1
        perf["success_rate"] = (perf["success_rate"] * perf["total_uses"] + (1 if success else 0)) / total
        perf["avg_latency"] = (perf["avg_latency"] * perf["total_uses"] + latency) / total
        perf["total_uses"] = total
        perf["last_used_at"] = datetime.now(timezone.utc).isoformat()

        self._save_json()
        return True

    def prune_old_plans(self, retention_days: int = PLAN_RETENTION_DAYS) -> int:
        """Remove plan records older than *retention_days*.

        Returns:
            Number of pruned plans.
        """
        if self.use_json_fallback:
            return self._prune_old_json(retention_days)

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            conn = get_connection()

            old_plans = [
                row[0]
                for row in conn.execute(
                    "SELECT plan_id FROM PlanHistory WHERE created_at < ?",
                    (cutoff.isoformat(),),
                ).fetchall()
            ]

            for plan_id in old_plans:
                conn.execute("DELETE FROM SubtaskMemory WHERE plan_id = ?", (plan_id,))
                conn.execute("DELETE FROM PlanHistory WHERE plan_id = ?", (plan_id,))

            conn.commit()
            logger.info("Pruned %d old plans", len(old_plans))
            return len(old_plans)

        except sqlite3.Error as e:
            logger.error("Failed to prune old plans: %s", e)
            return 0

    def _prune_old_json(self, retention_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_str = cutoff.isoformat()

        to_delete = [pid for pid, p in self._json_data["plans"].items() if p.get("created_at", "") < cutoff_str]

        for pid in to_delete:
            del self._json_data["plans"][pid]
            self._json_data["subtasks"] = {
                sid: s for sid, s in self._json_data["subtasks"].items() if s.get("plan_id") != pid
            }

        self._save_json()
        return len(to_delete)

    def close(self) -> None:
        """Close the underlying database connection and release resources.

        Delegates to the unified database module's thread-local connection
        management. After calling this, the next operation will re-open the
        connection via ``get_connection()``.
        """
        from vetinari.database import close_connection

        with contextlib.suppress(Exception):
            close_connection()

    def __del__(self) -> None:
        """Safety-net cleanup if close() was not called explicitly."""
        contextlib_module = globals().get("contextlib")
        suppress = getattr(contextlib_module, "suppress", None)
        if suppress is None:
            return
        with suppress(Exception):
            self.close()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get aggregate statistics about stored plan data.

        Returns:
            Dict with plan/subtask/model counts and storage type.
        """
        if self.use_json_fallback:
            return {
                "total_plans": len(self._json_data["plans"]),
                "total_subtasks": len(self._json_data["subtasks"]),
                "total_model_records": len(self._json_data["model_performance"]),
                "storage_type": "json",
            }

        try:
            conn = get_connection()
            plan_count = conn.execute("SELECT COUNT(*) FROM PlanHistory").fetchone()[0]
            subtask_count = conn.execute("SELECT COUNT(*) FROM SubtaskMemory").fetchone()[0]
            model_count = conn.execute("SELECT COUNT(*) FROM ModelPerformance").fetchone()[0]

            return {
                "total_plans": plan_count,
                "total_subtasks": subtask_count,
                "total_model_records": model_count,
                "storage_type": "sqlite",
            }

        except sqlite3.Error as e:
            logger.error("Failed to get memory stats: %s", e)
            return {}


# ── Singleton management ──────────────────────────────────────────────

_memory_store: MemoryStore | None = None
_memory_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store instance.

    Returns:
        The MemoryStore singleton.
    """
    global _memory_store
    if _memory_store is None:
        with _memory_store_lock:
            if _memory_store is None:
                use_json = os.environ.get("PLAN_USE_JSON_FALLBACK", "false").lower() in ("1", "true", "yes")
                _memory_store = MemoryStore(use_json_fallback=use_json)
    return _memory_store


def init_memory_store(db_path: str | None = None, use_json_fallback: bool = False) -> MemoryStore:
    """Initialize a new memory store instance.

    Args:
        db_path: Ignored — all data goes to the unified database. Retained
            for backward-compatibility only.
        use_json_fallback: If True, use JSON storage instead of SQLite.

    Returns:
        The newly created MemoryStore instance.
    """
    global _memory_store
    _memory_store = MemoryStore(use_json_fallback=use_json_fallback)  # noqa: VET111 — global side-effect required
    return _memory_store
