"""Cost Tracker (C8).

==================
SQLite-backed cost tracking for every LLM inference call.

Records model_id, agent_type, token counts, estimated cost, and timestamp.
Provides query methods for cost breakdowns by model, agent, and time period.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("VETINARI_COST_DB", "./vetinari_costs.db")

# ── Model pricing (per 1K tokens) ────────────────────────────────────

MODEL_COST_PER_1K: dict[str, float] = {
    # Local models — free
    "qwen2.5-coder-7b": 0.0,
    "qwen2.5-coder-14b": 0.0,
    "qwen2.5-coder-32b": 0.0,
    "qwen3-32b": 0.0,
    "qwen3-vl-32b": 0.0,
    "llama-3.3-70b": 0.0,
    "gemma-3-27b": 0.0,
    "phi-4-14b": 0.0,
    "deepseek-r1-14b": 0.0,
    "default": 0.0,
    # Cloud models
    "claude-opus-4": 0.075,
    "claude-sonnet-4": 0.015,
    "claude-haiku-3.5": 0.004,
    "gpt-4o": 0.025,
    "gpt-4o-mini": 0.0075,
    "gemini-2.5-pro": 0.03125,
    "gemini-2.5-flash": 0.0075,
}


@dataclass
class CostRecord:
    """A single cost tracking record."""

    timestamp: str
    model_id: str
    agent_type: str
    mode: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    task_id: str
    duration_ms: float


class CostTracker:
    """Tracks LLM inference costs in SQLite.

    Thread-safe singleton.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    agent_type TEXT DEFAULT '',
                    mode TEXT DEFAULT '',
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    task_id TEXT DEFAULT '',
                    duration_ms REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_timestamp ON cost_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_model ON cost_records(model_id)
            """)
            conn.commit()
            conn.close()

    def record(
        self,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        agent_type: str = "",
        mode: str = "",
        task_id: str = "",
        duration_ms: float = 0.0,
    ) -> CostRecord:
        """Record an LLM inference cost.

        Args:
            model_id: The model id.
            input_tokens: The input tokens.
            output_tokens: The output tokens.
            agent_type: The agent type.
            mode: The mode.
            task_id: The task id.
            duration_ms: The duration ms.

        Returns:
            The result string.
        """
        total_tokens = input_tokens + output_tokens
        cost_per_1k = MODEL_COST_PER_1K.get(model_id, 0.0)
        estimated_cost = (total_tokens / 1000.0) * cost_per_1k

        record = CostRecord(
            timestamp=datetime.utcnow().isoformat(),
            model_id=model_id,
            agent_type=agent_type,
            mode=mode,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            task_id=task_id,
            duration_ms=duration_ms,
        )

        with self._lock:
            self._total_cost += estimated_cost
            self._total_tokens += total_tokens
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    """INSERT INTO cost_records
                       (timestamp, model_id, agent_type, mode, input_tokens,
                        output_tokens, total_tokens, estimated_cost, task_id, duration_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.timestamp,
                        model_id,
                        agent_type,
                        mode,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        estimated_cost,
                        task_id,
                        duration_ms,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning("Cost tracking write failed: %s", e)

        return record

    def get_total_cost(self) -> float:
        """Get total cumulative cost."""
        return self._total_cost

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self._total_tokens

    def get_cost_by_model(self, hours: int = 24) -> dict[str, float]:
        """Get cost breakdown by model for the last N hours.

        Returns:
            The result string.
        """
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                rows = conn.execute(
                    """SELECT model_id, SUM(estimated_cost), SUM(total_tokens)
                       FROM cost_records WHERE timestamp >= ?
                       GROUP BY model_id ORDER BY SUM(estimated_cost) DESC""",
                    (since,),
                ).fetchall()
                conn.close()
                return {r[0]: {"cost": r[1], "tokens": r[2]} for r in rows}
            except Exception as e:
                logger.warning("Cost query failed: %s", e)
                return {}

    def get_cost_by_agent(self, hours: int = 24) -> dict[str, float]:
        """Get cost breakdown by agent type for the last N hours.

        Returns:
            The result string.
        """
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._lock:
            try:
                conn = sqlite3.connect(self._db_path)
                rows = conn.execute(
                    """SELECT agent_type, SUM(estimated_cost), SUM(total_tokens), COUNT(*)
                       FROM cost_records WHERE timestamp >= ?
                       GROUP BY agent_type ORDER BY SUM(estimated_cost) DESC""",
                    (since,),
                ).fetchall()
                conn.close()
                return {r[0]: {"cost": r[1], "tokens": r[2], "calls": r[3]} for r in rows}
            except Exception as e:
                logger.warning("Cost query failed: %s", e)
                return {}

    def get_summary(self) -> dict[str, Any]:
        """Dashboard-friendly summary."""
        return {
            "total_cost": self._total_cost,
            "total_tokens": self._total_tokens,
            "by_model": self.get_cost_by_model(),
            "by_agent": self.get_cost_by_agent(),
        }


# ── Singleton ─────────────────────────────────────────────────────────

_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get cost tracker.

    Returns:
        The CostTracker result.
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
