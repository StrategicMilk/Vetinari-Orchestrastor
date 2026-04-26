"""Tests for PipelineCheckpointStore — focused on list_traces D8/D9 fix correctness.

D8: list_traces (no since) orders by MAX(created_at) DESC, not MIN.
D9: list_traces (with since) uses HAVING MAX(created_at) >= ? so a trace whose
    LAST checkpoint is after the cutoff is included even when its FIRST checkpoint
    predates it.
"""

from __future__ import annotations

import sqlite3

import pytest

import vetinari.database as _db_module
from vetinari.observability.checkpoints import (
    PipelineCheckpointStore,
    reset_checkpoint_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_in_memory_conn() -> sqlite3.Connection:
    """Create an isolated in-memory SQLite connection with the pipeline_traces schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE pipeline_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            step_index INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'completed',
            input_snapshot_json TEXT NOT NULL DEFAULT '{}',
            output_snapshot_json TEXT NOT NULL DEFAULT '{}',
            tokens_used INTEGER NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0.0,
            model_id TEXT NOT NULL DEFAULT '',
            quality_score REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _insert_row(
    conn: sqlite3.Connection,
    trace_id: str,
    execution_id: str,
    step_name: str,
    step_index: int,
    created_at: str,
    tokens_used: int = 0,
    latency_ms: float = 0.0,
) -> None:
    """Insert a single pipeline_traces row for testing."""
    conn.execute(
        """
        INSERT INTO pipeline_traces
            (trace_id, execution_id, step_name, step_index, status,
             input_snapshot_json, output_snapshot_json,
             tokens_used, latency_ms, model_id, quality_score, created_at)
        VALUES (?, ?, ?, ?, 'completed', '{}', '{}', ?, ?, '', NULL, ?)
        """,
        (trace_id, execution_id, step_name, step_index, tokens_used, latency_ms, created_at),
    )
    conn.commit()


@pytest.fixture
def in_memory_store():
    """Yield (store, conn) with an isolated in-memory SQLite injected via thread-local cache.

    Injects the connection directly into _thread_local.connection so that
    `from vetinari.database import get_connection` inside list_traces returns
    our in-memory connection without touching the filesystem.
    """
    reset_checkpoint_store()
    conn = _make_in_memory_conn()

    # Inject into the thread-local cache; get_connection returns cached conn immediately.
    _db_module._thread_local.connection = conn
    # Mark schema as initialized so get_connection skips init_schema() on this conn.
    original_initialized = _db_module._schema_initialized
    _db_module._schema_initialized = True

    store = PipelineCheckpointStore()
    yield store, conn

    # Restore database state
    _db_module._thread_local.connection = None
    _db_module._schema_initialized = original_initialized
    reset_checkpoint_store()


# ---------------------------------------------------------------------------
# D8: ordering uses MAX(created_at), not MIN
# ---------------------------------------------------------------------------


class TestListTracesOrdering:
    """D8: list_traces without 'since' must order by MAX(created_at) DESC."""

    def test_older_trace_first_when_its_last_step_is_newest(self, in_memory_store) -> None:
        """Trace A started before trace B, but its LAST step is newest — A must rank first.

        Without D8 fix: MIN(created_at) used → B ranks first (B.step-0 > A.step-0).
        With D8 fix: MAX(created_at) used → A ranks first (A.step-1 is newest overall).
        """
        store, conn = in_memory_store

        # Trace A: two steps — first at T=1, last at T=4 (newest checkpoint overall)
        _insert_row(conn, "trace-A", "exec-A", "intake", 0, "2024-01-01T00:00:01Z")
        _insert_row(conn, "trace-A", "exec-A", "worker", 1, "2024-01-01T00:00:04Z")

        # Trace B: one step at T=3 (newer than A.step-0 but older than A.step-1)
        _insert_row(conn, "trace-B", "exec-B", "intake", 0, "2024-01-01T00:00:03Z")

        traces = store.list_traces()

        assert len(traces) == 2
        assert traces[0]["trace_id"] == "trace-A", (
            "D8 fix: trace-A has newest MAX(created_at) so must rank first"
        )
        assert traces[1]["trace_id"] == "trace-B"

    def test_step_count_aggregation_is_correct(self, in_memory_store) -> None:
        """step_count must equal the number of rows per trace, not 1."""
        store, conn = in_memory_store

        _insert_row(conn, "trace-C", "exec-C", "intake", 0, "2024-01-01T00:01:00Z")
        _insert_row(conn, "trace-C", "exec-C", "planning", 1, "2024-01-01T00:01:01Z")
        _insert_row(conn, "trace-C", "exec-C", "worker", 2, "2024-01-01T00:01:02Z")

        traces = store.list_traces()

        assert len(traces) == 1
        assert traces[0]["step_count"] == 3, "step_count must count all rows for the trace"


# ---------------------------------------------------------------------------
# D9: 'since' filter uses HAVING MAX(created_at) >= ?
# ---------------------------------------------------------------------------


class TestListTracesSinceFilter:
    """D9: list_traces with 'since' must include traces whose LAST checkpoint is after cutoff."""

    def test_trace_included_when_last_checkpoint_after_cutoff(self, in_memory_store) -> None:
        """A trace that STARTED before the cutoff must still be included if its last step is after.

        Without D9 fix: WHERE created_at >= ? filters row-level → only rows after cutoff survive
        the WHERE clause, so a trace can be excluded or have wrong step_count.
        With D9 fix: HAVING MAX(created_at) >= ? → the full trace is included if any row is after.
        """
        store, conn = in_memory_store
        cutoff = "2024-01-01T00:05:00Z"

        # Trace X: started before cutoff at T=1, finished after cutoff at T=10
        _insert_row(conn, "trace-X", "exec-X", "intake", 0, "2024-01-01T00:01:00Z")
        _insert_row(conn, "trace-X", "exec-X", "worker", 1, "2024-01-01T00:10:00Z")

        # Trace Y: entirely before the cutoff — must be excluded
        _insert_row(conn, "trace-Y", "exec-Y", "intake", 0, "2024-01-01T00:02:00Z")
        _insert_row(conn, "trace-Y", "exec-Y", "worker", 1, "2024-01-01T00:03:00Z")

        traces = store.list_traces(since=cutoff)

        trace_ids = [t["trace_id"] for t in traces]
        assert "trace-X" in trace_ids, (
            "D9 fix: trace-X must be included because its last checkpoint (T=10) is after cutoff"
        )
        assert "trace-Y" not in trace_ids, (
            "trace-Y must be excluded — all its checkpoints precede the cutoff"
        )

    def test_trace_excluded_when_all_checkpoints_before_cutoff(self, in_memory_store) -> None:
        """A trace with all checkpoints before the cutoff must not appear in results."""
        store, conn = in_memory_store
        cutoff = "2024-02-01T00:00:00Z"

        _insert_row(conn, "trace-old", "exec-old", "intake", 0, "2024-01-01T00:00:01Z")
        _insert_row(conn, "trace-old", "exec-old", "worker", 1, "2024-01-01T00:00:02Z")

        traces = store.list_traces(since=cutoff)

        assert traces == [], "trace-old must not appear — all its checkpoints predate the cutoff"

    def test_since_filter_step_count_includes_all_steps(self, in_memory_store) -> None:
        """When a trace passes the HAVING filter, its step_count must include ALL rows, not just post-cutoff ones.

        This is the key correctness guarantee of the D9 fix: HAVING operates post-GROUP BY,
        so aggregates include all rows for the trace.
        """
        store, conn = in_memory_store
        cutoff = "2024-01-01T00:05:00Z"

        # Three steps: first two before cutoff, last one after
        _insert_row(conn, "trace-multi", "exec-m", "intake", 0, "2024-01-01T00:01:00Z")
        _insert_row(conn, "trace-multi", "exec-m", "planning", 1, "2024-01-01T00:02:00Z")
        _insert_row(conn, "trace-multi", "exec-m", "worker", 2, "2024-01-01T00:10:00Z")

        traces = store.list_traces(since=cutoff)

        assert len(traces) == 1
        assert traces[0]["trace_id"] == "trace-multi"
        assert traces[0]["step_count"] == 3, (
            "D9 fix: step_count must be 3 (all steps), not 1 (only post-cutoff step)"
        )
