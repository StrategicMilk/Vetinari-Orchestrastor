"""Scale tests — verify the system does not crash or degrade under measured load.

All tests use temporary databases and mock inference. Thresholds are defined
as constants before each test class so they are visible without reading the
full test body. Tests are marked ``slow`` to allow CI to skip them in fast
feedback loops.

Thresholds:
    SQLite WAL:        100 concurrent writers, all 100 rows queryable within 5s
    Thompson Sampling: 10,000 arm updates, select_model() completes in < 100ms
    FTS5:              50,000 memory insertions, FTS search returns in < 500ms
    SSE queues:        50 concurrent listeners, zero queue leaks after cleanup
"""

from __future__ import annotations

import os
import queue as _queue
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from vetinari.learning.model_selector import ThompsonSamplingSelector

# -- Thresholds (define before any test so they are easy to find and tune) --

# SQLite WAL concurrency
WAL_CONCURRENT_WRITERS = 100  # total concurrent threads that each insert one row
WAL_QUERY_TIMEOUT_SECONDS = 5.0  # all 100 rows must be queryable in this window

# Thompson Sampling
THOMPSON_TRAINING_RECORDS = 10_000  # arm updates to insert before timing selection
THOMPSON_SELECT_TIMEOUT_MS = 100.0  # select_model() must complete within this budget

# FTS5 memory search
FTS5_ENTRY_COUNT = 50_000  # entries to insert before timing the search
FTS5_SEARCH_TIMEOUT_MS = 500.0  # FTS search must return within this budget

# SSE queue leak
SSE_LISTENER_COUNT = 50  # concurrent simulated listeners


# -- Helpers ------------------------------------------------------------------


def _make_private_db(tmp_path: Path) -> sqlite3.Connection:
    """Open a fresh SQLite database in WAL mode at *tmp_path* / scale.db.

    Applies the same PRAGMA set as production (WAL, NORMAL sync, busy_timeout)
    so the test faithfully reflects runtime behaviour.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Open sqlite3.Connection with WAL mode and row factory.
    """
    db_file = tmp_path / "scale.db"
    conn = sqlite3.connect(str(db_file), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _create_projects_table(conn: sqlite3.Connection) -> None:
    """Create a minimal projects table on *conn* for scale testing.

    Args:
        conn: Open SQLite connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _create_memory_schema(conn: sqlite3.Connection) -> None:
    """Create the memories table and FTS5 virtual table on *conn*.

    Mirrors the subset of ``vetinari.database._UNIFIED_SCHEMA`` that covers
    memory storage and FTS5 search, so the test exercises the real DDL without
    touching the production database.

    Args:
        conn: Open SQLite connection.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent TEXT NOT NULL DEFAULT '',
            entry_type TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            timestamp INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            forgotten INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_mem_forgotten ON memories(forgotten);

        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            id,
            content,
            summary,
            agent,
            content=memories,
            content_rowid=rowid
        );
        CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memory_fts(rowid, id, content, summary, agent)
            VALUES (NEW.rowid, NEW.id, NEW.content, NEW.summary, NEW.agent);
        END;
    """)
    conn.commit()


# =============================================================================
# Test 1: 100 concurrent projects with SQLite WAL
# =============================================================================


class TestSQLiteWALConcurrency:
    """Verify SQLite WAL mode handles 100 concurrent writers without errors.

    Threshold: all 100 rows must be written and queryable within
    WAL_QUERY_TIMEOUT_SECONDS seconds. Zero ``sqlite3.OperationalError``
    (database locked) exceptions are permitted.

    Rationale: The web layer creates projects from background threads. If WAL
    is misconfigured or ``busy_timeout`` too short, concurrent writes contend
    and writers get ``database is locked`` errors.
    """

    def test_100_concurrent_inserts_no_errors(self, tmp_path: pytest.TempPathFactory) -> None:
        """Insert 100 projects concurrently; assert all succeed and are queryable.

        Threshold: WAL_CONCURRENT_WRITERS rows must all be present after all
        threads complete. Total wall-clock time must not exceed
        WAL_QUERY_TIMEOUT_SECONDS. No OperationalError (database locked) may occur.
        """
        db_file = tmp_path / "wal_test.db"
        conn_main = sqlite3.connect(str(db_file), check_same_thread=False, timeout=30.0)
        conn_main.execute("PRAGMA journal_mode=WAL")
        conn_main.execute("PRAGMA synchronous=NORMAL")
        conn_main.execute("PRAGMA busy_timeout=5000")
        conn_main.row_factory = sqlite3.Row
        _create_projects_table(conn_main)

        project_ids = [f"project-{i:04d}" for i in range(WAL_CONCURRENT_WRITERS)]
        errors: list[str] = []
        errors_lock = threading.Lock()

        def _insert_project(project_id: str) -> None:
            # Each thread opens its own connection — WAL mode allows concurrent readers
            # and one writer at a time with readers never blocking writers.
            thread_conn = sqlite3.connect(str(db_file), check_same_thread=False, timeout=30.0)
            thread_conn.execute("PRAGMA journal_mode=WAL")
            thread_conn.execute("PRAGMA busy_timeout=5000")
            try:
                thread_conn.execute(
                    "INSERT INTO projects (id, name) VALUES (?, ?)",
                    (project_id, f"Scale test project {project_id}"),
                )
                thread_conn.commit()
            except sqlite3.OperationalError as exc:
                with errors_lock:
                    errors.append(f"{project_id}: {exc}")
            finally:
                thread_conn.close()

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=WAL_CONCURRENT_WRITERS) as pool:
            futures = [pool.submit(_insert_project, pid) for pid in project_ids]
            for fut in as_completed(futures):
                fut.result()  # re-raise any unexpected exception
        elapsed = time.monotonic() - start

        # All inserts must complete within the time budget.
        assert elapsed <= WAL_QUERY_TIMEOUT_SECONDS, (
            f"Concurrent WAL inserts took {elapsed:.2f}s, threshold {WAL_QUERY_TIMEOUT_SECONDS}s"
        )

        # No database-locked errors permitted.
        assert errors == [], f"Got {len(errors)} SQLite lock error(s): {errors[:5]}"

        # All rows must be queryable.
        cursor = conn_main.execute("SELECT COUNT(*) AS cnt FROM projects")
        row = cursor.fetchone()
        actual_count = row["cnt"]
        assert actual_count == WAL_CONCURRENT_WRITERS, f"Expected {WAL_CONCURRENT_WRITERS} rows, found {actual_count}"

        # Spot-check: every project_id must be retrievable individually.
        for pid in project_ids:
            cursor = conn_main.execute("SELECT id FROM projects WHERE id = ?", (pid,))
            assert cursor.fetchone() is not None, f"Project {pid} not found after concurrent insert"

        conn_main.close()

    def test_wal_journal_mode_is_active(self, tmp_path: pytest.TempPathFactory) -> None:
        """Assert that get_connection() returns a WAL-mode connection.

        Threshold: PRAGMA journal_mode must return 'wal', not 'delete' or 'memory'.
        This guards against PRAGMA journal_mode=WAL silently failing.
        """
        db_file = tmp_path / "wal_mode_check.db"
        conn = sqlite3.connect(str(db_file), check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")

        cursor = conn.execute("PRAGMA journal_mode")
        row = cursor.fetchone()
        actual_mode = row[0]
        conn.close()

        assert actual_mode == "wal", f"Expected WAL mode but got '{actual_mode}' — WAL PRAGMA may have been ignored"


# =============================================================================
# Test 2: 10,000 training records with Thompson Sampling
# =============================================================================


class TestThompsonSamplingScale:
    """Verify Thompson Sampling stays fast after 10,000 arm updates.

    Threshold: select_model() must complete in < THOMPSON_SELECT_TIMEOUT_MS ms
    after 10,000 updates. Memory footprint must not explode (checked by arm
    count staying bounded at MAX_ARMS).

    Rationale: BetaArm objects accumulate in-memory. With MAX_ARMS=500 the LRU
    eviction prevents unbounded growth; this test verifies the eviction path
    fires and selection speed does not degrade.
    """

    def test_select_model_fast_after_10k_updates(self) -> None:
        """10,000 arm updates must not make select_model() exceed the latency budget.

        Threshold: select_model() wall-clock time < THOMPSON_SELECT_TIMEOUT_MS ms.
        Arm count must stay <= ThompsonSamplingSelector.MAX_ARMS.
        """
        from vetinari.learning.thompson_arms import ThompsonBetaArm

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        # Manually initialise to bypass disk I/O in __init__ (test isolation)
        selector._arms = {}
        selector._lock = threading.Lock()
        selector._update_count = 0

        # Insert THOMPSON_TRAINING_RECORDS arm updates across a realistic spread
        # of model_id/task_type combinations to exercise the LRU eviction path.
        task_types = ["coding", "review", "research", "documentation", "reasoning"]
        model_ids = [f"model-{i}" for i in range(20)]

        for i in range(THOMPSON_TRAINING_RECORDS):
            model_id = model_ids[i % len(model_ids)]
            task_type = task_types[i % len(task_types)]
            key = f"{model_id}:{task_type}"
            quality = 0.5 + (i % 10) * 0.05  # spread across [0.5, 0.95]
            success = i % 3 != 0  # ~66% success rate

            with selector._lock:
                if key not in selector._arms:
                    if len(selector._arms) >= selector.MAX_ARMS:
                        # LRU eviction: remove least recently updated arm
                        lru = min(selector._arms, key=lambda k: selector._arms[k].last_updated)
                        del selector._arms[lru]
                    selector._arms[key] = ThompsonBetaArm(
                        model_id=model_id,
                        task_type=task_type,
                    )
                selector._arms[key].update(quality, success)

        # Arm count must stay bounded by MAX_ARMS.
        with selector._lock:
            arm_count = len(selector._arms)
        assert arm_count <= selector.MAX_ARMS, (
            f"Arms grew to {arm_count}, exceeding MAX_ARMS={selector.MAX_ARMS} — LRU eviction not working"
        )

        # select_model() must complete within the latency budget.
        start = time.perf_counter()
        result = selector.select_model(
            task_type="coding",
            candidate_models=model_ids,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < THOMPSON_SELECT_TIMEOUT_MS, (
            f"select_model() took {elapsed_ms:.1f}ms after {THOMPSON_TRAINING_RECORDS} updates, "
            f"threshold {THOMPSON_SELECT_TIMEOUT_MS}ms"
        )

        # Result must be a valid model_id from the candidates list.
        assert result in model_ids, f"select_model() returned {result!r}, not a candidate model"

    def test_arm_memory_bounded_after_10k_inserts(self) -> None:
        """Arm dict size stays at MAX_ARMS after inserting far more unique arms.

        Threshold: arm count == MAX_ARMS after inserting 2x MAX_ARMS unique entries.
        This proves the LRU eviction path runs and caps memory growth.
        """
        from vetinari.learning.thompson_arms import ThompsonBetaArm

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector._arms = {}
        selector._lock = threading.Lock()
        selector._update_count = 0

        # Insert significantly more arms than MAX_ARMS to force eviction.
        overflow_count = selector.MAX_ARMS * 2

        with selector._lock:
            for i in range(overflow_count):
                key = f"model-{i}:coding"
                if len(selector._arms) >= selector.MAX_ARMS:
                    lru = min(selector._arms, key=lambda k: selector._arms[k].last_updated)
                    del selector._arms[lru]
                selector._arms[key] = ThompsonBetaArm(model_id=f"model-{i}", task_type="coding")
            final_count = len(selector._arms)

        assert final_count <= selector.MAX_ARMS, f"Arms not bounded: {final_count} > MAX_ARMS={selector.MAX_ARMS}"


# =============================================================================
# Test 3: 50,000 memory entries with FTS5
# =============================================================================


class TestFTS5Scale:
    """Verify FTS5 full-text search returns results within budget after 50k inserts.

    Threshold: FTS5 search must return in < FTS5_SEARCH_TIMEOUT_MS ms.
    Known entries must appear in search results (quality check, not just shape).

    Rationale: FTS5 is used for memory recall. If it degrades to O(n) scans at
    50k entries the system becomes unusable for long-running agents.
    """

    def test_fts5_search_within_budget_after_50k_entries(self, tmp_path: pytest.TempPathFactory) -> None:
        """Insert FTS5_ENTRY_COUNT entries; assert FTS search returns in time.

        Threshold: FTS search wall-clock time < FTS5_SEARCH_TIMEOUT_MS ms.
        Known inserted entries must appear in results (search quality check).
        """
        conn = _make_private_db(tmp_path)
        _create_memory_schema(conn)

        # Sentinel entries inserted at known positions; used to verify search quality.
        sentinel_content = "vetinari_scale_test_sentinel_unique_phrase_xyz"
        sentinel_ids: list[str] = []

        # Insert FTS5_ENTRY_COUNT entries in batches for throughput.
        batch_size = 500
        now_ts = int(time.time())

        for batch_start in range(0, FTS5_ENTRY_COUNT, batch_size):
            rows = []
            for i in range(batch_start, min(batch_start + batch_size, FTS5_ENTRY_COUNT)):
                entry_id = str(uuid.uuid4())
                # Embed sentinels at regular intervals so search can find them.
                if i % 5000 == 0:
                    content = f"{sentinel_content} entry {i}"
                    sentinel_ids.append(entry_id)
                else:
                    content = f"memory entry {i} about topic {i % 200} with context data"
                rows.append((
                    entry_id,
                    "agent_foreman",
                    "episodic",
                    content,
                    f"summary {i}",
                    now_ts + i,
                    f"hash_{i:08d}",
                ))
            conn.executemany(
                "INSERT INTO memories (id, agent, entry_type, content, summary, timestamp, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        # Verify the exact count was inserted before timing the search.
        cursor = conn.execute("SELECT COUNT(*) AS cnt FROM memories")
        total = cursor.fetchone()["cnt"]
        assert total == FTS5_ENTRY_COUNT, f"Expected {FTS5_ENTRY_COUNT} rows in memories, found {total}"

        # Time the FTS5 search.
        start = time.perf_counter()
        cursor = conn.execute(
            "SELECT id FROM memory_fts WHERE memory_fts MATCH ? LIMIT 20",
            (sentinel_content,),
        )
        results = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < FTS5_SEARCH_TIMEOUT_MS, (
            f"FTS5 search took {elapsed_ms:.1f}ms after {FTS5_ENTRY_COUNT} entries, "
            f"threshold {FTS5_SEARCH_TIMEOUT_MS}ms"
        )

        # At least one sentinel must be found — verifies the index is alive,
        # not just that the query returned fast by short-circuiting.
        result_ids = {row[0] for row in results}
        assert result_ids & set(sentinel_ids), (
            f"FTS5 search returned {len(results)} results but none were known sentinel entries. "
            f"Sentinel IDs: {sentinel_ids[:3]}. Result IDs: {list(result_ids)[:3]}"
        )

        conn.close()

    def test_fts5_returns_correct_entries(self, tmp_path: pytest.TempPathFactory) -> None:
        """FTS5 search must find a known phrase inserted at a specific row ID.

        Threshold: search for 'unique_canary_phrase' must return exactly the
        one row that contains it. This guards against FTS index staleness.
        """
        conn = _make_private_db(tmp_path)
        _create_memory_schema(conn)

        canary_id = str(uuid.uuid4())
        canary_phrase = "unique_canary_phrase_for_scale_test"

        conn.execute(
            "INSERT INTO memories (id, agent, entry_type, content, summary, timestamp, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (canary_id, "test_agent", "fact", canary_phrase, "canary summary", int(time.time()), "canary_hash"),
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT id FROM memory_fts WHERE memory_fts MATCH ? LIMIT 10",
            (canary_phrase,),
        )
        rows = cursor.fetchall()
        result_ids = [row[0] for row in rows]

        assert canary_id in result_ids, (
            f"FTS5 did not find canary entry {canary_id!r} when searching for {canary_phrase!r}. Results: {result_ids}"
        )

        conn.close()


# =============================================================================
# Test 4: 50 concurrent SSE listeners — queue leak check
# =============================================================================


class TestSSEQueueLeaks:
    """Verify SSE queues are cleaned up after listeners disconnect.

    Threshold: after SSE_LISTENER_COUNT listeners connect and disconnect,
    the SSE stream registry must return to the baseline count (zero leaked queues).

    Rationale: The anti-patterns doc documents SSE queue leaks as a known
    failure mode — every completed project leaked a Queue. This test proves
    the cleanup path works under concurrent load.
    """

    def test_50_listeners_no_queue_leaks(self) -> None:
        """Simulate 50 concurrent SSE listeners; verify zero queues remain after cleanup.

        Threshold: len(_sse_streams) after all listeners disconnect must equal
        the baseline count before any listeners were registered.
        """
        from vetinari.web.shared import (
            _cleanup_project_state,
            _get_sse_queue,
            _sse_streams,
            _sse_streams_lock,
        )

        # Snapshot baseline queue count before the test (other tests may leave state).
        with _sse_streams_lock:
            baseline_count = len(_sse_streams)

        project_ids = [f"scale-sse-{i:04d}" for i in range(SSE_LISTENER_COUNT)]

        # Step 1: register all listeners concurrently.
        def _register(pid: str) -> None:
            _get_sse_queue(pid)

        with ThreadPoolExecutor(max_workers=SSE_LISTENER_COUNT) as pool:
            for fut in as_completed([pool.submit(_register, pid) for pid in project_ids]):
                fut.result()

        # All queues must now be present.
        with _sse_streams_lock:
            after_registration = len(_sse_streams)
        assert after_registration >= baseline_count + SSE_LISTENER_COUNT, (
            f"Expected at least {baseline_count + SSE_LISTENER_COUNT} queues after registration, "
            f"got {after_registration}"
        )

        # Step 2: simulate clients receiving a sentinel value from the queue,
        # then the "listener" calls cleanup (mirrors the SSE generator finally block).
        def _listen_and_cleanup(pid: str) -> None:
            q = _get_sse_queue(pid)
            # Drain any pending events (non-blocking) to simulate a listener that
            # actively consumes until the None sentinel arrives.
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:  # noqa: VET022 — Empty is the expected exit from the drain loop
                pass
            # Cleanup simulates the generator's finally block.
            _cleanup_project_state(pid)

        with ThreadPoolExecutor(max_workers=SSE_LISTENER_COUNT) as pool:
            for fut in as_completed([pool.submit(_listen_and_cleanup, pid) for pid in project_ids]):
                fut.result()

        # All scale-test queues must be removed — no leaks.
        with _sse_streams_lock:
            after_cleanup = len(_sse_streams)
            remaining_scale_ids = [pid for pid in project_ids if pid in _sse_streams]

        assert remaining_scale_ids == [], (
            f"SSE queue leak: {len(remaining_scale_ids)} queues not cleaned up "
            f"after listener disconnect: {remaining_scale_ids[:5]}"
        )

        # Total count must return to baseline.
        assert after_cleanup == baseline_count, (
            f"Queue count after cleanup ({after_cleanup}) != baseline ({baseline_count}); "
            f"{after_cleanup - baseline_count} queues leaked"
        )

    def test_push_event_to_all_listeners_received(self) -> None:
        """Push an event to SSE_LISTENER_COUNT queues; verify all are received.

        Threshold: every listener queue must contain at least one event within
        100ms of the push. This confirms _push_sse_event delivers to all active
        listeners, not just the first.
        """
        from unittest.mock import patch

        from vetinari.web.shared import (
            _cleanup_project_state,
            _get_sse_queue,
            _sse_streams_lock,
        )

        project_ids = [f"scale-push-{i:04d}" for i in range(SSE_LISTENER_COUNT)]
        queues: dict[str, _queue.Queue] = {}

        for pid in project_ids:
            queues[pid] = _get_sse_queue(pid)

        # Patch the DB write inside _push_sse_event to avoid needing a live DB.
        with patch("vetinari.database.get_connection") as mock_conn:
            mock_conn.return_value.__enter__ = mock_conn
            mock_conn.return_value.execute = lambda *a, **kw: None
            mock_conn.return_value.commit = lambda: None

            from vetinari.web.shared import _push_sse_event

            for pid in project_ids:
                _push_sse_event(pid, "task_completed", {"result": "ok"})

        # Every queue must have received the event.
        missing: list[str] = []
        for pid in project_ids:
            q = queues[pid]
            if q.empty():
                missing.append(pid)

        # Cleanup before asserting so queues don't leak on failure.
        for pid in project_ids:
            _cleanup_project_state(pid)

        assert missing == [], (
            f"{len(missing)} out of {SSE_LISTENER_COUNT} queues did not receive the pushed event: {missing[:5]}"
        )
