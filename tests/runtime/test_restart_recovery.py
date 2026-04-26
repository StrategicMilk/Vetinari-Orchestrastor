"""Cross-process persistence tests for Vetinari's durable state systems.

Each test spawns writer and reader subprocesses that communicate via a shared
temporary SQLite database, verifying that state written in one process survives
and is readable in a completely separate process — the same scenario as a
server restart.

These tests are intentionally slow (subprocess overhead) but irreplaceable:
in-process tests with mocks cannot prove that SQLite durability actually works
across a process boundary.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 30  # seconds per subprocess


def _run(code: str, db_path: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter with VETINARI_DB_PATH set.

    Args:
        code: Python source to execute.
        db_path: Absolute path to the temp SQLite database.

    Returns:
        Completed process with stdout/stderr captured.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        env={
            # Inherit the parent environment so vetinari imports work, then
            # override the DB path so the subprocess uses our temp database.
            **os.environ,
            "VETINARI_DB_PATH": db_path,
        },
    )


def _last_json(stdout: str) -> dict:
    """Parse the last non-empty line of *stdout* as JSON.

    Args:
        stdout: Full stdout string from a subprocess.

    Returns:
        Parsed JSON dict from the last output line.

    Raises:
        AssertionError: If stdout is empty or the last line is not valid JSON.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, f"Subprocess produced no output. stdout={stdout!r}"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Test 1 — conversation messages
# ---------------------------------------------------------------------------


def test_conversation_persistence_across_restart(tmp_path: Path) -> None:
    """Messages written to SQLite in one process are readable by the next.

    Writer inserts a conversation message directly via the unified database
    module. Reader creates a fresh connection and queries back by session_id.
    This proves the conversation store's SQLite persistence survives a restart.
    """
    db_path = str(tmp_path / "conv.db")

    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        conn.execute(
            "INSERT INTO conversation_messages (session_id, role, content, timestamp, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("sess-001", "user", "hello from writer", 1000.0, "{{}}"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT content FROM conversation_messages WHERE session_id = ?",
            ("sess-001",),
        ).fetchall()
        print(json.dumps({{"count": len(rows), "content": rows[0][0]}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        rows = conn.execute(
            "SELECT role, content FROM conversation_messages "
            "WHERE session_id = ? ORDER BY timestamp",
            ("sess-001",),
        ).fetchall()
        print(json.dumps({{"count": len(rows), "messages": [dict(role=r[0], content=r[1]) for r in rows]}}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    writer_data = _last_json(result_w.stdout)
    assert writer_data["count"] == 1, "Writer did not persist the message"

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)
    assert data["count"] == 1, f"Expected 1 message across restart, got {data['count']}"
    assert data["messages"][0]["content"] == "hello from writer"
    assert data["messages"][0]["role"] == "user"

    logger.info("test_conversation_persistence_across_restart: PASS")


# ---------------------------------------------------------------------------
# Test 2 — telemetry snapshot restoration
# ---------------------------------------------------------------------------


def test_telemetry_snapshot_restoration_across_restart(tmp_path: Path) -> None:
    """Telemetry counters written to telemetry_snapshots are restored by TelemetryCollector.

    Writer inserts a snapshot row directly into SQLite with known adapter
    request counts. Reader creates a fresh TelemetryCollector and calls
    restore_from_snapshot(), then checks get_summary() for the restored totals.
    """
    db_path = str(tmp_path / "telemetry.db")

    snapshot_payload = json.dumps({
        "adapter_details": {
            "local:llama3": {
                "total_requests": 42,
                "failed_requests": 2,
                "avg_latency_ms": 150.0,
                "min_latency_ms": 90.0,
                "max_latency_ms": 300.0,
            }
        },
        "by_model": {"llama3": {"tokens": 8000}},
    })

    writer_code = f"""
        import json, os, time
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry_snapshots "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, data TEXT NOT NULL)"
        )
        conn.commit()
        conn.execute(
            "INSERT INTO telemetry_snapshots (timestamp, data) VALUES (?, ?)",
            (time.time(), {snapshot_payload!r}),
        )
        conn.commit()
        rows = conn.execute("SELECT COUNT(*) FROM telemetry_snapshots").fetchone()
        print(json.dumps({{"rows": rows[0]}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        _d.get_connection()
        from vetinari.telemetry import TelemetryCollector
        tc = TelemetryCollector()
        tc.restore_from_snapshot()
        summary = tc.get_summary()
        # get_summary() returns session_requests (total), by_provider, by_model
        by_provider = summary.get("by_provider", {{}})
        local_stats = by_provider.get("local", {{}})
        by_model = summary.get("by_model", {{}})
        llama_stats = by_model.get("llama3", {{}})
        print(json.dumps({{
            "session_requests": summary.get("session_requests", 0),
            "provider_requests": local_stats.get("requests", 0),
            "model_requests": llama_stats.get("requests", 0),
        }}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    assert _last_json(result_w.stdout)["rows"] == 1

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)
    assert data["session_requests"] == 42, f"Expected 42 session requests, got {data['session_requests']}"
    assert data["provider_requests"] == 42, f"Expected 42 provider requests, got {data['provider_requests']}"
    assert data["model_requests"] == 42, f"Expected 42 model requests, got {data['model_requests']}"

    logger.info("test_telemetry_snapshot_restoration_across_restart: PASS")


# ---------------------------------------------------------------------------
# Test 3 — checkpoint graph persistence
# ---------------------------------------------------------------------------


def test_checkpoint_persistence_across_restart(tmp_path: Path) -> None:
    """A graph saved by CheckpointStore in one process is loadable in the next.

    Writer builds a minimal graph_dict and calls save_checkpoint(). Reader
    creates a new CheckpointStore pointing at the same database file and calls
    load_checkpoint_graph_json() — proving the checkpoint survives restart.
    """
    db_path = str(tmp_path / "checkpoint.db")
    plan_id = "plan-restart-001"

    graph_dict = json.dumps({"goal": "test goal", "tasks": [{"id": "t1"}], "created_at": "2026-01-01T00:00:00Z"})

    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        graph_dict = json.loads({graph_dict!r})
        store.save_checkpoint(
            plan_id={plan_id!r},
            graph_dict=graph_dict,
            pipeline_state="executing",
            task_rows=[],
            now="2026-01-01T00:00:00Z",
        )
        loaded = store.load_checkpoint_graph_json({plan_id!r})
        print(json.dumps({{"saved": loaded is not None}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        raw = store.load_checkpoint_graph_json({plan_id!r})
        if raw is None:
            print(json.dumps({{"found": False, "goal": None}}))
        else:
            parsed = json.loads(raw)
            print(json.dumps({{"found": True, "goal": parsed.get("goal")}}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    assert _last_json(result_w.stdout)["saved"] is True

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)
    assert data["found"] is True, "Checkpoint not found after restart"
    assert data["goal"] == "test goal", f"Goal mismatch: {data['goal']!r}"

    logger.info("test_checkpoint_persistence_across_restart: PASS")


# ---------------------------------------------------------------------------
# Test 4 — incomplete execution recovery
# ---------------------------------------------------------------------------


def test_incomplete_execution_recovery_across_restart(tmp_path: Path) -> None:
    """Executions in 'executing' state are detected as incomplete after restart.

    Writer saves a checkpoint with pipeline_state="executing". Reader calls
    find_incomplete_ids() — the plan_id must appear in the returned list,
    proving the recovery detector works across a process boundary.
    """
    db_path = str(tmp_path / "incomplete.db")
    plan_id = "plan-incomplete-001"

    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        store.save_checkpoint(
            plan_id={plan_id!r},
            graph_dict={{"goal": "incomplete goal", "tasks": [], "created_at": "2026-01-01T00:00:00Z"}},
            pipeline_state="executing",
            task_rows=[],
            now="2026-01-01T00:00:00Z",
        )
        ids = store.find_incomplete_ids("completed", "failed")
        print(json.dumps({{"incomplete_ids": ids}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        ids = store.find_incomplete_ids("completed", "failed")
        print(json.dumps({{"incomplete_ids": ids, "found": {plan_id!r} in ids}}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    writer_data = _last_json(result_w.stdout)
    assert plan_id in writer_data["incomplete_ids"], "Writer's own find_incomplete_ids missed the plan"

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)
    assert data["found"] is True, (
        f"Plan {plan_id!r} not detected as incomplete after restart. Got: {data['incomplete_ids']}"
    )

    logger.info("test_incomplete_execution_recovery_across_restart: PASS")


# ---------------------------------------------------------------------------
# Test 5 — SSE event log replay
# ---------------------------------------------------------------------------


def test_sse_event_replay_across_restart(tmp_path: Path) -> None:
    """SSE events written to sse_event_log are queryable by a new process.

    Writer inserts two SSE events for a project. Reader queries them back,
    proving the SSE audit trail survives across a process boundary and
    supports replay after restart.
    """
    db_path = str(tmp_path / "sse.db")
    project_id = "proj-sse-001"

    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        for i, event_type in enumerate(["task.started", "task.completed"]):
            conn.execute(
                "INSERT INTO sse_event_log (project_id, event_type, payload_json, sequence_num) "
                "VALUES (?, ?, ?, ?)",
                ({project_id!r}, event_type, json.dumps({{"seq": i}}), i),
            )
        conn.commit()
        rows = conn.execute(
            "SELECT COUNT(*) FROM sse_event_log WHERE project_id = ?",
            ({project_id!r},),
        ).fetchone()
        print(json.dumps({{"count": rows[0]}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        rows = conn.execute(
            "SELECT event_type, payload_json, sequence_num FROM sse_event_log "
            "WHERE project_id = ? ORDER BY sequence_num",
            ({project_id!r},),
        ).fetchall()
        events = [
            {{"event_type": r[0], "payload": json.loads(r[1]), "seq": r[2]}}
            for r in rows
        ]
        print(json.dumps({{"count": len(events), "events": events}}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    assert _last_json(result_w.stdout)["count"] == 2

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)
    assert data["count"] == 2, f"Expected 2 SSE events after restart, got {data['count']}"
    assert data["events"][0]["event_type"] == "task.started"
    assert data["events"][1]["event_type"] == "task.completed"
    assert data["events"][0]["seq"] == 0
    assert data["events"][1]["seq"] == 1

    logger.info("test_sse_event_replay_across_restart: PASS")


# ---------------------------------------------------------------------------
# Test 6 — no synthetic completed state
# ---------------------------------------------------------------------------


def test_no_synthetic_completed_state(tmp_path: Path) -> None:
    """Recovery must never fabricate 'completed' state that was not written.

    Writer saves a checkpoint with pipeline_state="executing" and a task with
    status="pending". Reader loads it back and verifies both values are exactly
    as written — recovery logic must not invent a completed outcome.
    """
    db_path = str(tmp_path / "nosynth.db")
    plan_id = "plan-nosynth-001"
    task_id = "task-nosynth-001"

    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        task_rows = [
            (
                {task_id!r},   # task_id
                {plan_id!r},   # execution_id
                "worker",      # agent_type
                "standard",    # mode
                "pending",     # status — intentionally not completed
                json.dumps({{"input": "data"}}),  # input_json
                None,          # output_json
                None,          # manifest_hash
                None,          # started_at
                None,          # completed_at
                0,             # retry_count
            )
        ]
        store.save_checkpoint(
            plan_id={plan_id!r},
            graph_dict={{"goal": "no synth goal", "tasks": [], "created_at": "2026-01-01T00:00:00Z"}},
            pipeline_state="executing",
            task_rows=task_rows,
            now="2026-01-01T00:00:00Z",
        )
        print(json.dumps({{"written": True}}))
    """

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        # Check pipeline state
        exec_row = conn.execute(
            "SELECT pipeline_state FROM execution_state WHERE execution_id = ?",
            ({plan_id!r},),
        ).fetchone()
        # Check task status
        task_row = conn.execute(
            "SELECT status FROM task_checkpoints WHERE task_id = ?",
            ({task_id!r},),
        ).fetchone()
        print(json.dumps({{
            "pipeline_state": exec_row[0] if exec_row else None,
            "task_status": task_row[0] if task_row else None,
        }}))
    """

    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"

    result_r = _run(reader_code, db_path)
    assert result_r.returncode == 0, f"Reader failed:\n{result_r.stderr}"
    data = _last_json(result_r.stdout)

    assert data["pipeline_state"] == "executing", (
        f"Pipeline state was altered to {data['pipeline_state']!r} — recovery must not synthesize 'completed'"
    )
    assert data["task_status"] == "pending", (
        f"Task status was altered to {data['task_status']!r} — recovery must not synthesize task completion"
    )

    logger.info("test_no_synthetic_completed_state: PASS")


# ---------------------------------------------------------------------------
# Test 7 — empty database graceful start
# ---------------------------------------------------------------------------


def test_empty_database_graceful_start(tmp_path: Path) -> None:
    """All persistence systems return empty/zero results on a fresh database.

    A process starting against an empty database must not raise exceptions
    when querying for conversations, telemetry snapshots, or checkpoints.
    This validates the graceful-start guarantee needed at first boot.
    """
    db_path = str(tmp_path / "empty.db")

    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()

        # 1) Conversations — should return empty list
        conv_rows = conn.execute(
            "SELECT COUNT(*) FROM conversation_messages"
        ).fetchone()

        # 2) Telemetry restore — TelemetryCollector must handle missing table gracefully.
        #    The table is created by TelemetryPersistence.start(), which is not called
        #    here; restore_from_snapshot() must gracefully handle the missing table.
        from vetinari.telemetry import TelemetryCollector
        tc = TelemetryCollector()
        tc.restore_from_snapshot()  # must not raise
        summary = tc.get_summary()

        # 3) Checkpoints — should return empty lists
        from vetinari.orchestration.checkpoint_store import CheckpointStore
        store = CheckpointStore()
        checkpoint_ids = store.list_checkpoint_ids()
        incomplete_ids = store.find_incomplete_ids("completed", "failed")

        print(json.dumps({{
            "conv_count": conv_rows[0],
            "telemetry_adapters": len(summary.get("adapter_details", {{}})),
            "checkpoint_ids": checkpoint_ids,
            "incomplete_ids": incomplete_ids,
        }}))
    """

    result = _run(reader_code, db_path)
    assert result.returncode == 0, f"Empty-DB reader crashed:\n{result.stderr}"
    data = _last_json(result.stdout)

    assert data["conv_count"] == 0, f"Expected 0 conversations, got {data['conv_count']}"
    assert data["telemetry_adapters"] == 0, (
        f"Expected 0 telemetry adapters on fresh DB, got {data['telemetry_adapters']}"
    )
    assert data["checkpoint_ids"] == [], f"Expected empty checkpoint list, got {data['checkpoint_ids']}"
    assert data["incomplete_ids"] == [], f"Expected empty incomplete list, got {data['incomplete_ids']}"

    logger.info("test_empty_database_graceful_start: PASS")


# ---------------------------------------------------------------------------
# Test 6 — A2A acknowledged task survival across restart (US-30A.4)
# ---------------------------------------------------------------------------


def test_a2a_acknowledged_task_survives_restart(tmp_path: Path) -> None:
    """An ACKNOWLEDGED A2A task is re-processed (or orphaned) on restart — never silently dropped.

    Process 1 (writer): inserts an a2a_tasks row with status='acknowledged'
    directly into SQLite, simulating a previous run where the orchestrator was
    unavailable when the task arrived.

    Process 2 (restarter): creates a fresh ``VetinariA2AExecutor`` with
    ``recover_on_init=True``.  The executor sees the acknowledged task, attempts
    re-execution, and — because the orchestrator is unavailable in the test
    environment — transitions the task to ``STATUS_ORPHANED`` (a terminal state).

    Process 3 (reader): queries the task row and asserts that the status is no
    longer ``'acknowledged'``.  The exact terminal state (``'orphaned'``,
    ``'completed'``, or ``'failed'``) depends on orchestrator availability, but
    the task MUST NOT remain stuck in the acknowledged limbo state.
    """
    db_path = str(tmp_path / "a2a_recovery.db")

    # -- Process 1: write an acknowledged task directly into SQLite ------------
    writer_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS a2a_tasks ("
            "    task_id TEXT PRIMARY KEY,"
            "    task_type TEXT NOT NULL,"
            "    status TEXT NOT NULL DEFAULT 'pending',"
            "    input_json TEXT,"
            "    output_json TEXT,"
            "    error TEXT DEFAULT '',"
            "    created_at TEXT NOT NULL,"
            "    updated_at TEXT NOT NULL"
            ")"
        )
        conn.commit()
        conn.execute(
            "INSERT INTO a2a_tasks "
            "(task_id, task_type, status, input_json, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "ack-task-restart-test",
                "build",
                "acknowledged",
                json.dumps({{"goal": "survive restart"}}),
                "",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT task_id, status FROM a2a_tasks WHERE task_id = ?",
            ("ack-task-restart-test",),
        ).fetchone()
        print(json.dumps({{"task_id": row[0], "initial_status": row[1]}}))
    """

    # -- Process 2: restart the executor so recovery fires --------------------
    restarter_code = f"""
        import os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        from vetinari.a2a.executor import VetinariA2AExecutor
        # recover_on_init=True triggers _run_startup_recovery(), which will:
        #   1. Load the acknowledged task from DB
        #   2. Attempt re-execution (orchestrator unavailable -> still acknowledged)
        #   3. Call _persist_orphaned() -> status = 'orphaned'
        executor = VetinariA2AExecutor(recover_on_init=True)
        import json
        conn = _d.get_connection()
        row = conn.execute(
            "SELECT task_id, status FROM a2a_tasks WHERE task_id = ?",
            ("ack-task-restart-test",),
        ).fetchone()
        print(json.dumps({{"task_id": row[0], "post_recovery_status": row[1]}}))
    """

    # -- Process 3: reader verifies the final state ---------------------------
    reader_code = f"""
        import json, os
        os.environ["VETINARI_DB_PATH"] = {db_path!r}
        import vetinari.database as _d
        _d.reset_for_testing()
        conn = _d.get_connection()
        row = conn.execute(
            "SELECT task_id, status FROM a2a_tasks WHERE task_id = ?",
            ("ack-task-restart-test",),
        ).fetchone()
        assert row is not None, "Task row disappeared from database"
        print(json.dumps({{"task_id": row[0], "final_status": row[1]}}))
    """

    # Run process 1 — write acknowledged task
    result_w = _run(writer_code, db_path)
    assert result_w.returncode == 0, f"Writer failed:\n{result_w.stderr}"
    writer_data = _last_json(result_w.stdout)
    assert writer_data["initial_status"] == "acknowledged", (
        f"Writer did not persist acknowledged status, got: {writer_data['initial_status']!r}"
    )

    # Run process 2 — restart executor, triggering recovery
    result_r = _run(restarter_code, db_path)
    assert result_r.returncode == 0, f"Restarter crashed during executor recovery:\n{result_r.stderr}"
    restarter_data = _last_json(result_r.stdout)
    post_status = restarter_data["post_recovery_status"]
    assert post_status != "acknowledged", (
        f"Task is still 'acknowledged' after restart — recovery did not process it. "
        f"post_recovery_status={post_status!r}"
    )

    # Run process 3 — independent reader confirms durable state
    result_q = _run(reader_code, db_path)
    assert result_q.returncode == 0, f"Reader failed:\n{result_q.stderr}"
    final_data = _last_json(result_q.stdout)
    final_status = final_data["final_status"]
    assert final_status != "acknowledged", (
        f"Task returned to 'acknowledged' in a fresh reader process — "
        f"durable state was not written correctly. final_status={final_status!r}"
    )
    assert final_status in {"orphaned", "completed", "failed", "pending", "running"}, (
        f"Task has unexpected terminal status {final_status!r} — not a valid A2A task state"
    )

    logger.info(
        "test_a2a_acknowledged_task_survives_restart: PASS (final_status=%r)",
        final_status,
    )
