"""Tests for SQLite WAL checkpointing in durable_execution.py (Dept 4.2)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.durable_execution import (
    DurableExecutionEngine,
    _DatabaseManager,
)


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Provide a temporary checkpoint directory."""
    return tmp_path / "checkpoints"


@pytest.fixture
def engine(tmp_checkpoint_dir):
    """Create a DurableExecutionEngine with temp directory."""
    return DurableExecutionEngine(checkpoint_dir=str(tmp_checkpoint_dir))


class TestDatabaseManager:
    """Tests for the _DatabaseManager thread-safe SQLite wrapper."""

    def test_init_creates_db(self, tmp_path):
        """Database file is created on init."""
        db_path = tmp_path / "test.db"
        _DatabaseManager(db_path)
        assert db_path.exists()

    def test_wal_mode_enabled(self, tmp_path):
        """WAL journal mode is set."""
        db_path = tmp_path / "test.db"
        _DatabaseManager(db_path)
        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_schema_created(self, tmp_path):
        """All three tables exist after init."""
        db_path = tmp_path / "test.db"
        _DatabaseManager(db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "execution_state" in tables
        assert "task_checkpoints" in tables
        assert "paused_questions" in tables

    def test_execute_returns_rows(self, tmp_path):
        """execute() returns query results."""
        db_path = tmp_path / "test.db"
        mgr = _DatabaseManager(db_path)
        mgr.execute(
            "INSERT INTO execution_state (execution_id, goal, pipeline_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-1", "goal", "executing", "2026-01-01", "2026-01-01"),
        )
        rows = mgr.execute("SELECT execution_id FROM execution_state")
        assert len(rows) == 1
        assert rows[0][0] == "test-1"

    def test_thread_safety(self, tmp_path):
        """Concurrent writes from multiple threads don't corrupt data."""
        db_path = tmp_path / "test.db"
        mgr = _DatabaseManager(db_path)
        errors = []

        def writer(idx):
            try:
                mgr.execute(
                    "INSERT INTO execution_state (execution_id, goal, pipeline_state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (f"plan-{idx}", "goal", "executing", "2026-01-01", "2026-01-01"),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        rows = mgr.execute("SELECT COUNT(*) FROM execution_state")
        assert rows[0][0] == 10


class TestSaveLoadCheckpoint:
    """Tests for save/load checkpoint via SQLite."""

    def _make_mock_graph(self, plan_id="test-plan"):
        """Create a mock ExecutionGraph."""
        graph = MagicMock()
        graph.plan_id = plan_id
        graph.status = MagicMock()
        graph.status.value = "executing"
        graph.to_dict.return_value = {
            "plan_id": plan_id,
            "goal": "test goal",
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
            "status": "executing",
            "current_layer": 0,
            "completed_count": 0,
            "failed_count": 0,
            "nodes": {},
        }
        graph.get_completed_tasks.return_value = []
        node = MagicMock()
        node.id = "task-1"
        node.status = MagicMock()
        node.status.value = "pending"

        node.input_data = {}
        node.output_data = None
        node.task_type = "build"
        node.started_at = None
        node.completed_at = None
        node.retry_count = 0
        graph.nodes = {"task-1": node}
        return graph

    def test_save_creates_execution_state(self, engine):
        """Saving a checkpoint creates an execution_state row."""
        graph = self._make_mock_graph()
        engine._save_checkpoint("test-plan", graph)

        rows = engine._db.execute("SELECT execution_id, goal FROM execution_state")
        assert len(rows) == 1
        assert rows[0][0] == "test-plan"

    def test_save_creates_task_checkpoints(self, engine):
        """Saving a checkpoint creates task_checkpoint rows."""
        graph = self._make_mock_graph()
        engine._save_checkpoint("test-plan", graph)

        rows = engine._db.execute("SELECT task_id, execution_id FROM task_checkpoints")
        assert len(rows) == 1
        assert rows[0][0] == "task-1"

    def test_save_upsert_updates(self, engine):
        """Saving twice for same plan_id updates (not duplicates)."""
        graph = self._make_mock_graph()
        engine._save_checkpoint("test-plan", graph)
        engine._save_checkpoint("test-plan", graph)

        rows = engine._db.execute("SELECT COUNT(*) FROM execution_state WHERE execution_id = ?", ("test-plan",))
        assert rows[0][0] == 1

    def test_load_returns_graph(self, engine):
        """Loading a saved checkpoint returns an ExecutionGraph."""
        graph = self._make_mock_graph()
        engine._save_checkpoint("test-plan", graph)

        loaded = engine.load_checkpoint("test-plan")
        assert loaded is not None
        assert loaded.plan_id == "test-plan"

    def test_load_missing_returns_none(self, engine):
        """Loading a non-existent plan returns None."""
        loaded = engine.load_checkpoint("nonexistent")
        assert loaded is None

    def test_list_checkpoints(self, engine):
        """list_checkpoints returns saved plan IDs."""
        graph = self._make_mock_graph("plan-a")
        engine._save_checkpoint("plan-a", graph)
        graph2 = self._make_mock_graph("plan-b")
        engine._save_checkpoint("plan-b", graph2)

        checkpoints = engine.list_checkpoints()
        assert "plan-a" in checkpoints
        assert "plan-b" in checkpoints


class TestPauseResume:
    """Tests for clarification pause/resume."""

    def test_save_paused_questions(self, engine):
        """Saving paused questions stores them in SQLite."""
        qid = engine.save_paused_questions("exec-1", ["What file?", "What format?"])
        assert qid  # non-empty string

        rows = engine._db.execute("SELECT questions_json FROM paused_questions WHERE question_id = ?", (qid,))
        assert len(rows) == 1
        questions = json.loads(rows[0][0])
        assert len(questions) == 2

    def test_answer_paused_questions(self, engine):
        """Answering questions updates the row."""
        qid = engine.save_paused_questions("exec-1", ["What file?"])
        engine.answer_paused_questions(qid, ["main.py"])

        rows = engine._db.execute(
            "SELECT answers_json, answered_at FROM paused_questions WHERE question_id = ?", (qid,)
        )
        assert isinstance(rows[0][0], str)
        answers = json.loads(rows[0][0])
        assert answers == ["main.py"]
        assert isinstance(rows[0][1], str)  # answered_at set as ISO string
        assert "T" in rows[0][1]

    def test_get_paused_questions(self, engine):
        """get_paused_questions returns all questions for an execution."""
        engine.save_paused_questions("exec-1", ["Q1"])
        engine.save_paused_questions("exec-1", ["Q2"])
        engine.save_paused_questions("exec-2", ["Q3"])

        result = engine.get_paused_questions("exec-1")
        assert len(result) == 2

    def test_get_paused_unanswered(self, engine):
        """Unanswered questions have answers=None."""
        engine.save_paused_questions("exec-1", ["Q1"])
        result = engine.get_paused_questions("exec-1")
        assert result[0]["answers"] is None


class TestCleanup:
    """Tests for cleanup_completed."""

    def test_cleanup_removes_old(self, engine):
        """Completed executions older than cutoff are removed."""
        engine._db.execute(
            "INSERT INTO execution_state (execution_id, goal, pipeline_state, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("old-1", "goal", "complete", "2020-01-01", "2020-01-01", "2020-01-01"),
        )
        removed = engine.cleanup_completed(max_age_days=1)
        assert removed == 1
        rows = engine._db.execute("SELECT COUNT(*) FROM execution_state")
        assert rows[0][0] == 0

    def test_cleanup_keeps_recent(self, engine):
        """Recent completed executions are not removed."""
        from datetime import datetime

        now = datetime.now().isoformat()
        engine._db.execute(
            "INSERT INTO execution_state (execution_id, goal, pipeline_state, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("recent-1", "goal", "complete", now, now, now),
        )
        removed = engine.cleanup_completed(max_age_days=1)
        assert removed == 0
