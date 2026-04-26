"""Tests for vetinari.async_support — async executor, streaming, and conversation memory.

Coverage:
- AsyncExecutor: single task, wave execution, plan execution, timeout, error handling
- StreamHandler hierarchy: SSEStreamHandler, LoggingStreamHandler, BufferedStreamHandler
- StreamRouter: fan-out, handler isolation
- ConversationStore: session lifecycle, message history, context window, token limiting
- ContextReconstructor: formatting, truncation
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.factories import make_conversation_message as _make_conversation_message
from vetinari.async_support.async_executor import AsyncExecutor
from vetinari.async_support.conversation import (
    ContextReconstructor,
    ConversationMessage,
    ConversationStore,
    _reset_conversation_store,
    get_conversation_store,
)
from vetinari.async_support.streaming import (
    BufferedStreamHandler,
    SSEStreamHandler,
    StreamChunk,
    StreamRouter,
)
from vetinari.exceptions import ExecutionError
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously (compatible with Python 3.10+)."""
    return asyncio.run(coro)


# ===========================================================================
# AsyncExecutor tests
# ===========================================================================


class TestAsyncExecutorSingleTask:
    """execute_task returns a completed result dict for a basic task."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.executor = AsyncExecutor()

    @pytest.mark.parametrize(
        ("task", "agent_type", "field", "expected"),
        [
            (
                {"id": "t1", "prompt": "do something", "agent_type": AgentType.WORKER.value},
                AgentType.WORKER.value,
                "status",
                "completed",
            ),
            ({"id": "task-abc", "agent_type": AgentType.FOREMAN.value}, AgentType.FOREMAN.value, "task_id", "task-abc"),
            ({"id": "t2", "agent_type": "ORACLE"}, "ORACLE", "agent_type", "ORACLE"),
        ],
    )
    def test_execute_task_result_fields(self, task, agent_type, field, expected):
        """execute_task result contains expected status, task_id, and agent_type."""
        result = _run(self.executor.execute_task(task, agent_type))
        assert result[field] == expected

    def test_single_task_missing_id_defaults_to_unknown(self):
        result = _run(self.executor.execute_task({}, AgentType.FOREMAN.value))
        assert result["task_id"] == "unknown"

    def test_single_task_exception_returns_failed_status(self):
        """Patch _dispatch to raise; execute_task must return status='failed'."""

        def _bad_dispatch(task, agent_type):
            raise RuntimeError("simulated failure")

        self.executor._dispatch = _bad_dispatch
        task = {"id": "err-task", "agent_type": AgentType.WORKER.value}
        result = _run(self.executor.execute_task(task, AgentType.WORKER.value))
        assert result["status"] == "failed"
        assert "simulated failure" in result["error"]


class TestAsyncExecutorTimeout:
    """execute_task handles timeouts correctly."""

    def test_task_timeout_returns_failed(self):
        executor = AsyncExecutor(task_timeout=0)  # 0 s forces immediate timeout

        async def _slow_dispatch(task, agent_type):
            await asyncio.sleep(10)
            return {"task_id": "x", "agent_type": agent_type, "status": "completed", "output": ""}

        executor._dispatch = _slow_dispatch
        task = {"id": "slow", "agent_type": AgentType.WORKER.value}
        result = _run(executor.execute_task(task, AgentType.WORKER.value))
        assert result["status"] == "failed"
        assert "Timeout" in result["error"]


class TestAsyncExecutorWave:
    """execute_wave runs tasks concurrently and collects all results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.executor = AsyncExecutor()

    def test_wave_returns_result_for_each_task(self):
        tasks = [
            {"id": "w1", "agent_type": AgentType.WORKER.value},
            {"id": "w2", "agent_type": "ORACLE"},
            {"id": "w3", "agent_type": AgentType.FOREMAN.value},
        ]
        results = _run(self.executor.execute_wave(tasks))
        assert len(results) == 3

    def test_wave_empty_returns_empty_list(self):
        results = _run(self.executor.execute_wave([]))
        assert results == []

    def test_wave_all_completed_by_default(self):
        tasks = [{"id": f"t{i}", "agent_type": AgentType.WORKER.value} for i in range(5)]
        results = _run(self.executor.execute_wave(tasks))
        statuses = {r["status"] for r in results}
        assert statuses == {"completed"}

    def test_wave_uses_agent_type_from_task_dict(self):
        tasks = [{"id": "a", "agent_type": AgentType.INSPECTOR.value}]
        results = _run(self.executor.execute_wave(tasks))
        assert results[0]["agent_type"] == AgentType.INSPECTOR.value


class TestAsyncExecutorPlan:
    """execute_plan processes waves sequentially."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.executor = AsyncExecutor()

    def _make_plan(self, wave_sizes: list[int]) -> dict:
        waves = []
        task_counter = 0
        for size in wave_sizes:
            tasks = [{"id": f"t{task_counter + i}", "agent_type": AgentType.WORKER.value} for i in range(size)]
            task_counter += size
            waves.append({"wave_index": len(waves), "tasks": tasks})
        return {"id": "plan-1", "waves": waves}

    def test_plan_empty_waves_returns_skipped(self):
        result = _run(self.executor.execute_plan({"id": "empty", "waves": []}))
        assert result["status"] == "skipped"
        assert result["total_tasks"] == 0

    def test_plan_single_wave(self):
        plan = self._make_plan([3])
        result = _run(self.executor.execute_plan(plan))
        assert result["status"] == "completed"
        assert result["total_tasks"] == 3

    def test_plan_multi_wave_task_count(self):
        plan = self._make_plan([2, 3, 1])
        result = _run(self.executor.execute_plan(plan))
        assert result["total_tasks"] == 6

    def test_plan_wave_results_length_matches_wave_count(self):
        plan = self._make_plan([1, 2])
        result = _run(self.executor.execute_plan(plan))
        assert len(result["wave_results"]) == 2


# ===========================================================================
# StreamChunk tests
# ===========================================================================


class TestStreamChunk:
    def test_dataclass_fields(self):
        chunk = StreamChunk(content="hello", chunk_index=0, is_final=False)
        assert chunk.content == "hello"
        assert chunk.chunk_index == 0
        assert not chunk.is_final
        assert chunk.metadata == {}

    def test_metadata_field(self):
        chunk = StreamChunk("x", 1, True, metadata={"model": "gpt-4"})
        assert chunk.metadata["model"] == "gpt-4"


# ===========================================================================
# SSEStreamHandler tests
# ===========================================================================


class TestSSEStreamHandler:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.handler = SSEStreamHandler()

    def test_non_final_chunk_formats_as_data_event(self):
        _run(self.handler.on_chunk(StreamChunk("Hi", 0, False)))
        assert self.handler.events[0].startswith("data:")

    def test_final_chunk_formats_as_done_event(self):
        _run(self.handler.on_chunk(StreamChunk("End", 1, True)))
        assert "event: done" in self.handler.events[0]

    def test_on_complete_appends_complete_event(self):
        _run(self.handler.on_complete("full text"))
        assert "event: complete" in self.handler.events[0]

    def test_on_error_appends_error_event(self):
        _run(self.handler.on_error(ValueError("boom")))
        assert "event: error" in self.handler.events[0]
        assert "boom" in self.handler.events[0]

    def test_multiple_chunks_accumulate(self):
        _run(self.handler.on_chunk(StreamChunk("A", 0, False)))
        _run(self.handler.on_chunk(StreamChunk("B", 1, True)))
        assert len(self.handler.events) == 2


# ===========================================================================
# BufferedStreamHandler tests
# ===========================================================================


class TestBufferedStreamHandler:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.handler = BufferedStreamHandler()

    def test_on_complete_sets_buffer(self):
        _run(self.handler.on_chunk(StreamChunk("Hello ", 0, False)))
        _run(self.handler.on_chunk(StreamChunk("world", 1, True)))
        _run(self.handler.on_complete("Hello world"))
        assert self.handler.buffer == "Hello world"
        assert self.handler.completed

    def test_on_error_records_error(self):
        exc = RuntimeError("fail")
        _run(self.handler.on_error(exc))
        assert self.handler.error is exc
        assert not self.handler.completed


# ===========================================================================
# StreamRouter tests
# ===========================================================================


class TestStreamRouter:
    def test_route_chunk_fans_out_to_all_handlers(self):
        router = StreamRouter()
        h1 = BufferedStreamHandler()
        h2 = SSEStreamHandler()
        router.add_handler(h1)
        router.add_handler(h2)

        chunk = StreamChunk("test", 0, False)
        _run(router.route_chunk(chunk))

        # h1 accumulated the chunk internally
        assert h1._chunks == ["test"]
        # h2 formatted it as SSE
        assert len(h2.events) == 1

    def test_route_complete_fans_out(self):
        router = StreamRouter()
        h1 = BufferedStreamHandler()
        router.add_handler(h1)
        _run(router.route_complete("full"))
        assert h1.completed
        assert h1.buffer == "full"

    def test_failing_handler_does_not_block_others(self):
        """A handler that raises on_chunk must not prevent other handlers from receiving."""
        router = StreamRouter()

        bad = MagicMock(spec=SSEStreamHandler)
        bad.on_chunk = AsyncMock(side_effect=RuntimeError("handler error"))

        good = BufferedStreamHandler()
        router.add_handler(bad)
        router.add_handler(good)

        _run(router.route_chunk(StreamChunk("safe", 0, False)))
        assert good._chunks == ["safe"]


# ===========================================================================
# ConversationStore tests
# ===========================================================================


class TestConversationStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_conversation_store()
        self.store = ConversationStore()

    def test_create_session_returns_id(self):
        sid = self.store.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_create_session_with_explicit_id(self):
        sid = self.store.create_session("my-session")
        assert sid == "my-session"

    def test_create_duplicate_session_raises(self):
        self.store.create_session("dup")
        with pytest.raises(ExecutionError):
            self.store.create_session("dup")

    def test_add_and_get_history(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "Hello")
        self.store.add_message(sid, "assistant", "Hi there")
        history = self.store.get_history(sid)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_get_history_limit(self):
        sid = self.store.create_session()
        for i in range(10):
            self.store.add_message(sid, "user", f"msg {i}")
        history = self.store.get_history(sid, limit=3)
        assert len(history) == 3
        assert history[-1].content == "msg 9"

    def test_add_message_to_unknown_session_raises(self):
        with pytest.raises(KeyError):
            self.store.add_message("nonexistent", "user", "hello")

    def test_clear_session(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "x")
        self.store.clear_session(sid)
        assert self.store.get_history(sid) == []

    def test_list_sessions(self):
        self.store.create_session("alpha")
        self.store.create_session("beta")
        sessions = self.store.list_sessions()
        assert "alpha" in sessions
        assert "beta" in sessions

    def test_get_context_window_token_limit(self):
        sid = self.store.create_session()
        # 4 chars per token * 10 tokens = 40 chars budget
        # each message ~10 chars content; max_tokens=10 should fit ~4 messages
        for i in range(20):
            self.store.add_message(sid, "user", f"msg{i:05d}")  # 8 chars each
        window = self.store.get_context_window(sid, max_tokens=10)
        # With budget=40 chars and 8 chars/msg, we fit at most 5 messages
        assert len(window) <= 5

    def test_singleton_returns_same_instance(self):
        _reset_conversation_store()
        s1 = get_conversation_store()
        s2 = get_conversation_store()
        assert s1 is s2


# ===========================================================================
# ContextReconstructor tests
# ===========================================================================


class TestContextReconstructor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rc = ContextReconstructor()

    def test_empty_messages_returns_system_header(self):
        result = self.rc.reconstruct([])
        assert "helpful AI assistant" in result

    def test_messages_are_included_in_output(self):
        msgs = [
            _make_conversation_message("user", "Hello"),
            _make_conversation_message("assistant", "Hi there"),
        ]
        result = self.rc.reconstruct(msgs)
        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there" in result

    def test_truncation_omits_old_messages(self):
        msgs = [_make_conversation_message("user", "x" * 500) for _ in range(20)]
        result = self.rc.reconstruct(msgs, max_tokens=50)
        assert "earlier message" in result.lower()

    def test_output_starts_with_system_header(self):
        result = self.rc.reconstruct([_make_conversation_message("user", "hi")])
        assert result.startswith("You are a helpful AI assistant")


# ===========================================================================
# ConversationStore SQLite persistence tests
# ===========================================================================


class TestConversationStorePersistence:
    """Verify that messages survive a ConversationStore reset (simulated restart).

    Each test calls ``_reset_conversation_store()`` to wipe both the singleton
    and the SQLite table, then creates a fresh store, writes messages, resets
    the singleton (without wiping SQLite), and constructs a new store to confirm
    the messages were loaded back from the database.
    """

    @pytest.fixture(autouse=True)
    def clean_state(self):
        """Wipe the singleton and SQLite table before and after each test."""
        _reset_conversation_store()
        yield
        _reset_conversation_store()

    def test_messages_survive_store_reset(self):
        """Messages added to one store instance are visible in a fresh instance."""
        store1 = ConversationStore()
        sid = store1.create_session("persist-session")
        store1.add_message(sid, "user", "Hello from store1")
        store1.add_message(sid, "assistant", "Reply from store1")

        # Reset singleton only — leave SQLite intact.
        import vetinari.async_support.conversation as _conv_mod

        with _conv_mod._store_lock:
            _conv_mod._store = None

        store2 = ConversationStore()
        history = store2.get_history("persist-session")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello from store1"
        assert history[1].role == "assistant"

    def test_metadata_roundtrips_through_sqlite(self):
        """Message metadata is serialised to JSON and deserialised on reload."""
        store1 = ConversationStore()
        sid = store1.create_session("meta-session")
        store1.add_message(sid, "user", "tagged message", metadata={"tag": "important", "score": 42})

        import vetinari.async_support.conversation as _conv_mod

        with _conv_mod._store_lock:
            _conv_mod._store = None

        store2 = ConversationStore()
        history = store2.get_history("meta-session")
        assert len(history) == 1
        assert history[0].metadata == {"tag": "important", "score": 42}

    def test_clear_session_removes_from_sqlite(self):
        """clear_session deletes rows from SQLite so they don't reappear on reload."""
        store1 = ConversationStore()
        sid = store1.create_session("clear-session")
        store1.add_message(sid, "user", "to be cleared")
        store1.clear_session(sid)

        import vetinari.async_support.conversation as _conv_mod

        with _conv_mod._store_lock:
            _conv_mod._store = None

        store2 = ConversationStore()
        # The session is gone from SQLite; get_history should raise KeyError.
        with pytest.raises(KeyError):
            store2.get_history("clear-session")

    def test_evicted_session_reloads_from_sqlite(self):
        """A session evicted from the LRU cache is transparently re-loaded on access."""
        store = ConversationStore()
        sid = store.create_session("evict-session")
        store.add_message(sid, "user", "before eviction")

        # Manually evict from in-memory dict without touching SQLite.
        with store._lock:
            del store._sessions[sid]

        # get_history must load from SQLite transparently.
        history = store.get_history(sid)
        assert len(history) == 1
        assert history[0].content == "before eviction"

    def test_get_context_window_reloads_evicted_session(self):
        """get_context_window re-loads an evicted session from SQLite."""
        store = ConversationStore()
        sid = store.create_session("evict-ctx-session")
        store.add_message(sid, "user", "context message")

        with store._lock:
            del store._sessions[sid]

        window = store.get_context_window(sid, max_tokens=1000)
        assert len(window) == 1
        assert window[0].content == "context message"

    def test_get_history_raises_for_truly_missing_session(self):
        """get_history raises KeyError for a session that was never created."""
        store = ConversationStore()
        with pytest.raises(KeyError):
            store.get_history("session-that-never-existed-xyz")

    def test_multiple_sessions_all_restored(self):
        """All sessions written in one store are available in a fresh store."""
        store1 = ConversationStore()
        for i in range(5):
            sid = f"multi-session-{i}"
            store1.create_session(sid)
            store1.add_message(sid, "user", f"message for session {i}")

        import vetinari.async_support.conversation as _conv_mod

        with _conv_mod._store_lock:
            _conv_mod._store = None

        store2 = ConversationStore()
        for i in range(5):
            history = store2.get_history(f"multi-session-{i}")
            assert len(history) == 1
            assert f"session {i}" in history[0].content

    def test_message_order_preserved_on_reload(self):
        """Messages are restored in chronological order after a restart."""
        store1 = ConversationStore()
        sid = store1.create_session("order-session")
        for i in range(6):
            store1.add_message(sid, "user", f"msg-{i}")

        import vetinari.async_support.conversation as _conv_mod

        with _conv_mod._store_lock:
            _conv_mod._store = None

        store2 = ConversationStore()
        history = store2.get_history("order-session", limit=0)
        contents = [m.content for m in history]
        assert contents == [f"msg-{i}" for i in range(6)]
