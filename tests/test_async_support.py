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

from vetinari.async_support.async_executor import AsyncExecutor
from vetinari.async_support.conversation import (
    ConversationMessage,
    ConversationStore,
    ContextReconstructor,
    _reset_conversation_store,
    get_conversation_store,
)
from vetinari.async_support.streaming import (
    BufferedStreamHandler,
    LoggingStreamHandler,
    SSEStreamHandler,
    StreamChunk,
    StreamRouter,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously (compatible with Python 3.10+)."""
    return asyncio.run(coro)


# ===========================================================================
# AsyncExecutor tests
# ===========================================================================


class TestAsyncExecutorSingleTask(unittest.TestCase):
    """execute_task returns a completed result dict for a basic task."""

    def setUp(self):
        self.executor = AsyncExecutor()

    def test_single_task_status_completed(self):
        task = {"id": "t1", "prompt": "do something", "agent_type": "BUILDER"}
        result = _run(self.executor.execute_task(task, "BUILDER"))
        self.assertEqual(result["status"], "completed")

    def test_single_task_id_preserved(self):
        task = {"id": "task-abc", "agent_type": "PLANNER"}
        result = _run(self.executor.execute_task(task, "PLANNER"))
        self.assertEqual(result["task_id"], "task-abc")

    def test_single_task_agent_type_preserved(self):
        task = {"id": "t2", "agent_type": "ORACLE"}
        result = _run(self.executor.execute_task(task, "ORACLE"))
        self.assertEqual(result["agent_type"], "ORACLE")

    def test_single_task_missing_id_defaults_to_unknown(self):
        result = _run(self.executor.execute_task({}, "PLANNER"))
        self.assertEqual(result["task_id"], "unknown")

    def test_single_task_exception_returns_failed_status(self):
        """Patch _dispatch to raise; execute_task must return status='failed'."""

        async def _bad_dispatch(task, agent_type):
            raise RuntimeError("simulated failure")

        self.executor._dispatch = _bad_dispatch
        task = {"id": "err-task", "agent_type": "BUILDER"}
        result = _run(self.executor.execute_task(task, "BUILDER"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("simulated failure", result["error"])


class TestAsyncExecutorTimeout(unittest.TestCase):
    """execute_task handles timeouts correctly."""

    def test_task_timeout_returns_failed(self):
        executor = AsyncExecutor(task_timeout=0)  # 0 s forces immediate timeout

        async def _slow_dispatch(task, agent_type):
            await asyncio.sleep(10)
            return {"task_id": "x", "agent_type": agent_type, "status": "completed", "output": ""}

        executor._dispatch = _slow_dispatch
        task = {"id": "slow", "agent_type": "BUILDER"}
        result = _run(executor.execute_task(task, "BUILDER"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Timeout", result["error"])


class TestAsyncExecutorWave(unittest.TestCase):
    """execute_wave runs tasks concurrently and collects all results."""

    def setUp(self):
        self.executor = AsyncExecutor()

    def test_wave_returns_result_for_each_task(self):
        tasks = [
            {"id": "w1", "agent_type": "BUILDER"},
            {"id": "w2", "agent_type": "ORACLE"},
            {"id": "w3", "agent_type": "PLANNER"},
        ]
        results = _run(self.executor.execute_wave(tasks))
        self.assertEqual(len(results), 3)

    def test_wave_empty_returns_empty_list(self):
        results = _run(self.executor.execute_wave([]))
        self.assertEqual(results, [])

    def test_wave_all_completed_by_default(self):
        tasks = [{"id": f"t{i}", "agent_type": "BUILDER"} for i in range(5)]
        results = _run(self.executor.execute_wave(tasks))
        statuses = {r["status"] for r in results}
        self.assertEqual(statuses, {"completed"})

    def test_wave_uses_agent_type_from_task_dict(self):
        tasks = [{"id": "a", "agent_type": "QUALITY"}]
        results = _run(self.executor.execute_wave(tasks))
        self.assertEqual(results[0]["agent_type"], "QUALITY")


class TestAsyncExecutorPlan(unittest.TestCase):
    """execute_plan processes waves sequentially."""

    def setUp(self):
        self.executor = AsyncExecutor()

    def _make_plan(self, wave_sizes: list[int]) -> dict:
        waves = []
        task_counter = 0
        for size in wave_sizes:
            tasks = [
                {"id": f"t{task_counter + i}", "agent_type": "BUILDER"}
                for i in range(size)
            ]
            task_counter += size
            waves.append({"wave_index": len(waves), "tasks": tasks})
        return {"id": "plan-1", "waves": waves}

    def test_plan_empty_waves_returns_completed(self):
        result = _run(self.executor.execute_plan({"id": "empty", "waves": []}))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_tasks"], 0)

    def test_plan_single_wave(self):
        plan = self._make_plan([3])
        result = _run(self.executor.execute_plan(plan))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_tasks"], 3)

    def test_plan_multi_wave_task_count(self):
        plan = self._make_plan([2, 3, 1])
        result = _run(self.executor.execute_plan(plan))
        self.assertEqual(result["total_tasks"], 6)

    def test_plan_wave_results_length_matches_wave_count(self):
        plan = self._make_plan([1, 2])
        result = _run(self.executor.execute_plan(plan))
        self.assertEqual(len(result["wave_results"]), 2)


# ===========================================================================
# StreamChunk tests
# ===========================================================================


class TestStreamChunk(unittest.TestCase):
    def test_dataclass_fields(self):
        chunk = StreamChunk(content="hello", chunk_index=0, is_final=False)
        self.assertEqual(chunk.content, "hello")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertFalse(chunk.is_final)
        self.assertEqual(chunk.metadata, {})

    def test_metadata_field(self):
        chunk = StreamChunk("x", 1, True, metadata={"model": "gpt-4"})
        self.assertEqual(chunk.metadata["model"], "gpt-4")


# ===========================================================================
# SSEStreamHandler tests
# ===========================================================================


class TestSSEStreamHandler(unittest.TestCase):
    def setUp(self):
        self.handler = SSEStreamHandler()

    def test_non_final_chunk_formats_as_data_event(self):
        _run(self.handler.on_chunk(StreamChunk("Hi", 0, False)))
        self.assertTrue(self.handler.events[0].startswith("data:"))

    def test_final_chunk_formats_as_done_event(self):
        _run(self.handler.on_chunk(StreamChunk("End", 1, True)))
        self.assertIn("event: done", self.handler.events[0])

    def test_on_complete_appends_complete_event(self):
        _run(self.handler.on_complete("full text"))
        self.assertIn("event: complete", self.handler.events[0])

    def test_on_error_appends_error_event(self):
        _run(self.handler.on_error(ValueError("boom")))
        self.assertIn("event: error", self.handler.events[0])
        self.assertIn("boom", self.handler.events[0])

    def test_multiple_chunks_accumulate(self):
        _run(self.handler.on_chunk(StreamChunk("A", 0, False)))
        _run(self.handler.on_chunk(StreamChunk("B", 1, True)))
        self.assertEqual(len(self.handler.events), 2)


# ===========================================================================
# BufferedStreamHandler tests
# ===========================================================================


class TestBufferedStreamHandler(unittest.TestCase):
    def setUp(self):
        self.handler = BufferedStreamHandler()

    def test_on_complete_sets_buffer(self):
        _run(self.handler.on_chunk(StreamChunk("Hello ", 0, False)))
        _run(self.handler.on_chunk(StreamChunk("world", 1, True)))
        _run(self.handler.on_complete("Hello world"))
        self.assertEqual(self.handler.buffer, "Hello world")
        self.assertTrue(self.handler.completed)

    def test_on_error_records_error(self):
        exc = RuntimeError("fail")
        _run(self.handler.on_error(exc))
        self.assertIs(self.handler.error, exc)
        self.assertFalse(self.handler.completed)


# ===========================================================================
# StreamRouter tests
# ===========================================================================


class TestStreamRouter(unittest.TestCase):
    def test_route_chunk_fans_out_to_all_handlers(self):
        router = StreamRouter()
        h1 = BufferedStreamHandler()
        h2 = SSEStreamHandler()
        router.add_handler(h1)
        router.add_handler(h2)

        chunk = StreamChunk("test", 0, False)
        _run(router.route_chunk(chunk))

        # h1 accumulated the chunk internally
        self.assertEqual(h1._chunks, ["test"])
        # h2 formatted it as SSE
        self.assertEqual(len(h2.events), 1)

    def test_route_complete_fans_out(self):
        router = StreamRouter()
        h1 = BufferedStreamHandler()
        router.add_handler(h1)
        _run(router.route_complete("full"))
        self.assertTrue(h1.completed)
        self.assertEqual(h1.buffer, "full")

    def test_failing_handler_does_not_block_others(self):
        """A handler that raises on_chunk must not prevent other handlers from receiving."""
        router = StreamRouter()

        bad = MagicMock(spec=SSEStreamHandler)
        bad.on_chunk = AsyncMock(side_effect=RuntimeError("handler error"))

        good = BufferedStreamHandler()
        router.add_handler(bad)
        router.add_handler(good)

        _run(router.route_chunk(StreamChunk("safe", 0, False)))
        self.assertEqual(good._chunks, ["safe"])


# ===========================================================================
# ConversationStore tests
# ===========================================================================


class TestConversationStore(unittest.TestCase):
    def setUp(self):
        _reset_conversation_store()
        self.store = ConversationStore()

    def test_create_session_returns_id(self):
        sid = self.store.create_session()
        self.assertIsInstance(sid, str)
        self.assertGreater(len(sid), 0)

    def test_create_session_with_explicit_id(self):
        sid = self.store.create_session("my-session")
        self.assertEqual(sid, "my-session")

    def test_create_duplicate_session_raises(self):
        self.store.create_session("dup")
        with self.assertRaises(ValueError):
            self.store.create_session("dup")

    def test_add_and_get_history(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "Hello")
        self.store.add_message(sid, "assistant", "Hi there")
        history = self.store.get_history(sid)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")

    def test_get_history_limit(self):
        sid = self.store.create_session()
        for i in range(10):
            self.store.add_message(sid, "user", f"msg {i}")
        history = self.store.get_history(sid, limit=3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1].content, "msg 9")

    def test_add_message_to_unknown_session_raises(self):
        with self.assertRaises(KeyError):
            self.store.add_message("nonexistent", "user", "hello")

    def test_clear_session(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "x")
        self.store.clear_session(sid)
        self.assertEqual(self.store.get_history(sid), [])

    def test_list_sessions(self):
        self.store.create_session("alpha")
        self.store.create_session("beta")
        sessions = self.store.list_sessions()
        self.assertIn("alpha", sessions)
        self.assertIn("beta", sessions)

    def test_get_context_window_token_limit(self):
        sid = self.store.create_session()
        # 4 chars per token * 10 tokens = 40 chars budget
        # each message ~10 chars content; max_tokens=10 should fit ~4 messages
        for i in range(20):
            self.store.add_message(sid, "user", f"msg{i:05d}")  # 8 chars each
        window = self.store.get_context_window(sid, max_tokens=10)
        # With budget=40 chars and 8 chars/msg, we fit at most 5 messages
        self.assertLessEqual(len(window), 5)

    def test_singleton_returns_same_instance(self):
        _reset_conversation_store()
        s1 = get_conversation_store()
        s2 = get_conversation_store()
        self.assertIs(s1, s2)


# ===========================================================================
# ContextReconstructor tests
# ===========================================================================


class TestContextReconstructor(unittest.TestCase):
    def setUp(self):
        self.rc = ContextReconstructor()

    def _make_msg(self, role: str, content: str) -> ConversationMessage:
        import time
        return ConversationMessage(role=role, content=content, timestamp=time.time())

    def test_empty_messages_returns_system_header(self):
        result = self.rc.reconstruct([])
        self.assertIn("helpful AI assistant", result)

    def test_messages_are_included_in_output(self):
        msgs = [
            self._make_msg("user", "Hello"),
            self._make_msg("assistant", "Hi there"),
        ]
        result = self.rc.reconstruct(msgs)
        self.assertIn("USER: Hello", result)
        self.assertIn("ASSISTANT: Hi there", result)

    def test_truncation_omits_old_messages(self):
        msgs = [self._make_msg("user", "x" * 500) for _ in range(20)]
        result = self.rc.reconstruct(msgs, max_tokens=50)
        self.assertIn("earlier message", result.lower())

    def test_output_starts_with_system_header(self):
        result = self.rc.reconstruct([self._make_msg("user", "hi")])
        self.assertTrue(result.startswith("You are a helpful AI assistant"))


if __name__ == "__main__":
    unittest.main()
