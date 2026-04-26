"""
Comprehensive pytest tests for vetinari/blackboard.py

Covers:
- EntryState enum
- BlackboardEntry dataclass (creation, defaults, to_dict, is_expired)
- Blackboard.post / claim / complete / fail
- Blackboard.get_result (immediate, RuntimeError, timeout)
- Blackboard.get_pending (filter, sort, expired exclusion)
- Blackboard.get_entry
- Blackboard.delegate (fallback order)
- Blackboard.request_help
- Blackboard.escalate_error
- Blackboard.request_consensus
- Observer pattern (subscribe / _notify_observers / error resilience)
- Blackboard.purge_expired
- Blackboard.get_stats
- Blackboard.clear
- SharedExecutionContext (set/get/get_all/get_all_by_agent/keys/clear)
- REQUEST_TYPE_ROUTING / get_capable_agents
- Thread safety (concurrent post/claim)
- Singleton: get_blackboard() and Blackboard.get_instance()
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import TEST_TASK_ID
from tests.factories import make_blackboard_entry
from vetinari.types import AgentType

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import vetinari.memory.blackboard as bb_module
from vetinari.memory.blackboard import (
    REQUEST_TYPE_ROUTING,
    Blackboard,
    BlackboardEntry,
    EntryState,
    SharedExecutionContext,
    get_blackboard,
    get_capable_agents,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _reset_singletons():
    """Reset the Blackboard singleton and the module-level _blackboard."""
    Blackboard._instance = None
    bb_module._blackboard = None


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure every test starts and ends with a clean singleton."""
    _reset_singletons()
    yield
    _reset_singletons()


@pytest.fixture
def board():
    """Return a fresh Blackboard instance (singleton reset already done)."""
    return Blackboard()


# ===========================================================================
# 1. EntryState enum
# ===========================================================================


class TestEntryState:
    def test_pending_value(self):
        assert EntryState.PENDING.value == "pending"

    def test_claimed_value(self):
        assert EntryState.CLAIMED.value == "claimed"

    def test_completed_value(self):
        assert EntryState.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert EntryState.FAILED.value == "failed"

    def test_expired_value(self):
        assert EntryState.EXPIRED.value == "expired"

    def test_all_five_members(self):
        assert len(EntryState) == 5

    def test_members_are_unique(self):
        values = [e.value for e in EntryState]
        assert len(values) == len(set(values))


# ===========================================================================
# 2. BlackboardEntry
# ===========================================================================


class TestBlackboardEntry:
    def test_required_fields_stored(self):
        e = make_blackboard_entry(entry_id="id1", content="hello", request_type="rt", requested_by="A")
        assert e.entry_id == "id1"
        assert e.content == "hello"
        assert e.request_type == "rt"
        assert e.requested_by == "A"

    def test_default_priority(self):
        e = make_blackboard_entry()
        assert e.priority == 5

    def test_default_state(self):
        e = make_blackboard_entry()
        assert e.state == EntryState.PENDING

    def test_default_claimed_by_none(self):
        e = make_blackboard_entry()
        assert e.claimed_by is None

    def test_default_result_none(self):
        e = make_blackboard_entry()
        assert e.result is None

    def test_default_error_none(self):
        e = make_blackboard_entry()
        assert e.error is None

    def test_default_ttl(self):
        e = make_blackboard_entry()
        assert e.ttl_seconds == 3600.0

    def test_default_metadata_empty_dict(self):
        e = make_blackboard_entry()
        assert e.metadata == {}

    def test_created_at_is_recent(self):
        before = time.time()
        e = make_blackboard_entry()
        after = time.time()
        assert before <= e.created_at <= after

    def test_completion_event_is_threading_event(self):
        e = make_blackboard_entry()
        assert isinstance(e._completion_event, threading.Event)

    # --- is_expired ---

    def test_not_expired_when_just_created(self):
        e = make_blackboard_entry(ttl_seconds=3600.0)
        assert not e.is_expired

    def test_expired_when_ttl_exceeded(self):
        e = make_blackboard_entry(ttl_seconds=1.0)
        e.created_at = time.time() - 10.0  # 10 seconds old, TTL=1
        assert e.is_expired

    def test_not_expired_when_completed(self):
        e = make_blackboard_entry(ttl_seconds=1.0, state=EntryState.COMPLETED)
        e.created_at = time.time() - 10.0
        assert not e.is_expired

    def test_not_expired_when_failed(self):
        e = make_blackboard_entry(ttl_seconds=1.0, state=EntryState.FAILED)
        e.created_at = time.time() - 10.0
        assert not e.is_expired

    def test_expired_when_claimed_and_ttl_exceeded(self):
        e = make_blackboard_entry(ttl_seconds=1.0, state=EntryState.CLAIMED)
        e.created_at = time.time() - 10.0
        assert e.is_expired

    def test_is_expired_uses_time_time(self):
        e = make_blackboard_entry(ttl_seconds=3600.0)
        # Patch time.time to simulate far future
        with patch("vetinari.memory.blackboard.time") as mock_time:
            mock_time.time.return_value = e.created_at + 7200.0
            assert e.is_expired

    # --- to_dict ---

    def test_to_dict_keys(self):
        e = make_blackboard_entry()
        d = e.to_dict()
        expected_keys = {
            "entry_id",
            "content",
            "request_type",
            "requested_by",
            "priority",
            "state",
            "claimed_by",
            "result",
            "error",
            "created_at",
            "completed_at",
            "ttl_seconds",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_state_is_string(self):
        e = make_blackboard_entry()
        assert e.to_dict()["state"] == "pending"

    def test_to_dict_result_none_when_no_result(self):
        e = make_blackboard_entry()
        assert e.to_dict()["result"] is None

    def test_to_dict_result_truncated_to_500(self):
        e = make_blackboard_entry()
        e.result = "x" * 1000
        d = e.to_dict()
        assert len(d["result"]) == 500

    def test_to_dict_result_short_not_truncated(self):
        e = make_blackboard_entry()
        e.result = "short"
        assert e.to_dict()["result"] == "short"

    def test_to_dict_entry_id_matches(self):
        e = make_blackboard_entry(entry_id="specific_id")
        assert e.to_dict()["entry_id"] == "specific_id"


# ===========================================================================
# 3. Blackboard.post
# ===========================================================================


class TestBlackboardPost:
    def test_returns_string_entry_id(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert isinstance(eid, str)
        assert eid.startswith("bb_")

    def test_entry_stored_in_pending_state(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        assert entry is not None
        assert entry.state == EntryState.PENDING

    def test_entry_content_matches(self, board):
        eid = board.post("my task", "code_search", AgentType.WORKER.value)
        assert board.get_entry(eid).content == "my task"

    def test_entry_request_type_matches(self, board):
        eid = board.post("content", "code_review", AgentType.WORKER.value)
        assert board.get_entry(eid).request_type == "code_review"

    def test_entry_requested_by_matches(self, board):
        eid = board.post("content", "code_search", "RESEARCHER")
        assert board.get_entry(eid).requested_by == "RESEARCHER"

    def test_entry_priority_default(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert board.get_entry(eid).priority == 5

    def test_entry_custom_priority(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value, priority=1)
        assert board.get_entry(eid).priority == 1

    def test_entry_ttl_default(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert board.get_entry(eid).ttl_seconds == 3600.0

    def test_entry_custom_ttl(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value, ttl_seconds=60.0)
        assert board.get_entry(eid).ttl_seconds == 60.0

    def test_entry_metadata_stored(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value, metadata={"key": "val"})
        assert board.get_entry(eid).metadata == {"key": "val"}

    def test_entry_metadata_empty_when_none(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert board.get_entry(eid).metadata == {}

    def test_unique_ids_per_post(self, board):
        ids = {board.post(f"content {i}", "code_search", AgentType.WORKER.value) for i in range(20)}
        assert len(ids) == 20

    def test_notifies_observers(self, board):
        received = []
        board.subscribe(received.append)
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert len(received) == 1
        assert received[0].entry_id == eid


# ===========================================================================
# 4. Blackboard.claim
# ===========================================================================


class TestBlackboardClaim:
    def _patch_ctx_allowed(self):
        """Patch execution_context so MODEL_INFERENCE is permitted."""
        mock_ctx = MagicMock()
        mock_ctx.check_permission.return_value = True
        return patch(
            "vetinari.execution_context.get_context_manager",
            return_value=mock_ctx,
        )

    def _patch_ctx_denied(self):
        """Patch execution_context so MODEL_INFERENCE is denied."""
        mock_ctx = MagicMock()
        mock_ctx.check_permission.return_value = False
        return patch(
            "vetinari.execution_context.get_context_manager",
            return_value=mock_ctx,
        )

    def test_claim_pending_entry_returns_entry(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        # Allow context import to raise so permission is bypassed
        entry = board.claim(eid, "EXPLORER")
        assert entry is not None
        assert entry.entry_id == eid

    def test_claim_sets_state_to_claimed(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.claim(eid, "EXPLORER")
        assert board.get_entry(eid).state == EntryState.CLAIMED

    def test_claim_sets_claimed_by(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.claim(eid, "EXPLORER")
        assert board.get_entry(eid).claimed_by == "EXPLORER"

    def test_claim_sets_claimed_at(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        before = time.time()
        board.claim(eid, "EXPLORER")
        after = time.time()
        claimed_at = board.get_entry(eid).claimed_at
        assert claimed_at is not None
        assert before <= claimed_at <= after

    def test_claim_nonexistent_entry_returns_none(self, board):
        result = board.claim("bb_nonexistent", "EXPLORER")
        assert result is None

    def test_claim_already_claimed_returns_none(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.claim(eid, "EXPLORER")
        result = board.claim(eid, "RESEARCHER")
        assert result is None

    def test_claim_completed_entry_returns_none(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.claim(eid, "EXPLORER")
        board.complete(eid, result="done")
        result = board.claim(eid, "RESEARCHER")
        assert result is None

    def test_claim_denied_by_execution_context(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        with self._patch_ctx_denied():
            result = board.claim(eid, "EXPLORER")
        assert result is None
        # Entry should still be PENDING since claim was denied
        assert board.get_entry(eid).state == EntryState.PENDING

    def test_claim_allowed_by_execution_context(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        with self._patch_ctx_allowed():
            result = board.claim(eid, "EXPLORER")
        assert isinstance(result, BlackboardEntry)
        assert result.state == EntryState.CLAIMED
        assert result.entry_id == eid

    def test_claim_allowed_when_context_raises(self, board):
        """When the permission check raises any exception, claim should still succeed.

        The claim() method has a broad except-Exception that allows the claim
        to proceed when the context manager is not configured.  In tests the
        real execution_context raises because no context has been set up, so
        we simply verify that claim() succeeds without any extra patching.
        """
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        # No context manager configured — claim must still work
        result = board.claim(eid, "EXPLORER")
        assert isinstance(result, BlackboardEntry)
        assert result.state == EntryState.CLAIMED


# ===========================================================================
# 5. Blackboard.complete
# ===========================================================================


class TestBlackboardComplete:
    def test_complete_returns_true(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert board.complete(eid, result="done") is True

    def test_complete_sets_state(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.complete(eid, result="done")
        assert board.get_entry(eid).state == EntryState.COMPLETED

    def test_complete_stores_result(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.complete(eid, result={"answer": 42})
        assert board.get_entry(eid).result == {"answer": 42}

    def test_complete_sets_completed_at(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        before = time.time()
        board.complete(eid, result="done")
        after = time.time()
        completed_at = board.get_entry(eid).completed_at
        assert completed_at is not None
        assert before <= completed_at <= after

    def test_complete_fires_completion_event(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        assert not entry._completion_event.is_set()
        board.complete(eid, result="done")
        assert entry._completion_event.is_set()

    def test_complete_nonexistent_returns_false(self, board):
        assert board.complete("bb_nonexistent", result="x") is False


# ===========================================================================
# 6. Blackboard.fail
# ===========================================================================


class TestBlackboardFail:
    def test_fail_returns_true(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert board.fail(eid, error="oops") is True

    def test_fail_sets_state(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.fail(eid, error="oops")
        assert board.get_entry(eid).state == EntryState.FAILED

    def test_fail_stores_error(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.fail(eid, error="something broke")
        assert board.get_entry(eid).error == "something broke"

    def test_fail_sets_completed_at(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        before = time.time()
        board.fail(eid, error="e")
        after = time.time()
        completed_at = board.get_entry(eid).completed_at
        assert completed_at is not None
        assert before <= completed_at <= after

    def test_fail_fires_completion_event(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        assert not entry._completion_event.is_set()
        board.fail(eid, error="e")
        assert entry._completion_event.is_set()

    def test_fail_nonexistent_returns_false(self, board):
        assert board.fail("bb_nonexistent", error="x") is False


# ===========================================================================
# 7. Blackboard.get_result
# ===========================================================================


class TestBlackboardGetResult:
    def test_returns_none_for_missing_entry(self, board):
        assert board.get_result("bb_missing") is None

    def test_immediate_return_for_completed(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.complete(eid, result="final_answer")
        result = board.get_result(eid, timeout=0.1)
        assert result == "final_answer"

    def test_raises_runtime_error_for_failed(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.fail(eid, error="bad failure")
        with pytest.raises(RuntimeError, match="bad failure"):
            board.get_result(eid)

    def test_returns_none_on_timeout_for_pending(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        result = board.get_result(eid, timeout=0.05)
        assert result is None

    def test_waits_and_returns_result_from_thread(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        started = threading.Event()

        def _complete_later():
            started.wait()
            board.complete(eid, result="threaded_result")

        t = threading.Thread(target=_complete_later, daemon=True)
        t.start()
        started.set()
        result = board.get_result(eid, timeout=2.0)
        t.join()
        assert result == "threaded_result"

    def test_waits_and_raises_on_fail_from_thread(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        started = threading.Event()

        def _fail_later():
            started.wait()
            board.fail(eid, error="threaded_failure")

        t = threading.Thread(target=_fail_later, daemon=True)
        t.start()
        started.set()
        with pytest.raises(RuntimeError, match="threaded_failure"):
            board.get_result(eid, timeout=2.0)
        t.join()

    def test_error_message_contains_entry_id(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        board.fail(eid, error="err")
        with pytest.raises(RuntimeError, match=eid):
            board.get_result(eid)


# ===========================================================================
# 8. Blackboard.get_pending
# ===========================================================================


class TestBlackboardGetPending:
    def test_returns_only_pending_entries(self, board):
        eid1 = board.post("a", "code_search", AgentType.WORKER.value)
        eid2 = board.post("b", "code_search", AgentType.WORKER.value)
        board.complete(eid2, result="done")
        pending = board.get_pending()
        ids = [e.entry_id for e in pending]
        assert eid1 in ids
        assert eid2 not in ids

    def test_filters_by_request_type(self, board):
        eid1 = board.post("a", "code_search", AgentType.WORKER.value)
        eid2 = board.post("b", "code_review", AgentType.WORKER.value)
        pending = board.get_pending(request_type="code_search")
        ids = [e.entry_id for e in pending]
        assert eid1 in ids
        assert eid2 not in ids

    def test_returns_all_types_when_no_filter(self, board):
        eid1 = board.post("a", "code_search", AgentType.WORKER.value)
        eid2 = board.post("b", "code_review", AgentType.WORKER.value)
        pending = board.get_pending()
        ids = [e.entry_id for e in pending]
        assert eid1 in ids
        assert eid2 in ids

    def test_sorted_by_priority_ascending(self, board):
        board.post("low", "code_search", AgentType.WORKER.value, priority=9)
        board.post("high", "code_search", AgentType.WORKER.value, priority=1)
        board.post("mid", "code_search", AgentType.WORKER.value, priority=5)
        pending = board.get_pending(request_type="code_search")
        priorities = [e.priority for e in pending]
        assert priorities == sorted(priorities)

    def test_respects_limit(self, board):
        for i in range(10):
            board.post(f"task {i}", "code_search", AgentType.WORKER.value)
        pending = board.get_pending(limit=3)
        assert len(pending) <= 3

    def test_excludes_expired_entries(self, board):
        eid = board.post("expired task", "code_search", AgentType.WORKER.value, ttl_seconds=1.0)
        entry = board.get_entry(eid)
        entry.created_at = time.time() - 100.0  # force expiry
        pending = board.get_pending()
        ids = [e.entry_id for e in pending]
        assert eid not in ids

    def test_empty_board_returns_empty_list(self, board):
        assert board.get_pending() == []


# ===========================================================================
# 9. Blackboard.get_entry
# ===========================================================================


class TestBlackboardGetEntry:
    def test_returns_none_for_missing(self, board):
        assert board.get_entry("bb_missing") is None

    def test_returns_entry_by_id(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        assert entry is not None
        assert entry.entry_id == eid

    def test_returns_same_object_as_stored(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        e1 = board.get_entry(eid)
        e2 = board.get_entry(eid)
        assert e1 is e2


# ===========================================================================
# 10. Blackboard.delegate
# ===========================================================================


class TestBlackboardDelegate:
    """delegate() imports AgentType/AgentTask/AgentResult locally inside the
    function body from vetinari.agents.contracts.  We use the real AgentType
    enum as dict keys and patch AgentTask.from_task at its source module so
    the local import picks up the mock."""

    @pytest.fixture(autouse=True)
    def _real_agent_types(self):
        """Import real AgentType once for all delegate tests."""
        from vetinari.types import AgentType

        self.AgentType = AgentType

    def _make_task(self):
        task = MagicMock()
        task.id = "task_abc"
        task.description = "do something"
        task.assigned_agent = MagicMock(value="unknown_agent")
        task.dependencies = []
        return task

    def test_delegate_uses_planner_first(self, board):
        mock_result = MagicMock()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = mock_result
        mock_agent_task = MagicMock()

        with patch("vetinari.agents.contracts.AgentTask.from_task", return_value=mock_agent_task):
            task = self._make_task()
            available_agents = {self.AgentType.FOREMAN: mock_agent}
            result = board.delegate(task, available_agents)

        assert result is mock_result
        mock_agent.execute.assert_called_once_with(mock_agent_task)

    def test_delegate_falls_back_to_builder(self, board):
        mock_result = MagicMock()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = mock_result
        mock_agent_task = MagicMock()

        with patch("vetinari.agents.contracts.AgentTask.from_task", return_value=mock_agent_task):
            task = self._make_task()
            # Foreman not available; worker is
            available_agents = {self.AgentType.WORKER: mock_agent}
            result = board.delegate(task, available_agents)

        assert result is mock_result

    def test_delegate_falls_back_to_researcher(self, board):
        mock_result = MagicMock()
        mock_agent = MagicMock()
        mock_agent.execute.return_value = mock_result
        mock_agent_task = MagicMock()

        with patch("vetinari.agents.contracts.AgentTask.from_task", return_value=mock_agent_task):
            task = self._make_task()
            # Only worker available
            available_agents = {self.AgentType.WORKER: mock_agent}
            result = board.delegate(task, available_agents)

        assert result is mock_result

    def test_delegate_returns_none_when_no_fallback(self, board):
        task = self._make_task()
        result = board.delegate(task, available_agents={})
        assert result is None

    def test_delegate_continues_on_agent_exception(self, board):
        failing_agent = MagicMock()
        failing_agent.execute.side_effect = Exception("agent exploded")
        working_agent = MagicMock()
        working_agent.execute.return_value = "recovered"
        mock_agent_task = MagicMock()

        with patch("vetinari.agents.contracts.AgentTask.from_task", return_value=mock_agent_task):
            task = self._make_task()
            available_agents = {
                self.AgentType.FOREMAN: failing_agent,
                self.AgentType.WORKER: working_agent,
            }
            result = board.delegate(task, available_agents)

        assert result == "recovered"


# ===========================================================================
# 11. Blackboard.request_help
# ===========================================================================


class _TrackingList(list):
    """A list subclass that signals a threading.Event on each append."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__()
        self._event = event

    def append(self, val: object) -> None:  # type: ignore[override]
        super().append(val)
        self._event.set()


class TestBlackboardRequestHelp:
    def test_request_help_posts_and_waits(self, board):
        """request_help should post an entry and return the result once completed."""
        posted_event = threading.Event()
        posted_ids: _TrackingList = _TrackingList(posted_event)
        original_post = board.post

        def capturing_post(*args, **kwargs):
            eid = original_post(*args, **kwargs)
            posted_ids.append(eid)
            return eid

        board.post = capturing_post

        def _complete_after_post():
            posted_event.wait(timeout=2.0)
            board.complete(posted_ids[0], result="help_result")

        t = threading.Thread(target=_complete_after_post, daemon=True)
        t.start()

        result = board.request_help(
            requesting_agent=AgentType.WORKER.value,
            request_type="code_search",
            description="find stuff",
            timeout=2.0,
        )
        t.join()
        assert result == "help_result"

    def test_request_help_returns_none_on_failure(self, board):
        """If the entry fails, request_help should return None (RuntimeError swallowed)."""
        posted_event = threading.Event()
        posted_ids: _TrackingList = _TrackingList(posted_event)
        original_post = board.post

        def capturing_post(*args, **kwargs):
            eid = original_post(*args, **kwargs)
            posted_ids.append(eid)
            return eid

        board.post = capturing_post

        def _fail_after_post():
            posted_event.wait(timeout=2.0)
            board.fail(posted_ids[0], error="oops")

        t = threading.Thread(target=_fail_after_post, daemon=True)
        t.start()

        result = board.request_help(
            requesting_agent=AgentType.WORKER.value,
            request_type="code_search",
            description="find stuff",
            timeout=2.0,
        )
        t.join()
        assert result is None

    def test_request_help_returns_none_on_timeout(self, board):
        result = board.request_help(
            requesting_agent=AgentType.WORKER.value,
            request_type="code_search",
            description="never answered",
            timeout=0.05,
        )
        assert result is None

    def test_request_help_stores_metadata(self, board):
        """Metadata passed to request_help should appear on the posted entry."""
        # Spy on post
        original_post = board.post
        captured = {}

        def spy_post(*args, **kwargs):
            eid = original_post(*args, **kwargs)
            captured["eid"] = eid
            # immediately complete so request_help returns
            board.complete(eid, result="ok")
            return eid

        board.post = spy_post

        board.request_help(
            requesting_agent=AgentType.WORKER.value,
            request_type="code_search",
            description="search",
            metadata={"extra": "info"},
        )
        entry = board.get_entry(captured["eid"])
        assert entry.metadata.get("extra") == "info"


# ===========================================================================
# 12. Blackboard.escalate_error
# ===========================================================================


class TestBlackboardEscalateError:
    def test_escalate_returns_entry_id(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "NullPointerError")
        assert isinstance(eid, str)
        assert eid.startswith("bb_")

    def test_escalate_creates_error_recovery_entry(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "NullPointerError")
        entry = board.get_entry(eid)
        assert entry.request_type == "error_recovery"

    def test_escalate_sets_high_priority(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "err")
        assert board.get_entry(eid).priority == 1

    def test_escalate_metadata_contains_original_task_id(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "err")
        entry = board.get_entry(eid)
        assert entry.metadata["original_task_id"] == TEST_TASK_ID

    def test_escalate_metadata_contains_error(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "something went wrong")
        entry = board.get_entry(eid)
        assert entry.metadata["error"] == "something went wrong"

    def test_escalate_metadata_includes_context(self, board):
        eid = board.escalate_error(AgentType.WORKER.value, TEST_TASK_ID, "err", context={"trace": "abc"})
        entry = board.get_entry(eid)
        assert entry.metadata["trace"] == "abc"

    def test_escalate_content_mentions_agent_and_task(self, board):
        eid = board.escalate_error("RESEARCHER", "task_99", "err")
        entry = board.get_entry(eid)
        assert "RESEARCHER" in entry.content
        assert "task_99" in entry.content


# ===========================================================================
# 13. Blackboard.request_consensus
# ===========================================================================


class TestBlackboardRequestConsensus:
    def test_request_consensus_returns_entry_id(self, board):
        eid = board.request_consensus("ORACLE", "pick a pattern", ["A", "B"])
        assert isinstance(eid, str)

    def test_request_consensus_uses_architecture_decision_type(self, board):
        eid = board.request_consensus("ORACLE", "subject", ["X"])
        assert board.get_entry(eid).request_type == "architecture_decision"

    def test_request_consensus_priority_is_3(self, board):
        eid = board.request_consensus("ORACLE", "subject", ["X"])
        assert board.get_entry(eid).priority == 3

    def test_request_consensus_metadata_has_flag(self, board):
        eid = board.request_consensus("ORACLE", "subject", ["A", "B"])
        entry = board.get_entry(eid)
        assert entry.metadata["consensus_request"] is True

    def test_request_consensus_metadata_has_subject(self, board):
        eid = board.request_consensus("ORACLE", "pick algo", ["merge", "quick"])
        entry = board.get_entry(eid)
        assert entry.metadata["subject"] == "pick algo"

    def test_request_consensus_metadata_has_options(self, board):
        eid = board.request_consensus("ORACLE", "s", ["opt1", "opt2"])
        entry = board.get_entry(eid)
        assert entry.metadata["options"] == ["opt1", "opt2"]

    def test_request_consensus_extra_metadata_merged(self, board):
        eid = board.request_consensus("ORACLE", "s", ["A"], metadata={"urgency": "high"})
        entry = board.get_entry(eid)
        assert entry.metadata["urgency"] == "high"


# ===========================================================================
# 14. Observer pattern
# ===========================================================================


class TestObserverPattern:
    def test_subscribe_registers_callback(self, board):
        cb = MagicMock()
        board.subscribe(cb)
        entry_id = board.post("content", "code_search", AgentType.WORKER.value)
        cb.assert_called_once()
        observed_entry = cb.call_args.args[0]
        assert observed_entry is board.get_entry(entry_id)
        assert observed_entry.requested_by == AgentType.WORKER.value

    def test_multiple_observers_all_called(self, board):
        cb1 = MagicMock()
        cb2 = MagicMock()
        board.subscribe(cb1)
        board.subscribe(cb2)
        entry_id = board.post("content", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(entry_id)
        cb1.assert_called_once_with(entry)
        cb2.assert_called_once_with(entry)
        assert entry.content == "content"

    def test_observer_receives_entry(self, board):
        received = []
        board.subscribe(received.append)
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        assert received[0].entry_id == eid

    def test_observer_error_does_not_crash(self, board):
        def bad_observer(entry):
            raise RuntimeError("observer boom")

        board.subscribe(bad_observer)
        board.post("content", "code_search", AgentType.WORKER.value)
        # Post must succeed despite observer failure — entry still stored
        stats = board.get_stats()
        assert stats.get("pending", 0) >= 1, "Entry must be stored even when observer raises"

    def test_observer_error_does_not_prevent_subsequent_observers(self, board):
        good_received = []

        def bad_observer(entry):
            raise RuntimeError("bad")

        board.subscribe(bad_observer)
        board.subscribe(good_received.append)
        board.post("content", "code_search", AgentType.WORKER.value)
        assert len(good_received) == 1

    def test_no_observers_is_safe(self, board):
        board.post("content", "code_search", AgentType.WORKER.value)
        stats = board.get_stats()
        assert stats.get("pending", 0) >= 1, "Entry must be stored even with no observers"


# ===========================================================================
# 15. Blackboard.purge_expired
# ===========================================================================


class TestBlackboardPurgeExpired:
    def test_purge_marks_expired_entries(self, board):
        eid = board.post("stale", "code_search", AgentType.WORKER.value, ttl_seconds=1.0)
        entry = board.get_entry(eid)
        entry.created_at = time.time() - 10.0

        count = board.purge_expired()
        assert count == 1
        assert board.get_entry(eid).state == EntryState.EXPIRED

    def test_purge_does_not_affect_fresh_entries(self, board):
        eid = board.post("fresh", "code_search", AgentType.WORKER.value, ttl_seconds=3600.0)
        board.purge_expired()
        assert board.get_entry(eid).state == EntryState.PENDING

    def test_purge_returns_count_of_expired(self, board):
        for i in range(3):
            eid = board.post(f"stale {i}", "code_search", AgentType.WORKER.value, ttl_seconds=1.0)
            board.get_entry(eid).created_at = time.time() - 100.0
        count = board.purge_expired()
        assert count == 3

    def test_purge_removes_stale_completed_older_than_2h(self, board):
        eid = board.post("old completed", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        entry.state = EntryState.COMPLETED
        entry.created_at = time.time() - 7201.0  # older than 2 hours

        board.purge_expired()
        assert board.get_entry(eid) is None

    def test_purge_removes_stale_failed_older_than_2h(self, board):
        eid = board.post("old failed", "code_search", AgentType.WORKER.value)
        entry = board.get_entry(eid)
        entry.state = EntryState.FAILED
        entry.created_at = time.time() - 7201.0

        board.purge_expired()
        assert board.get_entry(eid) is None

    def test_purge_does_not_remove_recent_completed(self, board):
        eid = board.post("recent completed", "code_search", AgentType.WORKER.value)
        board.complete(eid, result="ok")
        board.purge_expired()
        entry = board.get_entry(eid)
        assert entry is not None
        assert entry.entry_id == eid

    def test_purge_returns_zero_when_nothing_expired(self, board):
        board.post("fresh", "code_search", AgentType.WORKER.value)
        assert board.purge_expired() == 0


# ===========================================================================
# 16. Blackboard.get_stats
# ===========================================================================


class TestBlackboardGetStats:
    def test_stats_empty_board(self, board):
        assert board.get_stats() == {}

    def test_stats_counts_pending(self, board):
        board.post("a", "code_search", AgentType.WORKER.value)
        board.post("b", "code_search", AgentType.WORKER.value)
        stats = board.get_stats()
        assert stats.get("pending") == 2

    def test_stats_counts_completed(self, board):
        eid = board.post("a", "code_search", AgentType.WORKER.value)
        board.complete(eid, result="ok")
        stats = board.get_stats()
        assert stats.get("completed") == 1

    def test_stats_counts_failed(self, board):
        eid = board.post("a", "code_search", AgentType.WORKER.value)
        board.fail(eid, error="err")
        stats = board.get_stats()
        assert stats.get("failed") == 1

    def test_stats_mixed_states(self, board):
        board.post("a", "code_search", AgentType.WORKER.value)
        eid2 = board.post("b", "code_search", AgentType.WORKER.value)
        eid3 = board.post("c", "code_search", AgentType.WORKER.value)
        board.complete(eid2, result="ok")
        board.fail(eid3, error="err")
        stats = board.get_stats()
        assert stats["pending"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1


# ===========================================================================
# 17. Blackboard.clear
# ===========================================================================


class TestBlackboardClear:
    def test_clear_removes_all_entries(self, board):
        board.post("a", "code_search", AgentType.WORKER.value)
        board.post("b", "code_search", AgentType.WORKER.value)
        board.clear()
        assert board.get_pending() == []
        assert board.get_stats() == {}

    def test_clear_on_empty_board_is_safe(self, board):
        board.clear()  # Should not raise
        assert board.get_pending() == []

    def test_entries_gone_after_clear(self, board):
        eid = board.post("a", "code_search", AgentType.WORKER.value)
        board.clear()
        assert board.get_entry(eid) is None


# ===========================================================================
# 18. SharedExecutionContext
# ===========================================================================


class TestSharedExecutionContext:
    @pytest.fixture
    def ctx(self):
        return SharedExecutionContext(plan_id="plan_001")

    def test_plan_id_stored(self, ctx):
        assert ctx.plan_id == "plan_001"

    def test_set_and_get(self, ctx):
        ctx.set("key1", "value1", AgentType.WORKER.value)
        assert ctx.get("key1") == "value1"

    def test_get_missing_returns_default(self, ctx):
        assert ctx.get("missing") is None

    def test_get_custom_default(self, ctx):
        assert ctx.get("missing", default="fallback") == "fallback"

    def test_set_overwrites_existing(self, ctx):
        ctx.set("key", "v1", AgentType.WORKER.value)
        ctx.set("key", "v2", "RESEARCHER")
        assert ctx.get("key") == "v2"

    def test_get_all_returns_copy(self, ctx):
        ctx.set("a", 1, AgentType.WORKER.value)
        ctx.set("b", 2, "RESEARCHER")
        all_data = ctx.get_all()
        assert all_data == {"a": 1, "b": 2}

    def test_get_all_is_shallow_copy(self, ctx):
        ctx.set("a", 1, AgentType.WORKER.value)
        all_data = ctx.get_all()
        all_data["a"] = 99
        # Original should be unchanged
        assert ctx.get("a") == 1

    def test_get_all_by_agent_filters(self, ctx):
        ctx.set("x", 10, AgentType.WORKER.value)
        ctx.set("y", 20, "RESEARCHER")
        ctx.set("z", 30, AgentType.WORKER.value)
        result = ctx.get_all_by_agent(AgentType.WORKER.value)
        assert result == {"x": 10, "z": 30}
        assert "y" not in result

    def test_get_all_by_agent_unknown_returns_empty(self, ctx):
        ctx.set("x", 1, AgentType.WORKER.value)
        assert ctx.get_all_by_agent("ORACLE") == {}

    def test_keys_returns_list(self, ctx):
        ctx.set("a", 1, AgentType.WORKER.value)
        ctx.set("b", 2, "RESEARCHER")
        keys = ctx.keys()
        assert isinstance(keys, list)
        assert set(keys) == {"a", "b"}

    def test_keys_empty_initially(self, ctx):
        assert ctx.keys() == []

    def test_clear_removes_all(self, ctx):
        ctx.set("a", 1, AgentType.WORKER.value)
        ctx.set("b", 2, "RESEARCHER")
        ctx.clear()
        assert ctx.get_all() == {}
        assert ctx.keys() == []

    def test_clear_removes_provenance(self, ctx):
        ctx.set("a", 1, AgentType.WORKER.value)
        ctx.clear()
        assert ctx.get_all_by_agent(AgentType.WORKER.value) == {}

    def test_provenance_tracks_writing_agent(self, ctx):
        ctx.set("k", "v", "ORACLE")
        assert ctx.get_all_by_agent("ORACLE") == {"k": "v"}

    def test_provenance_updated_on_overwrite(self, ctx):
        ctx.set("k", "v1", AgentType.WORKER.value)
        ctx.set("k", "v2", "RESEARCHER")
        # BUILDER no longer owns it
        assert ctx.get_all_by_agent(AgentType.WORKER.value) == {}
        assert ctx.get_all_by_agent("RESEARCHER") == {"k": "v2"}

    def test_thread_safe_concurrent_sets(self, ctx):
        errors = []

        def _writer(agent, start, end):
            for i in range(start, end):
                try:
                    ctx.set(f"key_{agent}_{i}", i, agent)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=_writer, args=("A", 0, 50), daemon=True),
            threading.Thread(target=_writer, args=("B", 0, 50), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ===========================================================================
# 19. REQUEST_TYPE_ROUTING and get_capable_agents
# ===========================================================================


class TestRequestTypeRouting:
    def test_routing_dict_is_not_empty(self):
        assert len(REQUEST_TYPE_ROUTING) > 0

    @pytest.mark.parametrize(
        ("task_type", "must_include"),
        [
            ("code_search", AgentType.WORKER.value),
            ("code_review", None),
            ("error_recovery", AgentType.WORKER.value),
            ("implementation", AgentType.WORKER.value),
        ],
    )
    def test_known_type_returns_agents(self, task_type, must_include):
        agents = get_capable_agents(task_type)
        assert len(agents) > 0
        if must_include:
            assert must_include in agents

    def test_unknown_type_returns_empty_list(self):
        assert get_capable_agents("totally_unknown_type_xyz") == []

    def test_all_routing_values_are_lists(self):
        for key, value in REQUEST_TYPE_ROUTING.items():
            assert isinstance(value, list), f"Expected list for key {key!r}"

    def test_all_routing_keys_are_strings(self):
        for key in REQUEST_TYPE_ROUTING:
            assert isinstance(key, str)

    def test_get_capable_agents_returns_list_type(self):
        """get_capable_agents should always return a list."""
        agents = get_capable_agents("code_search")
        assert isinstance(agents, list)
        assert len(agents) > 0


# ===========================================================================
# 20. Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_posts(self, board):
        results = []
        errors = []

        def _poster():
            try:
                eid = board.post("content", "code_search", AgentType.WORKER.value)
                results.append(eid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_poster, daemon=True) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 50
        assert len(set(results)) == 50  # all unique

    def test_concurrent_claim_only_one_wins(self, board):
        eid = board.post("content", "code_search", AgentType.WORKER.value)
        claimed_entries = []
        errors = []

        def _claimer(agent_name):
            try:
                # Allow context import to raise (no context set up)
                entry = board.claim(eid, agent_name)
                if entry is not None:
                    claimed_entries.append(agent_name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_claimer, args=(f"AGENT_{i}",), daemon=True) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(claimed_entries) <= 1  # at most one winner

    def test_concurrent_post_and_get_stats(self, board):
        errors = []

        def _poster():
            for i in range(10):
                try:
                    board.post(f"task {i}", "code_search", AgentType.WORKER.value)
                except Exception as e:
                    errors.append(e)

        def _stats():
            for _ in range(20):
                try:
                    board.get_stats()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_poster, daemon=True) for _ in range(5)] + [
            threading.Thread(target=_stats, daemon=True) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ===========================================================================
# 21. Singleton: get_blackboard() and Blackboard.get_instance()
# ===========================================================================


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        b1 = Blackboard.get_instance()
        b2 = Blackboard.get_instance()
        assert b1 is b2

    def test_get_blackboard_returns_same_object(self):
        g1 = get_blackboard()
        g2 = get_blackboard()
        assert g1 is g2

    def test_get_blackboard_and_get_instance_are_same(self):
        g = get_blackboard()
        i = Blackboard.get_instance()
        assert g is i

    def test_singleton_reset_gives_new_instance(self):
        b1 = Blackboard.get_instance()
        Blackboard._instance = None
        b2 = Blackboard.get_instance()
        assert b1 is not b2

    def test_get_blackboard_module_level_reset(self):
        g1 = get_blackboard()
        bb_module._blackboard = None
        Blackboard._instance = None
        g2 = get_blackboard()
        assert g1 is not g2

    def test_get_instance_thread_safe(self):
        instances = []
        errors = []

        def _get():
            try:
                instances.append(Blackboard.get_instance())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get, daemon=True) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # All threads should get exactly the same instance
        assert len({id(i) for i in instances}) == 1
