"""Tests for RequestQueue production leveling (Dept 4.7)."""

from __future__ import annotations

import pytest

from vetinari.orchestration.request_routing import (
    PRIORITY_CUSTOM,
    PRIORITY_EXPRESS,
    PRIORITY_REWORK,
    PRIORITY_STANDARD,
    RequestQueue,
)


class TestRequestQueue:
    """Tests for RequestQueue."""

    def test_enqueue_returns_exec_id(self):
        """enqueue() returns a non-empty execution ID."""
        q = RequestQueue(max_concurrent=3)
        exec_id = q.enqueue("fix a bug", {})
        assert exec_id
        assert isinstance(exec_id, str)

    def test_dequeue_returns_request(self):
        """dequeue() returns the enqueued request."""
        q = RequestQueue(max_concurrent=3)
        q.enqueue("fix a bug", {"key": "val"})
        result = q.dequeue()
        assert isinstance(result, tuple)
        assert len(result) == 3
        _exec_id, goal, context = result
        assert goal == "fix a bug"
        assert context == {"key": "val"}

    def test_dequeue_empty_returns_none(self):
        """dequeue() on empty queue returns None."""
        q = RequestQueue(max_concurrent=3)
        assert q.dequeue() is None

    def test_concurrency_limit(self):
        """dequeue() returns None when at max_concurrent."""
        q = RequestQueue(max_concurrent=1)
        q.enqueue("task 1", {})
        q.enqueue("task 2", {})
        result1 = q.dequeue()
        assert isinstance(result1, tuple)
        assert len(result1) == 3
        # At limit now
        result2 = q.dequeue()
        assert result2 is None

    def test_complete_frees_slot(self):
        """complete() decrements active count allowing next dequeue."""
        q = RequestQueue(max_concurrent=1)
        q.enqueue("task 1", {})
        q.enqueue("task 2", {})
        result1 = q.dequeue()
        assert isinstance(result1, tuple)
        assert len(result1) == 3
        exec_id1 = result1[0]

        # At limit
        assert q.dequeue() is None

        # Complete first task
        q.complete(exec_id1)
        result2 = q.dequeue()
        assert isinstance(result2, tuple)
        assert len(result2) == 3

    def test_priority_ordering(self):
        """Higher priority requests dequeued first."""
        q = RequestQueue(max_concurrent=10)
        q.enqueue("custom task", {}, priority=PRIORITY_CUSTOM)
        q.enqueue("express task", {}, priority=PRIORITY_EXPRESS)
        q.enqueue("rework task", {}, priority=PRIORITY_REWORK)
        q.enqueue("standard task", {}, priority=PRIORITY_STANDARD)

        # Should come out in priority order: rework(2), express(3), standard(5), custom(7)
        r1 = q.dequeue()
        r2 = q.dequeue()
        r3 = q.dequeue()
        r4 = q.dequeue()

        assert r1[1] == "rework task"
        assert r2[1] == "express task"
        assert r3[1] == "standard task"
        assert r4[1] == "custom task"

    def test_fifo_within_same_priority(self):
        """Equal priority requests dequeued in FIFO order."""
        q = RequestQueue(max_concurrent=10)
        q.enqueue("first", {}, priority=5)
        q.enqueue("second", {}, priority=5)
        q.enqueue("third", {}, priority=5)

        r1 = q.dequeue()
        r2 = q.dequeue()
        r3 = q.dequeue()

        assert r1[1] == "first"
        assert r2[1] == "second"
        assert r3[1] == "third"

    def test_depth_property(self):
        """depth returns count of waiting requests."""
        q = RequestQueue(max_concurrent=10)
        assert q.depth == 0
        q.enqueue("a", {})
        q.enqueue("b", {})
        assert q.depth == 2
        q.dequeue()
        assert q.depth == 1

    def test_active_count_property(self):
        """active_count tracks executing requests."""
        q = RequestQueue(max_concurrent=5)
        assert q.active_count == 0
        q.enqueue("a", {})
        q.dequeue()
        assert q.active_count == 1

    def test_priority_constants(self):
        """Priority constants have correct relative ordering."""
        assert PRIORITY_REWORK < PRIORITY_EXPRESS
        assert PRIORITY_EXPRESS < PRIORITY_STANDARD
        assert PRIORITY_STANDARD < PRIORITY_CUSTOM

    def test_complete_unknown_exec_id_does_not_free_slot(self):
        """complete() with an unknown exec_id must not decrement active_count.

        Regression for Bug #25/#26: spurious slot release allowed extra work
        through the concurrency gate when callers supplied a phantom exec_id.
        """
        q = RequestQueue(max_concurrent=1)
        q.enqueue("task", {})
        q.dequeue()
        assert q.active_count == 1

        # Calling complete with an ID that was never dequeued must be a no-op
        q.complete("phantom-id-that-was-never-dequeued")
        assert q.active_count == 1, "Phantom complete() must not free a slot"

    def test_complete_double_call_does_not_free_extra_slot(self):
        """A second complete() for the same exec_id must be idempotent.

        Regression for Bug #25/#26: double-complete previously decremented
        active_count twice, freeing capacity that should remain consumed.
        """
        q = RequestQueue(max_concurrent=1)
        q.enqueue("task", {})
        result = q.dequeue()
        assert result is not None
        exec_id = result[0]

        q.complete(exec_id)
        assert q.active_count == 0

        # Second complete for the same exec_id must be a no-op
        q.complete(exec_id)
        assert q.active_count == 0, "Double complete() must not underflow active_count"

    def test_thread_safety(self):
        """Concurrent enqueue/dequeue from multiple threads."""
        import threading

        q = RequestQueue(max_concurrent=5, max_depth=25)
        errors = []

        def enqueuer(idx):
            try:
                q.enqueue(f"task-{idx}", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=enqueuer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert q.depth == 20
