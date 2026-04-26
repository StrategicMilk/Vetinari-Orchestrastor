"""Tests for per-pool WIP limits (item 7.14).

Covers set_pool_limit(), start_pool_task(), complete_pool_task(),
get_pool_utilization(), and get_all_pool_utilizations() on WIPTracker.
Backward compatibility with the per-agent-type API is verified.
"""

from __future__ import annotations

import pytest

from vetinari.types import AgentType
from vetinari.workflow import WIPConfig, WIPTracker


class TestAgentTypeLimitsBackwardCompat:
    """Existing per-agent-type WIP API must work unchanged."""

    def test_can_start_within_limit(self) -> None:
        """can_start returns True when agent type is below its limit."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 2, "default": 4}))
        assert tracker.can_start(AgentType.WORKER.value) is True

    def test_can_start_at_limit_returns_false(self) -> None:
        """can_start returns False when the agent type is at capacity."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 4}))
        tracker.start_task(AgentType.WORKER.value, "t1")
        assert tracker.can_start(AgentType.WORKER.value) is False

    def test_complete_task_releases_slot(self) -> None:
        """complete_task frees the slot so can_start returns True again."""
        tracker = WIPTracker(WIPConfig(limits={AgentType.WORKER.value: 1, "default": 4}))
        tracker.start_task(AgentType.WORKER.value, "t1")
        tracker.complete_task(AgentType.WORKER.value, "t1")
        assert tracker.can_start(AgentType.WORKER.value) is True

    def test_get_utilization_returns_all_agent_types(self) -> None:
        """get_utilization returns a dict covering all configured agent types."""
        tracker = WIPTracker()
        util = tracker.get_utilization()
        assert AgentType.FOREMAN.value in util
        assert AgentType.WORKER.value in util
        assert AgentType.INSPECTOR.value in util


class TestSetPoolLimit:
    """set_pool_limit() configures per-pool WIP ceilings."""

    def test_pool_limit_stored(self) -> None:
        """set_pool_limit records the limit in the tracker."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 4)
        util = tracker.get_pool_utilization("gpu-pool")
        assert util["limit"] == 4

    def test_pool_limit_can_be_updated(self) -> None:
        """Calling set_pool_limit twice updates the limit."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 2)
        tracker.set_pool_limit("gpu-pool", 8)
        util = tracker.get_pool_utilization("gpu-pool")
        assert util["limit"] == 8

    def test_pool_limit_zero_is_valid(self) -> None:
        """A limit of 0 is accepted — it blocks all tasks for the pool."""
        tracker = WIPTracker()
        tracker.set_pool_limit("blocked-pool", 0)
        util = tracker.get_pool_utilization("blocked-pool")
        assert util["limit"] == 0

    def test_negative_limit_raises(self) -> None:
        """set_pool_limit raises ValueError for negative limits."""
        tracker = WIPTracker()
        with pytest.raises(ValueError, match="limit must be >= 0"):
            tracker.set_pool_limit("bad-pool", -1)


class TestStartPoolTask:
    """start_pool_task() claims a pool slot for a task."""

    def test_start_pool_task_succeeds_within_limit(self) -> None:
        """start_pool_task returns True when the pool has capacity."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 2)
        result = tracker.start_pool_task("gpu-pool", "task-1")
        assert result is True

    def test_start_pool_task_fails_at_capacity(self) -> None:
        """start_pool_task returns False when the pool is full."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 1)
        tracker.start_pool_task("gpu-pool", "task-1")
        result = tracker.start_pool_task("gpu-pool", "task-2")
        assert result is False

    def test_start_pool_task_fails_for_unknown_pool(self) -> None:
        """start_pool_task returns False when pool was never configured."""
        tracker = WIPTracker()
        result = tracker.start_pool_task("unconfigured-pool", "task-1")
        assert result is False

    def test_start_pool_task_increments_current(self) -> None:
        """Active slot count increases after start_pool_task."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 4)
        tracker.start_pool_task("gpu-pool", "task-1")
        tracker.start_pool_task("gpu-pool", "task-2")
        util = tracker.get_pool_utilization("gpu-pool")
        assert util["current"] == 2

    def test_start_pool_task_at_zero_limit_fails(self) -> None:
        """start_pool_task returns False when pool limit is 0."""
        tracker = WIPTracker()
        tracker.set_pool_limit("blocked-pool", 0)
        result = tracker.start_pool_task("blocked-pool", "task-1")
        assert result is False


class TestCompletePoolTask:
    """complete_pool_task() releases a pool slot."""

    def test_complete_pool_task_frees_slot(self) -> None:
        """Completing a pool task decreases the active count."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 2)
        tracker.start_pool_task("gpu-pool", "task-1")
        tracker.complete_pool_task("gpu-pool", "task-1")
        util = tracker.get_pool_utilization("gpu-pool")
        assert util["current"] == 0

    def test_complete_pool_task_returns_true_on_success(self) -> None:
        """complete_pool_task returns True when the task was active."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 2)
        tracker.start_pool_task("gpu-pool", "task-1")
        result = tracker.complete_pool_task("gpu-pool", "task-1")
        assert result is True

    def test_complete_pool_task_returns_false_when_not_active(self) -> None:
        """complete_pool_task returns False for an unknown task."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 2)
        result = tracker.complete_pool_task("gpu-pool", "never-started")
        assert result is False

    def test_after_complete_new_task_can_start(self) -> None:
        """After completing a task the freed slot allows a new start."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 1)
        tracker.start_pool_task("gpu-pool", "task-1")
        tracker.complete_pool_task("gpu-pool", "task-1")
        result = tracker.start_pool_task("gpu-pool", "task-2")
        assert result is True


class TestGetPoolUtilization:
    """get_pool_utilization() returns current/limit/available for one pool."""

    def test_utilization_structure(self) -> None:
        """Return dict has keys current, limit, and available."""
        tracker = WIPTracker()
        tracker.set_pool_limit("my-pool", 5)
        util = tracker.get_pool_utilization("my-pool")
        assert set(util.keys()) == {"current", "limit", "available"}

    def test_utilization_values_before_any_tasks(self) -> None:
        """A freshly configured pool shows current=0 and available=limit."""
        tracker = WIPTracker()
        tracker.set_pool_limit("my-pool", 5)
        util = tracker.get_pool_utilization("my-pool")
        assert util["current"] == 0
        assert util["limit"] == 5
        assert util["available"] == 5

    def test_utilization_after_partial_use(self) -> None:
        """available = limit - current after some tasks are started."""
        tracker = WIPTracker()
        tracker.set_pool_limit("my-pool", 4)
        tracker.start_pool_task("my-pool", "t1")
        tracker.start_pool_task("my-pool", "t2")
        util = tracker.get_pool_utilization("my-pool")
        assert util["current"] == 2
        assert util["available"] == 2

    def test_utilization_unknown_pool_returns_zeros(self) -> None:
        """Querying an unconfigured pool returns zeros rather than raising."""
        tracker = WIPTracker()
        util = tracker.get_pool_utilization("no-such-pool")
        assert util["current"] == 0
        assert util["limit"] == 0
        assert util["available"] == 0

    def test_available_never_negative(self) -> None:
        """available is clamped to 0 even in degenerate state."""
        tracker = WIPTracker()
        tracker.set_pool_limit("p", 2)
        tracker.start_pool_task("p", "t1")
        tracker.start_pool_task("p", "t2")
        util = tracker.get_pool_utilization("p")
        assert util["available"] >= 0


class TestGetAllPoolUtilizations:
    """get_all_pool_utilizations() returns utilization for every configured pool."""

    def test_empty_when_no_pools_configured(self) -> None:
        """Returns empty dict when no pools have been configured."""
        tracker = WIPTracker()
        assert tracker.get_all_pool_utilizations() == {}

    def test_all_pools_present(self) -> None:
        """All configured pools appear in the result."""
        tracker = WIPTracker()
        tracker.set_pool_limit("gpu-pool", 4)
        tracker.set_pool_limit("api-quota", 10)
        all_utils = tracker.get_all_pool_utilizations()
        assert "gpu-pool" in all_utils
        assert "api-quota" in all_utils

    def test_sorted_keys(self) -> None:
        """Pool IDs are sorted in the returned dict iteration order."""
        tracker = WIPTracker()
        for name in ("zzz-pool", "aaa-pool", "mmm-pool"):
            tracker.set_pool_limit(name, 2)
        keys = list(tracker.get_all_pool_utilizations().keys())
        assert keys == sorted(keys)

    def test_each_pool_has_correct_structure(self) -> None:
        """Every entry in get_all_pool_utilizations has the expected keys."""
        tracker = WIPTracker()
        tracker.set_pool_limit("pool-A", 3)
        tracker.set_pool_limit("pool-B", 6)
        tracker.start_pool_task("pool-A", "t1")
        all_utils = tracker.get_all_pool_utilizations()
        for util in all_utils.values():
            assert "current" in util
            assert "limit" in util
            assert "available" in util

    def test_reflects_current_state(self) -> None:
        """Utilization values reflect the live task counts."""
        tracker = WIPTracker()
        tracker.set_pool_limit("pool-A", 5)
        tracker.start_pool_task("pool-A", "t1")
        tracker.start_pool_task("pool-A", "t2")
        tracker.start_pool_task("pool-A", "t3")
        all_utils = tracker.get_all_pool_utilizations()
        assert all_utils["pool-A"]["current"] == 3
        assert all_utils["pool-A"]["available"] == 2
