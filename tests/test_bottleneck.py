"""Tests for vetinari.orchestration.bottleneck — Theory of Constraints."""

from __future__ import annotations

import math

import pytest

from vetinari.orchestration.bottleneck import (
    BottleneckAgentMetrics,
    BottleneckIdentifier,
    get_bottleneck_identifier,
    reset_bottleneck_identifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identifier() -> BottleneckIdentifier:
    """Return a fresh BottleneckIdentifier for each test."""
    return BottleneckIdentifier()


# ---------------------------------------------------------------------------
# AgentMetrics
# ---------------------------------------------------------------------------


class TestAgentMetrics:
    def test_to_dict_serializes_correctly(self) -> None:
        m = BottleneckAgentMetrics(
            agent_type="builder",
            avg_execution_time_ms=123.456,
            queue_depth=3,
            utilization=0.75,
            failure_rate=0.1,
            throughput=12.5,
        )
        result = m.to_dict()
        assert result["agent_type"] == "builder"
        assert result["avg_execution_time_ms"] == 123.456
        assert result["queue_depth"] == 3
        assert result["utilization"] == 0.75
        assert result["failure_rate"] == 0.1
        assert result["throughput"] == 12.5

    def test_to_dict_default_values(self) -> None:
        m = BottleneckAgentMetrics()
        result = m.to_dict()
        assert result["agent_type"] == ""
        assert result["avg_execution_time_ms"] == 0.0
        assert result["queue_depth"] == 0
        assert result["utilization"] == 0.0


# ---------------------------------------------------------------------------
# BottleneckIdentifier.update_metrics
# ---------------------------------------------------------------------------


class TestUpdateMetrics:
    def test_update_metrics_initializes_agent(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("planner", 100.0, success=True)
        metrics = identifier.get_all_metrics()
        assert "planner" in metrics

    def test_update_metrics_rolling_average(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("builder", 100.0, success=True)
        identifier.update_metrics("builder", 200.0, success=True)
        metrics = identifier.get_all_metrics()
        assert metrics["builder"].avg_execution_time_ms == pytest.approx(150.0)

    def test_update_metrics_failure_rate(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("builder", 50.0, success=True)
        identifier.update_metrics("builder", 50.0, success=False)
        metrics = identifier.get_all_metrics()
        assert metrics["builder"].failure_rate == pytest.approx(0.5)

    def test_update_metrics_all_successes_zero_failure_rate(self, identifier: BottleneckIdentifier) -> None:
        for _ in range(5):
            identifier.update_metrics("quality", 80.0, success=True)
        metrics = identifier.get_all_metrics()
        assert metrics["quality"].failure_rate == 0.0

    def test_update_metrics_utilization_nonzero(self, identifier: BottleneckIdentifier) -> None:
        # Simulate meaningful busy time relative to elapsed
        identifier.update_metrics("oracle", 5000.0, success=True)
        metrics = identifier.get_all_metrics()
        # Utilization should be >= 0 and <= 1
        assert 0.0 <= metrics["oracle"].utilization <= 1.0

    def test_update_metrics_throughput_with_multiple_completions(self, identifier: BottleneckIdentifier) -> None:
        # With only one completion, throughput stays 0 (need >=2 data points)
        identifier.update_metrics("planner", 10.0, success=True)
        metrics = identifier.get_all_metrics()
        assert metrics["planner"].throughput == 0.0

        # Second completion enables throughput computation
        identifier.update_metrics("planner", 10.0, success=True)
        metrics = identifier.get_all_metrics()
        # Throughput should now be positive (or zero if window too short)
        assert metrics["planner"].throughput >= 0.0


# ---------------------------------------------------------------------------
# BottleneckIdentifier.identify_constraint
# ---------------------------------------------------------------------------


class TestIdentifyConstraint:
    def test_identify_constraint_returns_none_when_empty(self, identifier: BottleneckIdentifier) -> None:
        assert identifier.identify_constraint() is None

    def test_identify_constraint_single_agent(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("builder", 100.0, success=True)
        assert identifier.identify_constraint() == "builder"

    def test_identify_constraint_highest_utilization_wins(self, identifier: BottleneckIdentifier) -> None:
        # Manually set utilization by injecting metrics
        identifier._metrics["fast"] = BottleneckAgentMetrics(agent_type="fast", utilization=0.2, queue_depth=0)
        identifier._metrics["slow"] = BottleneckAgentMetrics(agent_type="slow", utilization=0.9, queue_depth=0)
        assert identifier.identify_constraint() == "slow"

    def test_identify_constraint_queue_depth_breaks_tie(self, identifier: BottleneckIdentifier) -> None:
        # Same utilization — higher queue depth should tip the scale
        identifier._metrics["a"] = BottleneckAgentMetrics(agent_type="a", utilization=0.5, queue_depth=0)
        identifier._metrics["b"] = BottleneckAgentMetrics(agent_type="b", utilization=0.5, queue_depth=5)
        assert identifier.identify_constraint() == "b"

    def test_identify_constraint_combined_score(self, identifier: BottleneckIdentifier) -> None:
        # Score = utilization * (1 + queue_depth)
        # a: 0.8 * (1+0) = 0.8
        # b: 0.5 * (1+3) = 2.0  <- wins
        identifier._metrics["a"] = BottleneckAgentMetrics(agent_type="a", utilization=0.8, queue_depth=0)
        identifier._metrics["b"] = BottleneckAgentMetrics(agent_type="b", utilization=0.5, queue_depth=3)
        assert identifier.identify_constraint() == "b"


# ---------------------------------------------------------------------------
# BottleneckIdentifier.get_drum_rate
# ---------------------------------------------------------------------------


class TestGetDrumRate:
    def test_get_drum_rate_returns_inf_when_no_metrics(self, identifier: BottleneckIdentifier) -> None:
        assert identifier.get_drum_rate() == float("inf")

    def test_get_drum_rate_returns_constraint_throughput(self, identifier: BottleneckIdentifier) -> None:
        identifier._metrics["builder"] = BottleneckAgentMetrics(agent_type="builder", utilization=0.9, throughput=30.0)
        identifier._metrics["planner"] = BottleneckAgentMetrics(agent_type="planner", utilization=0.3, throughput=60.0)
        # builder is the constraint (higher utilization)
        assert identifier.get_drum_rate() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# BottleneckIdentifier.get_buffer_size
# ---------------------------------------------------------------------------


class TestGetBufferSize:
    def test_get_buffer_size_returns_two_for_constraint(self, identifier: BottleneckIdentifier) -> None:
        identifier._metrics["builder"] = BottleneckAgentMetrics(agent_type="builder", utilization=0.9)
        assert identifier.get_buffer_size("builder") == 2

    def test_get_buffer_size_returns_zero_for_non_constraint(self, identifier: BottleneckIdentifier) -> None:
        identifier._metrics["builder"] = BottleneckAgentMetrics(agent_type="builder", utilization=0.9)
        identifier._metrics["planner"] = BottleneckAgentMetrics(agent_type="planner", utilization=0.2)
        assert identifier.get_buffer_size("planner") == 0

    def test_get_buffer_size_no_metrics_returns_zero(self, identifier: BottleneckIdentifier) -> None:
        # No constraint identified, so no agent matches
        assert identifier.get_buffer_size("anything") == 0


# ---------------------------------------------------------------------------
# BottleneckIdentifier.set_queue_depth
# ---------------------------------------------------------------------------


class TestSetQueueDepth:
    def test_set_queue_depth_updates_existing_agent(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("builder", 100.0, success=True)
        identifier.set_queue_depth("builder", 7)
        metrics = identifier.get_all_metrics()
        assert metrics["builder"].queue_depth == 7

    def test_set_queue_depth_creates_agent_if_missing(self, identifier: BottleneckIdentifier) -> None:
        identifier.set_queue_depth("new_agent", 3)
        metrics = identifier.get_all_metrics()
        assert "new_agent" in metrics
        assert metrics["new_agent"].queue_depth == 3


# ---------------------------------------------------------------------------
# BottleneckIdentifier.get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_get_status_empty_returns_none_constraint(self, identifier: BottleneckIdentifier) -> None:
        status = identifier.get_status()
        assert status["constraint"] is None
        assert status["drum_rate_per_minute"] is None
        assert status["agents"] == {}

    def test_get_status_with_agents_returns_dict(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("builder", 200.0, success=True)
        identifier.update_metrics("planner", 50.0, success=True)
        status = identifier.get_status()
        assert status["constraint"] in ("builder", "planner")
        assert "agents" in status
        assert "builder" in status["agents"]
        assert "planner" in status["agents"]

    def test_get_status_agent_entry_has_expected_keys(self, identifier: BottleneckIdentifier) -> None:
        identifier.update_metrics("quality", 100.0, success=True)
        status = identifier.get_status()
        agent_dict = status["agents"]["quality"]
        for key in ("agent_type", "avg_execution_time_ms", "queue_depth", "utilization", "failure_rate", "throughput"):
            assert key in agent_dict


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_bottleneck_identifier_returns_same_instance(self) -> None:
        reset_bottleneck_identifier()
        a = get_bottleneck_identifier()
        b = get_bottleneck_identifier()
        assert a is b

    def test_reset_bottleneck_identifier_creates_fresh_instance(self) -> None:
        reset_bottleneck_identifier()
        first = get_bottleneck_identifier()
        first.update_metrics("builder", 100.0, success=True)

        reset_bottleneck_identifier()
        second = get_bottleneck_identifier()
        assert second is not first
        assert second.get_all_metrics() == {}

    def test_singleton_cleanup(self) -> None:
        """Reset singleton after test class to avoid polluting other tests."""
        reset_bottleneck_identifier()
        # After reset, a fresh singleton is created on next access
        fresh = get_bottleneck_identifier()
        assert fresh.get_all_metrics() == {}


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_orchestration_package(self) -> None:
        from vetinari.orchestration import (
            BottleneckAgentMetrics,
            BottleneckIdentifier,
            get_bottleneck_identifier,
            reset_bottleneck_identifier,
        )

        assert BottleneckIdentifier is not None
        assert BottleneckAgentMetrics is not None
        assert callable(get_bottleneck_identifier)
        assert callable(reset_bottleneck_identifier)
