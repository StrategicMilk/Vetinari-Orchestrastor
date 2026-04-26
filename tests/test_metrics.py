"""Tests for vetinari.metrics module."""

from __future__ import annotations

import threading

import pytest

from vetinari.metrics import (
    MetricsCollector,
    get_metrics,
    increment_api_request,
    increment_task_count,
    record_model_latency,
    record_task_duration,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_increment_creates_counter(self) -> None:
        """Increment on a new key initialises counter to the value."""
        mc = MetricsCollector()
        mc.increment("my.counter")
        assert mc.get_counter("my.counter") == 1

    def test_increment_accumulates(self) -> None:
        """Multiple increments accumulate correctly."""
        mc = MetricsCollector()
        mc.increment("my.counter", 3)
        mc.increment("my.counter", 2)
        assert mc.get_counter("my.counter") == 5

    def test_increment_with_tags_uses_separate_key(self) -> None:
        """Tag-qualified counters are stored under distinct keys."""
        mc = MetricsCollector()
        mc.increment("req", status=200)
        mc.increment("req", status=500)
        assert mc.get_counter("req", status=200) == 1
        assert mc.get_counter("req", status=500) == 1
        assert mc.get_counter("req") == 0

    def test_get_counter_missing_returns_zero(self) -> None:
        """Missing counter key returns 0, not KeyError."""
        mc = MetricsCollector()
        assert mc.get_counter("nonexistent") == 0

    def test_record_histogram_value(self) -> None:
        """Recording values makes them visible in histogram stats."""
        mc = MetricsCollector()
        mc.record("latency", 10.0)
        mc.record("latency", 20.0)
        stats = mc.get_histogram_stats("latency")
        assert stats is not None
        assert stats["count"] == 2
        assert stats["min"] == 10.0
        assert stats["max"] == 20.0
        assert stats["avg"] == 15.0

    def test_get_histogram_stats_no_data_returns_none(self) -> None:
        """Stats for an unrecorded metric return None."""
        mc = MetricsCollector()
        assert mc.get_histogram_stats("nothing") is None

    def test_histogram_percentile_keys_present(self) -> None:
        """Stats dict contains p50, p95, p99 keys."""
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record("vals", float(i))
        stats = mc.get_histogram_stats("vals")
        assert stats is not None
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_make_key_no_tags(self) -> None:
        """Key without tags is just the metric name."""
        mc = MetricsCollector()
        assert mc._make_key("my.metric", {}) == "my.metric"

    def test_make_key_with_tags_is_deterministic(self) -> None:
        """Tag order does not affect the generated key."""
        mc = MetricsCollector()
        key_ab = mc._make_key("m", {"a": "1", "b": "2"})
        key_ba = mc._make_key("m", {"b": "2", "a": "1"})
        assert key_ab == key_ba

    def test_thread_safety_increment(self) -> None:
        """Concurrent increments from multiple threads produce correct total."""
        mc = MetricsCollector()
        threads = [threading.Thread(target=lambda: mc.increment("t")) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert mc.get_counter("t") == 100


class TestGetMetrics:
    """Tests for the get_metrics() singleton accessor."""

    def test_returns_metrics_collector_instance(self) -> None:
        """get_metrics() returns a MetricsCollector."""
        assert isinstance(get_metrics(), MetricsCollector)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        """get_metrics() is idempotent — same object every time."""
        assert get_metrics() is get_metrics()


class TestConvenienceFunctions:
    """Tests for module-level helper functions."""

    def test_record_task_duration_stores_value(self) -> None:
        """record_task_duration() writes to the global collector.

        The implementation tags the metric with task_type="generic", so
        stats must be queried with that same tag.
        """
        before = get_metrics().get_histogram_stats("vetinari.task.duration", task_type="generic")
        before_count = before["count"] if before else 0
        record_task_duration("task-1", 99.0)
        after = get_metrics().get_histogram_stats("vetinari.task.duration", task_type="generic")
        assert after is not None
        assert after["count"] == before_count + 1

    def test_increment_task_count_increments_counter(self) -> None:
        """increment_task_count() increments the global task counter."""
        before = get_metrics().get_counter("vetinari.task.count", status="completed")
        increment_task_count("completed")
        after = get_metrics().get_counter("vetinari.task.count", status="completed")
        assert after == before + 1

    def test_record_model_latency_stores_value(self) -> None:
        """record_model_latency() writes to the global collector."""
        before = get_metrics().get_histogram_stats("vetinari.model.latency")
        before_count = before["count"] if before else 0
        record_model_latency(55.5)
        after = get_metrics().get_histogram_stats("vetinari.model.latency")
        assert after is not None
        assert after["count"] == before_count + 1

    def test_increment_api_request_increments_counter(self) -> None:
        """increment_api_request() increments the global API counter."""
        before = get_metrics().get_counter("vetinari.api.request", status=200)
        increment_api_request(200)
        after = get_metrics().get_counter("vetinari.api.request", status=200)
        assert after == before + 1
