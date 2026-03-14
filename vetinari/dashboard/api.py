"""Dashboard Backend API for Vetinari.

Provides REST endpoints for:
- Real-time metrics visualization
- Historical data and time-series
- Alert configuration and status
- Trace searching and exploration

The dashboard integrates telemetry data from Phase 3 and exposes it through
a RESTful API for frontend consumption.

Usage:
    from vetinari.dashboard.api import DashboardAPI

    api = DashboardAPI()
    # Returns latest metrics snapshot
    latest = api.get_latest_metrics()

    # Returns time-series data for charting
    timeseries = api.get_timeseries_data('latency', timerange='24h')
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from vetinari.telemetry import get_telemetry_collector

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Current snapshot of all system metrics."""

    timestamp: str
    uptime_ms: float

    # Adapter metrics summary
    adapter_summary: dict[str, Any]

    # Memory metrics summary
    memory_summary: dict[str, Any]

    # Plan mode metrics summary
    plan_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "uptime_ms": self.uptime_ms,
            "adapters": self.adapter_summary,
            "memory": self.memory_summary,
            "plan": self.plan_summary,
        }


@dataclass
class TimeSeriesPoint:
    """A single point in a time-series."""

    timestamp: str
    value: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"timestamp": self.timestamp, "value": self.value, "metadata": self.metadata or {}}


@dataclass
class TimeSeriesData:
    """Time-series data for a metric."""

    metric: str
    unit: str
    points: list[TimeSeriesPoint]
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "unit": self.unit,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class TraceInfo:
    """Information about a distributed trace."""

    trace_id: str
    start_time: str
    duration_ms: float
    span_count: int
    status: str  # 'success', 'error', 'in_progress'
    root_operation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "span_count": self.span_count,
            "status": self.status,
            "root_operation": self.root_operation,
        }


@dataclass
class TraceDetail:
    """Detailed trace information with spans."""

    trace_id: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    spans: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "spans": self.spans,
        }


class DashboardAPI:
    """Backend API for dashboard metrics and monitoring.

    Provides REST-style methods for accessing telemetry data,
    alert configuration, and trace information.
    """

    def __init__(self):
        """Initialize dashboard API."""
        self.telemetry = get_telemetry_collector()
        self._lock = threading.RLock()
        self._start_time = datetime.now(timezone.utc)

        # In-memory trace storage (Phase 3 integration)
        self._traces: dict[str, TraceDetail] = {}
        self._trace_list: list[TraceInfo] = []

        logger.info("DashboardAPI initialized")

    # === Metrics Endpoints ===

    def get_latest_metrics(self) -> MetricsSnapshot:
        """Get current snapshot of all metrics.

        Returns:
            MetricsSnapshot with latest values
        """
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() * 1000

            # Collect adapter metrics
            adapter_metrics = self.telemetry.get_adapter_metrics()
            adapter_summary = {
                "total_providers": len({m.provider for m in adapter_metrics.values()}),
                "total_requests": sum(m.total_requests for m in adapter_metrics.values()),
                "total_successful": sum(m.successful_requests for m in adapter_metrics.values()),
                "total_failed": sum(m.failed_requests for m in adapter_metrics.values()),
                "average_latency_ms": self._calc_avg_latency(adapter_metrics),
                "total_tokens_used": sum(m.total_tokens_used for m in adapter_metrics.values()),
                "providers": {
                    k: {
                        "provider": v.provider,
                        "model": v.model,
                        "requests": v.total_requests,
                        "success_rate": v.success_rate,
                        "avg_latency_ms": v.avg_latency_ms,
                        "min_latency_ms": v.min_latency_ms if v.min_latency_ms != float("inf") else 0,
                        "max_latency_ms": v.max_latency_ms,
                        "last_request": v.last_request_time,
                    }
                    for k, v in adapter_metrics.items()
                },
            }

            # Collect memory metrics
            memory_metrics = self.telemetry.get_memory_metrics()
            memory_summary = {
                "backends": {
                    k: {
                        "backend": v.backend,
                        "writes": v.total_writes,
                        "reads": v.total_reads,
                        "searches": v.total_searches,
                        "avg_write_latency_ms": v.avg_write_latency(),
                        "avg_read_latency_ms": v.avg_read_latency(),
                        "avg_search_latency_ms": v.avg_search_latency(),
                        "dedup_hit_rate": v.dedup_hit_rate,
                        "sync_failures": v.sync_failures,
                    }
                    for k, v in memory_metrics.items()
                }
            }

            # Collect plan metrics
            plan_metrics = self.telemetry.get_plan_metrics()
            plan_summary = {
                "total_decisions": plan_metrics.total_decisions,
                "approved": plan_metrics.approved_decisions,
                "rejected": plan_metrics.rejected_decisions,
                "auto_approved": plan_metrics.auto_approved_decisions,
                "approval_rate": plan_metrics.approval_rate,
                "average_risk_score": plan_metrics.average_risk_score,
                "average_approval_time_ms": plan_metrics.average_approval_time_ms,
            }

            return MetricsSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_ms=uptime,
                adapter_summary=adapter_summary,
                memory_summary=memory_summary,
                plan_summary=plan_summary,
            )

    def get_timeseries_data(
        self, metric: str, timerange: str = "24h", provider: str | None = None
    ) -> TimeSeriesData | None:
        """Get time-series data for a metric.

        Args:
            metric: Metric name ('latency', 'success_rate', 'token_usage', 'memory_latency')
            timerange: Time range ('1h', '24h', '7d')
            provider: Optional provider filter for adapter metrics

        Returns:
            TimeSeriesData with historical points, or None if not available
        """
        with self._lock:
            if metric == "latency":
                return self._get_adapter_latency_timeseries(timerange, provider)
            elif metric == "success_rate":
                return self._get_success_rate_timeseries(timerange, provider)
            elif metric == "token_usage":
                return self._get_token_usage_timeseries(timerange, provider)
            elif metric == "memory_latency":
                return self._get_memory_latency_timeseries(timerange)
            else:
                logger.warning("Unknown metric requested: %s", metric)
                return None

    def _get_adapter_latency_timeseries(self, timerange: str, provider: str | None = None) -> TimeSeriesData:
        """Get adapter latency time-series."""
        metrics = self.telemetry.get_adapter_metrics(provider)

        points = []
        latencies = []

        for _key, metric in metrics.items():
            if metric.total_requests > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=metric.last_request_time or datetime.now(timezone.utc).isoformat(),
                        value=metric.avg_latency_ms,
                        metadata={"provider": metric.provider, "model": metric.model},
                    )
                )
                latencies.append(metric.avg_latency_ms)

        # Sort by timestamp
        points.sort(key=lambda p: p.timestamp)

        return TimeSeriesData(
            metric="latency",
            unit="ms",
            points=points,
            min_value=min(latencies) if latencies else 0.0,
            max_value=max(latencies) if latencies else 0.0,
            avg_value=sum(latencies) / len(latencies) if latencies else 0.0,
        )

    def _get_success_rate_timeseries(self, timerange: str, provider: str | None = None) -> TimeSeriesData:
        """Get success rate time-series."""
        metrics = self.telemetry.get_adapter_metrics(provider)

        points = []
        rates = []

        for _key, metric in metrics.items():
            if metric.total_requests > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=metric.last_request_time or datetime.now(timezone.utc).isoformat(),
                        value=metric.success_rate,
                        metadata={"provider": metric.provider, "model": metric.model},
                    )
                )
                rates.append(metric.success_rate)

        points.sort(key=lambda p: p.timestamp)

        return TimeSeriesData(
            metric="success_rate",
            unit="%",
            points=points,
            min_value=min(rates) if rates else 0.0,
            max_value=max(rates) if rates else 100.0,
            avg_value=sum(rates) / len(rates) if rates else 0.0,
        )

    def _get_token_usage_timeseries(self, timerange: str, provider: str | None = None) -> TimeSeriesData:
        """Get token usage time-series."""
        metrics = self.telemetry.get_adapter_metrics(provider)

        points = []
        usages = []

        for _key, metric in metrics.items():
            if metric.total_requests > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=metric.last_request_time or datetime.now(timezone.utc).isoformat(),
                        value=metric.total_tokens_used,
                        metadata={"provider": metric.provider, "model": metric.model},
                    )
                )
                usages.append(metric.total_tokens_used)

        points.sort(key=lambda p: p.timestamp)

        return TimeSeriesData(
            metric="token_usage",
            unit="tokens",
            points=points,
            min_value=min(usages) if usages else 0.0,
            max_value=max(usages) if usages else 0.0,
            avg_value=sum(usages) / len(usages) if usages else 0.0,
        )

    def _get_memory_latency_timeseries(self, timerange: str) -> TimeSeriesData:
        """Get memory latency time-series."""
        metrics = self.telemetry.get_memory_metrics()

        points = []
        all_latencies = []

        for _key, metric in metrics.items():
            read_latency = metric.avg_read_latency()
            write_latency = metric.avg_write_latency()
            search_latency = metric.avg_search_latency()

            if read_latency > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        value=read_latency,
                        metadata={"backend": metric.backend, "operation": "read"},
                    )
                )
                all_latencies.append(read_latency)

            if write_latency > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        value=write_latency,
                        metadata={"backend": metric.backend, "operation": "write"},
                    )
                )
                all_latencies.append(write_latency)

            if search_latency > 0:
                points.append(
                    TimeSeriesPoint(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        value=search_latency,
                        metadata={"backend": metric.backend, "operation": "search"},
                    )
                )
                all_latencies.append(search_latency)

        return TimeSeriesData(
            metric="memory_latency",
            unit="ms",
            points=points,
            min_value=min(all_latencies) if all_latencies else 0.0,
            max_value=max(all_latencies) if all_latencies else 0.0,
            avg_value=sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
        )

    # === Trace Endpoints ===

    def search_traces(self, trace_id: str | None = None, limit: int = 100) -> list[TraceInfo]:
        """Search for traces by ID or list recent traces.

        Args:
            trace_id: Optional trace ID to search for
            limit: Maximum number of traces to return

        Returns:
            List of TraceInfo objects
        """
        with self._lock:
            if trace_id:
                # Search for specific trace
                results = [t for t in self._trace_list if t.trace_id == trace_id]
            else:
                # Return recent traces
                results = sorted(self._trace_list, key=lambda t: t.start_time, reverse=True)[:limit]

            return results

    def get_trace_detail(self, trace_id: str) -> TraceDetail | None:
        """Get detailed information about a trace.

        Args:
            trace_id: Trace ID to retrieve

        Returns:
            TraceDetail object, or None if not found
        """
        with self._lock:
            return self._traces.get(trace_id)

    def add_trace(self, trace_detail: TraceDetail) -> bool:
        """Add a trace to the dashboard (called by logging integration).

        Args:
            trace_detail: Trace information to store

        Returns:
            True if successful
        """
        with self._lock:
            try:
                self._traces[trace_detail.trace_id] = trace_detail

                # Add to trace list
                trace_info = TraceInfo(
                    trace_id=trace_detail.trace_id,
                    start_time=trace_detail.start_time,
                    duration_ms=trace_detail.duration_ms,
                    span_count=len(trace_detail.spans),
                    status=trace_detail.status,
                    root_operation=trace_detail.spans[0].get("operation", "unknown")
                    if trace_detail.spans
                    else "unknown",
                )
                self._trace_list.append(trace_info)

                # Keep only last 1000 traces
                if len(self._trace_list) > 1000:
                    oldest = self._trace_list.pop(0)
                    self._traces.pop(oldest.trace_id, None)

                logger.debug("Trace added: %s", trace_detail.trace_id)
                return True
            except Exception as e:
                logger.error("Failed to add trace: %s", e)
                return False

    # === Helper Methods ===

    def _calc_avg_latency(self, metrics: dict[str, Any]) -> float:
        """Calculate average latency across all providers."""
        if not metrics:
            return 0.0

        latencies = [m.avg_latency_ms for m in metrics.values() if m.total_requests > 0]
        if not latencies:
            return 0.0

        return sum(latencies) / len(latencies)

    def clear_traces(self) -> None:
        """Clear all stored traces (useful for testing or memory management)."""
        with self._lock:
            self._traces.clear()
            self._trace_list.clear()
            logger.info("Traces cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get dashboard statistics.

        Returns:
            The result string.
        """
        with self._lock:
            return {
                "total_traces_stored": len(self._traces),
                "trace_list_size": len(self._trace_list),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global singleton instance
_dashboard_api: DashboardAPI | None = None
_dashboard_lock = threading.Lock()


def get_dashboard_api() -> DashboardAPI:
    """Get or create the global dashboard API instance.

    Returns:
        The DashboardAPI result.
    """
    global _dashboard_api
    if _dashboard_api is None:
        with _dashboard_lock:
            if _dashboard_api is None:
                _dashboard_api = DashboardAPI()
    return _dashboard_api


def reset_dashboard():
    """Reset dashboard API (mainly for testing)."""
    global _dashboard_api
    _dashboard_api = None
