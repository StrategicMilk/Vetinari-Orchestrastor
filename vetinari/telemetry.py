"""Telemetry Module for Vetinari.

Provides comprehensive metrics collection for:
- Adapter performance (latency, token usage, success rates)
- Memory operations (read/write latency, dedup hit rates)
- Plan mode metrics (approval ratios, risk scores)

Metrics are collected in-process and can be exported to JSON or Prometheus format.

Usage:
    from vetinari.telemetry import get_telemetry_collector

    telemetry = get_telemetry_collector()
    telemetry.record_adapter_latency("openai", "gpt-4", 150.5)
    telemetry.record_memory_operation("remember", "oc", 5.2)
    telemetry.record_plan_decision("approve", risk_score=0.35)
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetrics:
    """Metrics for a single adapter/model combination."""

    provider: str
    model: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    total_tokens_used: int = 0
    last_request_time: str | None = None

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"AdapterMetrics(provider={self.provider!r}, model={self.model!r}, total_requests={self.total_requests!r})"
        )

    @property
    def success_rate(self) -> float:
        """Percentage of successful requests out of total requests."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency in milliseconds across successful requests."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


@dataclass
class MemoryMetrics:
    """Metrics for memory backend operations."""

    backend: str  # 'oc' or 'mnemosyne'
    total_writes: int = 0
    total_reads: int = 0
    total_searches: int = 0
    write_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    read_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    search_latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    dedup_hits: int = 0
    dedup_misses: int = 0
    sync_failures: int = 0

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"MemoryMetrics(backend={self.backend!r},"
            f" total_writes={self.total_writes!r},"
            f" total_reads={self.total_reads!r})"
        )

    @property
    def dedup_hit_rate(self) -> float:
        """Percentage of deduplication cache hits out of total lookups."""
        total = self.dedup_hits + self.dedup_misses
        if total == 0:
            return 0.0
        return (self.dedup_hits / total) * 100

    def avg_write_latency(self) -> float:
        """Mean write latency in milliseconds across all recorded writes.

        Returns:
            Average latency, or 0.0 if no writes have been recorded.
        """
        if not self.write_latency_ms:
            return 0.0
        return sum(self.write_latency_ms) / len(self.write_latency_ms)

    def avg_read_latency(self) -> float:
        """Mean read latency in milliseconds across all recorded reads.

        Returns:
            Average latency, or 0.0 if no reads have been recorded.
        """
        if not self.read_latency_ms:
            return 0.0
        return sum(self.read_latency_ms) / len(self.read_latency_ms)

    def avg_search_latency(self) -> float:
        """Mean search latency in milliseconds across all recorded searches.

        Returns:
            Average latency, or 0.0 if no searches have been recorded.
        """
        if not self.search_latency_ms:
            return 0.0
        return sum(self.search_latency_ms) / len(self.search_latency_ms)


@dataclass
class PlanMetrics:
    """Metrics for plan mode decisions."""

    total_decisions: int = 0
    approved_decisions: int = 0
    rejected_decisions: int = 0
    auto_approved_decisions: int = 0
    average_risk_score: float = 0.0
    risk_scores: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    average_approval_time_ms: float = 0.0
    approval_times_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10000))

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"PlanMetrics(total_decisions={self.total_decisions!r},"
            f" approved_decisions={self.approved_decisions!r},"
            f" average_risk_score={self.average_risk_score!r})"
        )

    @property
    def approval_rate(self) -> float:
        """Percentage of approved decisions out of total decisions."""
        if self.total_decisions == 0:
            return 0.0
        return (self.approved_decisions / self.total_decisions) * 100

    def update_average_risk_score(self) -> None:
        """Recalculate the running average risk score from all recorded scores."""
        if self.risk_scores:
            self.average_risk_score = sum(self.risk_scores) / len(self.risk_scores)

    def update_average_approval_time(self) -> None:
        """Recalculate the running average approval time from all recorded durations."""
        if self.approval_times_ms:
            self.average_approval_time_ms = sum(self.approval_times_ms) / len(self.approval_times_ms)


class TelemetryCollector:
    """Singleton telemetry collector for system-wide metrics.

    Thread-safe collection and export of performance metrics.
    """

    def __init__(self):
        self.adapter_metrics: dict[str, AdapterMetrics] = {}
        self.memory_metrics: dict[str, MemoryMetrics] = {
            "oc": MemoryMetrics(backend="oc"),
            "mnemosyne": MemoryMetrics(backend="mnemosyne"),
        }
        self.plan_metrics = PlanMetrics()
        self._lock = threading.RLock()
        self._start_time = datetime.now(timezone.utc)

        logger.info("TelemetryCollector initialized")

        # Restore state from the most recent snapshot if available
        try:
            self.restore_from_snapshot()
        except Exception as exc:
            logger.warning("TelemetryCollector: snapshot restore failed in __init__ — continuing with empty state: %s", exc)

    # === Adapter Metrics ===

    def record_adapter_latency(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        success: bool = True,
        tokens_used: int = 0,
    ) -> None:
        """Record a single inference call's latency and outcome for a provider/model pair.

        Args:
            provider: Adapter provider name (e.g. ``"llama_cpp"``, ``"litellm"``).
            model: Model identifier within the provider (e.g. ``"mistral-7b"``).
            latency_ms: Round-trip inference latency in milliseconds.
            success: Whether the inference call completed without error.
            tokens_used: Total tokens consumed (prompt + completion).
        """
        with self._lock:
            key = f"{provider}:{model}"
            if key not in self.adapter_metrics:
                self.adapter_metrics[key] = AdapterMetrics(provider=provider, model=model)

            metrics = self.adapter_metrics[key]
            metrics.total_requests += 1
            # Only count latency for successful calls — failed calls may include
            # partial/timeout latency that would skew avg_latency_ms unfairly.
            if success:
                metrics.successful_requests += 1
                metrics.total_latency_ms += latency_ms
                metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
                metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            else:
                metrics.failed_requests += 1
            metrics.total_tokens_used += tokens_used
            metrics.last_request_time = datetime.now(timezone.utc).isoformat()

            logger.debug("Recorded adapter latency: %s = %sms (success=%s)", key, latency_ms, success)

    def get_adapter_metrics(self, provider: str | None = None) -> dict[str, AdapterMetrics]:
        """Retrieve collected adapter metrics, optionally filtered by provider.

        Args:
            provider: If given, return only metrics for this provider name.

        Returns:
            Dictionary mapping ``"provider:model"`` keys to their AdapterMetrics.
        """
        with self._lock:
            if provider:
                return {k: v for k, v in self.adapter_metrics.items() if v.provider == provider}
            return dict(self.adapter_metrics)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all telemetry data for the /api/v1/token-stats endpoint.

        Aggregates token usage and cost across all adapters and models.

        Returns:
            Dictionary with total_tokens_used, total_cost_usd, by_model,
            by_provider, and session_requests.
        """
        with self._lock:
            total_tokens = 0
            total_requests = 0
            by_model: dict[str, dict[str, int | float]] = {}
            by_provider: dict[str, dict[str, int | float]] = {}

            for _key, m in self.adapter_metrics.items():
                total_tokens += m.total_tokens_used
                total_requests += m.total_requests

                model_name = m.model
                prov = m.provider
                by_model.setdefault(model_name, {"tokens": 0, "requests": 0})
                by_model[model_name]["tokens"] += m.total_tokens_used
                by_model[model_name]["requests"] += m.total_requests

                by_provider.setdefault(prov, {"tokens": 0, "requests": 0})
                by_provider[prov]["tokens"] += m.total_tokens_used
                by_provider[prov]["requests"] += m.total_requests

            return {
                "total_tokens_used": total_tokens,
                "total_cost_usd": 0.0,  # Cost tracking done by CostTracker, not here
                "by_model": by_model,
                "by_provider": by_provider,
                "session_requests": total_requests,
            }

    # === Memory Metrics ===

    def record_memory_write(self, backend: str, latency_ms: float) -> None:
        """Record memory write operation.

        Args:
            backend: The backend.
            latency_ms: The latency ms.
        """
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_writes += 1
                metrics.write_latency_ms.append(latency_ms)
                logger.debug("Recorded memory write: %s = %sms", backend, latency_ms)

    def record_memory_read(self, backend: str, latency_ms: float) -> None:
        """Record memory read operation.

        Args:
            backend: The backend.
            latency_ms: The latency ms.
        """
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_reads += 1
                metrics.read_latency_ms.append(latency_ms)
                logger.debug("Recorded memory read: %s = %sms", backend, latency_ms)

    def record_memory_search(self, backend: str, latency_ms: float) -> None:
        """Record memory search operation.

        Args:
            backend: The backend.
            latency_ms: The latency ms.
        """
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_searches += 1
                metrics.search_latency_ms.append(latency_ms)
                logger.debug("Recorded memory search: %s = %sms", backend, latency_ms)

    def record_dedup_hit(self, backend: str) -> None:
        """Record a successful content deduplication match.

        Args:
            backend: Memory backend identifier ('oc' or 'mnemosyne').
        """
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].dedup_hits += 1

    def record_dedup_miss(self, backend: str) -> None:
        """Record a content deduplication miss (no duplicate found).

        Args:
            backend: Memory backend identifier ('oc' or 'mnemosyne').
        """
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].dedup_misses += 1

    def record_sync_failure(self, backend: str) -> None:
        """Record a memory backend synchronization failure.

        Args:
            backend: Memory backend identifier ('oc' or 'mnemosyne').
        """
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].sync_failures += 1
                logger.warning("Memory sync failure recorded for %s", backend)

    def get_memory_metrics(self, backend: str | None = None) -> dict[str, MemoryMetrics]:
        """Return memory metrics, optionally filtered to a single backend.

        Args:
            backend: If given, return only the metrics for this backend name
                ('oc' or 'mnemosyne').

        Returns:
            Dictionary mapping backend name to its MemoryMetrics. When a
            specific backend is requested but not found, returns an empty dict.
        """
        with self._lock:
            if backend and backend in self.memory_metrics:
                return {backend: self.memory_metrics[backend]}
            return dict(self.memory_metrics)

    # === Plan Mode Metrics ===

    def record_plan_decision(
        self,
        decision: str,
        risk_score: float = 0.0,
        approval_time_ms: float | None = None,
        auto_approved: bool = False,
    ) -> None:
        """Record a plan mode decision (approve/reject).

        Args:
            decision: 'approve' or 'reject'
            risk_score: Risk score (0.0 to 1.0)
            approval_time_ms: Time taken to make the decision
            auto_approved: Whether this was auto-approved
        """
        with self._lock:
            self.plan_metrics.total_decisions += 1
            self.plan_metrics.risk_scores.append(risk_score)

            if decision.lower() == "approve":
                self.plan_metrics.approved_decisions += 1
                if auto_approved:
                    self.plan_metrics.auto_approved_decisions += 1
            elif decision.lower() == "reject":
                self.plan_metrics.rejected_decisions += 1

            if approval_time_ms is not None:
                self.plan_metrics.approval_times_ms.append(approval_time_ms)

            self.plan_metrics.update_average_risk_score()
            self.plan_metrics.update_average_approval_time()

            logger.debug("Recorded plan decision: %s (risk=%s)", decision, risk_score)

    def get_plan_metrics(self) -> PlanMetrics:
        """Return a snapshot copy of the current plan mode metrics.

        Returns:
            A new PlanMetrics instance with the current decision counts,
            approval rate, and risk score averages. Safe to inspect without
            holding the internal lock.
        """
        with self._lock:
            return PlanMetrics(**asdict(self.plan_metrics))

    # === Export ===

    def export_json(self, path: str) -> bool:
        """Export all metrics to JSON file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self._lock:
                uptime_ms = (datetime.now(timezone.utc) - self._start_time).total_seconds() * 1000

                export_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_ms": uptime_ms,
                    "adapters": {
                        k: {
                            "provider": v.provider,
                            "model": v.model,
                            "total_requests": v.total_requests,
                            "successful_requests": v.successful_requests,
                            "failed_requests": v.failed_requests,
                            "success_rate": v.success_rate,
                            "avg_latency_ms": v.avg_latency_ms,
                            "min_latency_ms": v.min_latency_ms,
                            "max_latency_ms": v.max_latency_ms,
                            "total_tokens_used": v.total_tokens_used,
                            "last_request_time": v.last_request_time,
                        }
                        for k, v in self.adapter_metrics.items()
                    },
                    "memory": {
                        k: {
                            "backend": v.backend,
                            "total_writes": v.total_writes,
                            "total_reads": v.total_reads,
                            "total_searches": v.total_searches,
                            "avg_write_latency_ms": v.avg_write_latency(),
                            "avg_read_latency_ms": v.avg_read_latency(),
                            "avg_search_latency_ms": v.avg_search_latency(),
                            "dedup_hit_rate": v.dedup_hit_rate,
                            "sync_failures": v.sync_failures,
                        }
                        for k, v in self.memory_metrics.items()
                    },
                    "plan_mode": {
                        "total_decisions": self.plan_metrics.total_decisions,
                        "approved_decisions": self.plan_metrics.approved_decisions,
                        "rejected_decisions": self.plan_metrics.rejected_decisions,
                        "auto_approved_decisions": self.plan_metrics.auto_approved_decisions,
                        "approval_rate": self.plan_metrics.approval_rate,
                        "average_risk_score": self.plan_metrics.average_risk_score,
                        "average_approval_time_ms": self.plan_metrics.average_approval_time_ms,
                    },
                }

                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with Path(path).open("w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)

                logger.info("Telemetry exported to %s", path)
                return True
        except Exception as e:
            logger.error("Failed to export telemetry: %s", e)
            return False

    def export_prometheus(self, path: str) -> bool:
        """Export metrics in Prometheus text format.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self._lock:
                lines = []
                lines.extend((
                    "# HELP vetinari_adapter_requests_total Total adapter requests",
                    "# TYPE vetinari_adapter_requests_total counter",
                ))

                for _key, metrics in self.adapter_metrics.items():
                    labels = f'provider="{metrics.provider}",model="{metrics.model}"'
                    lines.append(f"vetinari_adapter_requests_total{{{labels}}} {metrics.total_requests}")

                lines.extend((
                    "# HELP vetinari_adapter_latency_ms Adapter latency in milliseconds",
                    "# TYPE vetinari_adapter_latency_ms gauge",
                ))

                for _key, metrics in self.adapter_metrics.items():
                    labels = f'provider="{metrics.provider}",model="{metrics.model}"'
                    lines.append(f"vetinari_adapter_latency_ms{{{labels}}} {metrics.avg_latency_ms}")

                lines.extend((
                    "# HELP vetinari_memory_operations_total Total memory operations",
                    "# TYPE vetinari_memory_operations_total counter",
                ))

                for _key, metrics in self.memory_metrics.items():
                    total_ops = metrics.total_writes + metrics.total_reads + metrics.total_searches
                    lines.append(f'vetinari_memory_operations_total{{backend="{metrics.backend}"}} {total_ops}')

                lines.extend((
                    "# HELP vetinari_plan_decisions_total Total plan decisions",
                    "# TYPE vetinari_plan_decisions_total counter",
                    f'vetinari_plan_decisions_total{{decision="approve"}} {self.plan_metrics.approved_decisions}',
                    f'vetinari_plan_decisions_total{{decision="reject"}} {self.plan_metrics.rejected_decisions}',
                ))

                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("\n".join(lines), encoding="utf-8")

                logger.info("Prometheus metrics exported to %s", path)
                return True
        except Exception as e:
            logger.error("Failed to export Prometheus metrics: %s", e)
            return False

    def restore_from_snapshot(self) -> None:
        """Seed in-memory counters from the most recent SQLite telemetry snapshot.

        Reads the newest row from the ``telemetry_snapshots`` table, parses
        the JSON ``data`` column, and adds the stored adapter request/latency
        counts as a baseline so that ``get_summary()`` continues from the last
        known totals after a process restart.

        Must be called after ``__init__``, not inside it, to keep construction
        lightweight and side-effect-free.

        Gracefully degrades on any error (missing table, empty DB, malformed
        JSON) by logging at INFO level and leaving counters at zero.
        """
        # Read the N most-recent snapshots so we can fall back to an older one
        # when the newest row is malformed or contains no adapter data.
        _SNAPSHOT_FALLBACK_LIMIT = 5
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            rows = conn.execute(
                "SELECT data FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT ?",
                (_SNAPSHOT_FALLBACK_LIMIT,),
            ).fetchall()
        except Exception as exc:
            logger.info("TelemetryCollector: no snapshot available for restore (DB not ready): %s", exc)
            return

        if not rows:
            logger.info("TelemetryCollector: no prior snapshot found — starting from zero")
            return

        for i, row in enumerate(rows):
            try:
                snapshot: dict[str, Any] = json.loads(row[0])
            except Exception as exc:
                logger.warning(
                    "TelemetryCollector: snapshot row %d JSON parse failed — trying older row: %s",
                    i,
                    exc,
                )
                continue

            # Skip snapshots with no useful adapter data; try the next older one.
            if not snapshot.get("adapter_details"):
                logger.info(
                    "TelemetryCollector: snapshot row %d has no adapter data — trying older row",
                    i,
                )
                continue

            try:
                self._apply_snapshot(snapshot)
                return
            except Exception as exc:
                logger.warning(
                    "TelemetryCollector: snapshot row %d apply failed — trying older row: %s",
                    i,
                    exc,
                )

        logger.info("TelemetryCollector: no usable snapshot found in last %d rows — starting from zero", _SNAPSHOT_FALLBACK_LIMIT)

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Apply parsed snapshot data to in-memory counters.

        Called only by ``restore_from_snapshot()`` after successful parse.
        Protected by ``self._lock`` for thread safety.

        Args:
            snapshot: Parsed JSON dict from a ``telemetry_snapshots`` row.
        """
        adapter_details: dict[str, Any] = snapshot.get("adapter_details", {})

        with self._lock:
            for key, detail in adapter_details.items():
                if not isinstance(detail, dict):
                    continue
                # Parse "provider:model" key — skip malformed entries
                parts = key.split(":", 1)
                if len(parts) != 2:
                    continue
                provider, model = parts
                total_req = int(detail.get("total_requests", 0))
                failed_req = int(detail.get("failed_requests", 0))
                successful_req = max(0, total_req - failed_req)
                avg_latency = float(detail.get("avg_latency_ms", 0.0))
                total_latency = avg_latency * successful_req
                min_lat = float(detail.get("min_latency_ms", float("inf")))
                max_lat = float(detail.get("max_latency_ms", 0.0))
                # Restore tokens directly from per-adapter detail, which avoids
                # double-counting when two providers share a model name.  The
                # legacy by_model path fabricated a 60/40 split based on
                # iteration order when model names overlapped across providers.
                tokens = int(detail.get("total_tokens_used", 0))

                m = AdapterMetrics(provider=provider, model=model)
                m.total_requests = total_req
                m.successful_requests = successful_req
                m.failed_requests = failed_req
                m.total_latency_ms = total_latency
                m.min_latency_ms = min_lat if min_lat != 0.0 else float("inf")
                m.max_latency_ms = max_lat
                m.total_tokens_used = tokens
                self.adapter_metrics[key] = m

            restored = len(adapter_details)
            if restored:
                logger.info(
                    "TelemetryCollector: restored baseline from snapshot (%d adapter(s))",
                    restored,
                )
            else:
                logger.info("TelemetryCollector: snapshot contained no adapter data — starting from zero")

    def reset(self) -> None:
        """Reset all collected metrics to their initial state."""
        with self._lock:
            self.adapter_metrics.clear()
            self.memory_metrics = {"oc": MemoryMetrics(backend="oc"), "mnemosyne": MemoryMetrics(backend="mnemosyne")}
            self.plan_metrics = PlanMetrics()
            self._start_time = datetime.now(timezone.utc)
            logger.info("Telemetry metrics reset")


# Global singleton instance
_telemetry: TelemetryCollector | None = None
_telemetry_lock = threading.Lock()


def get_telemetry_collector() -> TelemetryCollector:
    """Get or create the global telemetry collector instance.

    Returns:
        The singleton TelemetryCollector shared across all subsystems.
    """
    global _telemetry
    if _telemetry is None:
        with _telemetry_lock:
            if _telemetry is None:
                _telemetry = TelemetryCollector()
    return _telemetry


def reset_telemetry() -> None:
    """Reset the global telemetry collector metrics to their initial state."""
    global _telemetry
    if _telemetry:
        _telemetry.reset()
