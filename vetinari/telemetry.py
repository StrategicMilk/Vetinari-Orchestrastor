"""
Telemetry Module for Vetinari

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

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path

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
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    total_tokens_used: int = 0
    last_request_time: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_latency_ms(self) -> float:
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
    write_latency_ms: List[float] = field(default_factory=list)
    read_latency_ms: List[float] = field(default_factory=list)
    search_latency_ms: List[float] = field(default_factory=list)
    dedup_hits: int = 0
    dedup_misses: int = 0
    sync_failures: int = 0
    
    @property
    def dedup_hit_rate(self) -> float:
        total = self.dedup_hits + self.dedup_misses
        if total == 0:
            return 0.0
        return (self.dedup_hits / total) * 100
    
    def avg_write_latency(self) -> float:
        if not self.write_latency_ms:
            return 0.0
        return sum(self.write_latency_ms) / len(self.write_latency_ms)
    
    def avg_read_latency(self) -> float:
        if not self.read_latency_ms:
            return 0.0
        return sum(self.read_latency_ms) / len(self.read_latency_ms)
    
    def avg_search_latency(self) -> float:
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
    risk_scores: List[float] = field(default_factory=list)
    average_approval_time_ms: float = 0.0
    approval_times_ms: List[float] = field(default_factory=list)
    
    @property
    def approval_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return (self.approved_decisions / self.total_decisions) * 100
    
    def update_average_risk_score(self):
        if self.risk_scores:
            self.average_risk_score = sum(self.risk_scores) / len(self.risk_scores)
    
    def update_average_approval_time(self):
        if self.approval_times_ms:
            self.average_approval_time_ms = sum(self.approval_times_ms) / len(self.approval_times_ms)


class TelemetryCollector:
    """
    Singleton telemetry collector for system-wide metrics.
    
    Thread-safe collection and export of performance metrics.
    """
    
    def __init__(self):
        self.adapter_metrics: Dict[str, AdapterMetrics] = {}
        self.memory_metrics: Dict[str, MemoryMetrics] = {
            "oc": MemoryMetrics(backend="oc"),
            "mnemosyne": MemoryMetrics(backend="mnemosyne")
        }
        self.plan_metrics = PlanMetrics()
        self._lock = threading.RLock()
        self._start_time = datetime.now(timezone.utc)
        
        logger.info("TelemetryCollector initialized")
    
    # === Adapter Metrics ===
    
    def record_adapter_latency(self, provider: str, model: str, latency_ms: float, 
                              success: bool = True, tokens_used: int = 0):
        """Record adapter inference latency."""
        with self._lock:
            key = f"{provider}:{model}"
            if key not in self.adapter_metrics:
                self.adapter_metrics[key] = AdapterMetrics(provider=provider, model=model)
            
            metrics = self.adapter_metrics[key]
            metrics.total_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            metrics.total_tokens_used += tokens_used
            metrics.last_request_time = datetime.now(timezone.utc).isoformat()
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            logger.debug(f"Recorded adapter latency: {key} = {latency_ms}ms (success={success})")
    
    def get_adapter_metrics(self, provider: Optional[str] = None) -> Dict[str, AdapterMetrics]:
        """Get adapter metrics, optionally filtered by provider."""
        with self._lock:
            if provider:
                return {k: v for k, v in self.adapter_metrics.items() if v.provider == provider}
            return dict(self.adapter_metrics)
    
    # === Memory Metrics ===
    
    def record_memory_write(self, backend: str, latency_ms: float):
        """Record memory write operation."""
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_writes += 1
                metrics.write_latency_ms.append(latency_ms)
                logger.debug(f"Recorded memory write: {backend} = {latency_ms}ms")
    
    def record_memory_read(self, backend: str, latency_ms: float):
        """Record memory read operation."""
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_reads += 1
                metrics.read_latency_ms.append(latency_ms)
                logger.debug(f"Recorded memory read: {backend} = {latency_ms}ms")
    
    def record_memory_search(self, backend: str, latency_ms: float):
        """Record memory search operation."""
        with self._lock:
            if backend in self.memory_metrics:
                metrics = self.memory_metrics[backend]
                metrics.total_searches += 1
                metrics.search_latency_ms.append(latency_ms)
                logger.debug(f"Recorded memory search: {backend} = {latency_ms}ms")
    
    def record_dedup_hit(self, backend: str):
        """Record successful content deduplication."""
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].dedup_hits += 1
    
    def record_dedup_miss(self, backend: str):
        """Record failed content deduplication."""
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].dedup_misses += 1
    
    def record_sync_failure(self, backend: str):
        """Record memory backend synchronization failure."""
        with self._lock:
            if backend in self.memory_metrics:
                self.memory_metrics[backend].sync_failures += 1
                logger.warning(f"Memory sync failure recorded for {backend}")
    
    def get_memory_metrics(self, backend: Optional[str] = None) -> Dict[str, MemoryMetrics]:
        """Get memory metrics, optionally filtered by backend."""
        with self._lock:
            if backend and backend in self.memory_metrics:
                return {backend: self.memory_metrics[backend]}
            return dict(self.memory_metrics)
    
    # === Plan Mode Metrics ===
    
    def record_plan_decision(self, decision: str, risk_score: float = 0.0, 
                            approval_time_ms: Optional[float] = None, auto_approved: bool = False):
        """
        Record a plan mode decision (approve/reject).
        
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
            
            logger.debug(f"Recorded plan decision: {decision} (risk={risk_score})")
    
    def get_plan_metrics(self) -> PlanMetrics:
        """Get plan mode metrics."""
        with self._lock:
            return PlanMetrics(**asdict(self.plan_metrics))
    
    # === Export ===
    
    def export_json(self, path: str) -> bool:
        """Export all metrics to JSON file."""
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
                            "last_request_time": v.last_request_time
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
                            "sync_failures": v.sync_failures
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
                        "average_approval_time_ms": self.plan_metrics.average_approval_time_ms
                    }
                }
                
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Telemetry exported to {path}")
                return True
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")
            return False
    
    def export_prometheus(self, path: str) -> bool:
        """Export metrics in Prometheus text format."""
        try:
            with self._lock:
                lines = []
                lines.append("# HELP vetinari_adapter_requests_total Total adapter requests")
                lines.append("# TYPE vetinari_adapter_requests_total counter")
                
                for key, metrics in self.adapter_metrics.items():
                    labels = f'provider="{metrics.provider}",model="{metrics.model}"'
                    lines.append(f'vetinari_adapter_requests_total{{{labels}}} {metrics.total_requests}')
                
                lines.append("# HELP vetinari_adapter_latency_ms Adapter latency in milliseconds")
                lines.append("# TYPE vetinari_adapter_latency_ms gauge")
                
                for key, metrics in self.adapter_metrics.items():
                    labels = f'provider="{metrics.provider}",model="{metrics.model}"'
                    lines.append(f'vetinari_adapter_latency_ms{{{labels}}} {metrics.avg_latency_ms}')
                
                lines.append("# HELP vetinari_memory_operations_total Total memory operations")
                lines.append("# TYPE vetinari_memory_operations_total counter")
                
                for key, metrics in self.memory_metrics.items():
                    total_ops = metrics.total_writes + metrics.total_reads + metrics.total_searches
                    lines.append(f'vetinari_memory_operations_total{{backend="{metrics.backend}"}} {total_ops}')
                
                lines.append("# HELP vetinari_plan_decisions_total Total plan decisions")
                lines.append("# TYPE vetinari_plan_decisions_total counter")
                lines.append(f'vetinari_plan_decisions_total{{decision="approve"}} {self.plan_metrics.approved_decisions}')
                lines.append(f'vetinari_plan_decisions_total{{decision="reject"}} {self.plan_metrics.rejected_decisions}')
                
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    f.write('\n'.join(lines))
                
                logger.info(f"Prometheus metrics exported to {path}")
                return True
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return False
    
    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.adapter_metrics.clear()
            self.memory_metrics = {
                "oc": MemoryMetrics(backend="oc"),
                "mnemosyne": MemoryMetrics(backend="mnemosyne")
            }
            self.plan_metrics = PlanMetrics()
            self._start_time = datetime.now(timezone.utc)
            logger.info("Telemetry metrics reset")


# Global singleton instance
_telemetry: Optional[TelemetryCollector] = None
_telemetry_lock = threading.Lock()


def get_telemetry_collector() -> TelemetryCollector:
    """Get or create the global telemetry collector instance."""
    global _telemetry
    if _telemetry is None:
        with _telemetry_lock:
            if _telemetry is None:
                _telemetry = TelemetryCollector()
    return _telemetry


def reset_telemetry():
    """Reset telemetry (mainly for testing)."""
    global _telemetry
    if _telemetry:
        _telemetry.reset()
