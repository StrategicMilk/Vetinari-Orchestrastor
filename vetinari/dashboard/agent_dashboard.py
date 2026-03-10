"""
Agent Performance Dashboard (C10)
==================================
Aggregates metrics from CostTracker, CircuitBreakerRegistry, and agent
execution data into a unified dashboard view.

Provides Flask-compatible route handlers for the web UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent type."""
    agent_type: str
    total_executions: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    circuit_breaker_state: str = "CLOSED"
    active_modes: List[str] = field(default_factory=list)
    token_budget_remaining: int = 0
    last_execution: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful / self.total_executions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "total_executions": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "circuit_breaker_state": self.circuit_breaker_state,
            "active_modes": self.active_modes,
            "token_budget_remaining": self.token_budget_remaining,
            "last_execution": self.last_execution,
        }


@dataclass
class SystemHealth:
    """Overall system health summary."""
    status: str = "healthy"  # healthy, degraded, unhealthy
    uptime_seconds: float = 0.0
    total_agents: int = 6
    healthy_agents: int = 6
    open_circuit_breakers: List[str] = field(default_factory=list)
    total_cost_24h: float = 0.0
    total_tokens_24h: int = 0
    active_plans: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_agents": self.total_agents,
            "healthy_agents": self.healthy_agents,
            "open_circuit_breakers": self.open_circuit_breakers,
            "total_cost_24h": round(self.total_cost_24h, 6),
            "total_tokens_24h": self.total_tokens_24h,
            "active_plans": self.active_plans,
        }


_AGENT_TYPES = [
    "PLANNER", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
    "BUILDER", "QUALITY", "OPERATIONS",
]

_AGENT_MODES = {
    "PLANNER": ["plan", "clarify", "summarise", "prune", "extract", "consolidate"],
    "CONSOLIDATED_RESEARCHER": [
        "code_discovery", "domain_research", "api_lookup", "lateral_thinking",
        "ui_design", "database", "devops", "git_workflow",
    ],
    "CONSOLIDATED_ORACLE": [
        "architecture", "risk_assessment", "ontological_analysis", "contrarian_review",
    ],
    "BUILDER": ["build", "image_generation"],
    "QUALITY": ["code_review", "security_audit", "test_generation", "simplification"],
    "OPERATIONS": [
        "documentation", "creative_writing", "cost_analysis", "experiment",
        "error_recovery", "synthesis", "improvement", "monitor", "devops_ops",
    ],
}


class AgentDashboard:
    """Aggregates agent metrics from multiple sources."""

    def __init__(self):
        self._start_time = time.monotonic()

    def get_agent_metrics(self, agent_type: str) -> AgentMetrics:
        """Get metrics for a specific agent type."""
        metrics = AgentMetrics(
            agent_type=agent_type,
            active_modes=_AGENT_MODES.get(agent_type, []),
        )

        # Pull circuit breaker state
        try:
            from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
            cb = get_circuit_breaker_registry().get(agent_type)
            metrics.circuit_breaker_state = cb.state.value
        except Exception:
            pass

        # Pull cost data
        try:
            from vetinari.analytics.cost import get_cost_tracker
            tracker = get_cost_tracker()
            agent_costs = tracker.query_by_agent(agent_type)
            if agent_costs:
                metrics.total_cost = agent_costs.get("total_cost", 0.0)
                metrics.total_tokens = agent_costs.get("total_tokens", 0)
                metrics.total_executions = agent_costs.get("count", 0)
        except Exception:
            pass

        return metrics

    def get_all_agent_metrics(self) -> List[AgentMetrics]:
        """Get metrics for all agent types."""
        return [self.get_agent_metrics(at) for at in _AGENT_TYPES]

    def get_system_health(self) -> SystemHealth:
        """Get overall system health."""
        health = SystemHealth(
            uptime_seconds=time.monotonic() - self._start_time,
        )

        # Check circuit breakers
        try:
            from vetinari.resilience.circuit_breaker import get_circuit_breaker_registry
            registry = get_circuit_breaker_registry()
            for at in _AGENT_TYPES:
                cb = registry.get(at)
                if cb.state.value == "open":
                    health.open_circuit_breakers.append(at)
        except Exception:
            pass

        health.healthy_agents = health.total_agents - len(health.open_circuit_breakers)
        if len(health.open_circuit_breakers) >= 3:
            health.status = "unhealthy"
        elif len(health.open_circuit_breakers) >= 1:
            health.status = "degraded"

        # Cost summary
        try:
            from vetinari.analytics.cost import get_cost_tracker
            summary = get_cost_tracker().get_summary()
            health.total_cost_24h = summary.get("total_cost", 0.0)
            health.total_tokens_24h = summary.get("total_tokens", 0)
        except Exception:
            pass

        return health

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Complete dashboard payload."""
        return {
            "health": self.get_system_health().to_dict(),
            "agents": [m.to_dict() for m in self.get_all_agent_metrics()],
            "agent_modes": _AGENT_MODES,
        }


# ── Singleton ─────────────────────────────────────────────────────────

_dashboard: Optional[AgentDashboard] = None


def get_agent_dashboard() -> AgentDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = AgentDashboard()
    return _dashboard
