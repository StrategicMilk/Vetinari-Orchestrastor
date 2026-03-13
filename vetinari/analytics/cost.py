"""Cost Attribution — vetinari.analytics.cost  (Phase 5).

Tracks token and compute costs per agent, task, provider and model.

Pricing is configurable per provider/model pair ($/1k input-tokens,
$/1k output-tokens, $/request).  A built-in default table covers common
OpenAI and local-model scenarios so the module is useful out-of-the-box.

Usage
-----
    from vetinari.analytics.cost import get_cost_tracker, CostEntry, ModelPricing

    tracker = get_cost_tracker()

    # Override pricing for a model
    tracker.set_pricing("openai", "gpt-4",
                        ModelPricing(input_per_1k=0.03, output_per_1k=0.06))

    # Record a call
    tracker.record(CostEntry(
        agent="builder",
        task_id="task-001",
        provider="openai",
        model="gpt-4",
        input_tokens=500,
        output_tokens=200,
    ))

    report = tracker.get_report()
    logger.debug(report.total_cost_usd)
    logger.debug(report.by_agent)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """USD cost per 1 000 tokens and per request."""

    input_per_1k: float = 0.0  # cost per 1k input tokens
    output_per_1k: float = 0.0  # cost per 1k output tokens
    per_request: float = 0.0  # flat fee per API call

    def compute(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1000 * self.input_per_1k + output_tokens / 1000 * self.output_per_1k + self.per_request


# Built-in defaults (approximate public pricing as of early 2026)
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "openai:gpt-4": ModelPricing(input_per_1k=0.030, output_per_1k=0.060),
    "openai:gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "openai:gpt-3.5-turbo": ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
    "anthropic:claude-3-opus": ModelPricing(input_per_1k=0.015, output_per_1k=0.075),
    "anthropic:claude-3-sonnet": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    "anthropic:claude-3-haiku": ModelPricing(input_per_1k=0.00025, output_per_1k=0.00125),
    # Local / LM Studio models — zero cost by default
    "lmstudio:*": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
}


# ---------------------------------------------------------------------------
# Cost entry
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    """A single billable inference call."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    agent: str | None = None
    task_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    cost_usd: float = 0.0  # populated automatically by record()
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "agent": self.agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class CostReport:
    """Aggregated cost breakdown."""

    total_cost_usd: float
    total_tokens: int
    total_requests: int
    by_agent: dict[str, float]  # agent  -> total USD
    by_provider: dict[str, float]  # provider -> total USD
    by_model: dict[str, float]  # "provider:model" -> total USD
    by_task: dict[str, float]  # task_id -> total USD
    entries: int  # raw entry count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "by_agent": self.by_agent,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "by_task": self.by_task,
            "entries": self.entries,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe cost attribution tracker.  Singleton — use ``get_cost_tracker()``."""

    _instance: CostTracker | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> CostTracker:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[CostEntry] = []
        self._pricing: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def set_pricing(self, provider: str, model: str, pricing: ModelPricing) -> None:
        with self._lock:
            self._pricing[f"{provider}:{model}"] = pricing

    def get_pricing(self, provider: str, model: str) -> ModelPricing:
        with self._lock:
            key = f"{provider}:{model}"
            if key in self._pricing:
                return self._pricing[key]
            # Try wildcard for provider
            wildcard = f"{provider}:*"
            if wildcard in self._pricing:
                return self._pricing[wildcard]
            return ModelPricing()  # free / unknown

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, entry: CostEntry) -> CostEntry:
        """Record a cost entry.  ``entry.cost_usd`` is calculated automatically.

        using the configured pricing if it is not already set (> 0).
        """
        with self._lock:
            if entry.cost_usd == 0.0:
                pricing = self.get_pricing(entry.provider, entry.model)
                entry.cost_usd = pricing.compute(entry.input_tokens, entry.output_tokens)
            self._entries.append(entry)
            logger.debug(
                "Cost recorded: %s/%s  in=%d out=%d  $%.6f  agent=%s",
                entry.provider,
                entry.model,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost_usd,
                entry.agent,
            )
            return entry

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(
        self,
        agent: str | None = None,
        task_id: str | None = None,
        since: float | None = None,
    ) -> CostReport:
        """Build an aggregated cost report, optionally filtered.

        Args:
            agent:   Only include entries from this agent.
            task_id: Only include entries for this task.
            since:   Only include entries with timestamp >= since (unix epoch).
        """
        with self._lock:
            entries = list(self._entries)

        if agent:
            entries = [e for e in entries if e.agent == agent]
        if task_id:
            entries = [e for e in entries if e.task_id == task_id]
        if since is not None:
            entries = [e for e in entries if e.timestamp >= since]

        total_cost = sum(e.cost_usd for e in entries)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in entries)

        by_agent: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_task: dict[str, float] = {}

        for e in entries:
            key_a = e.agent or "unknown"
            key_p = e.provider or "unknown"
            key_m = f"{e.provider}:{e.model}"
            key_t = e.task_id or "unknown"

            by_agent[key_a] = by_agent.get(key_a, 0.0) + e.cost_usd
            by_provider[key_p] = by_provider.get(key_p, 0.0) + e.cost_usd
            by_model[key_m] = by_model.get(key_m, 0.0) + e.cost_usd
            by_task[key_t] = by_task.get(key_t, 0.0) + e.cost_usd

        return CostReport(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            total_requests=len(entries),
            by_agent=by_agent,
            by_provider=by_provider,
            by_model=by_model,
            by_task=by_task,
            entries=len(entries),
        )

    def get_top_agents(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the N most expensive agents."""
        report = self.get_report()
        ranked = sorted(report.by_agent.items(), key=lambda x: x[1], reverse=True)
        return [{"agent": k, "cost_usd": v} for k, v in ranked[:n]]

    def get_top_models(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the N most expensive provider/model combos."""
        report = self.get_report()
        ranked = sorted(report.by_model.items(), key=lambda x: x[1], reverse=True)
        return [{"model": k, "cost_usd": v} for k, v in ranked[:n]]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "configured_models": len(self._pricing),
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def get_cost_tracker() -> CostTracker:
    return CostTracker()


def reset_cost_tracker() -> None:
    with CostTracker._class_lock:
        CostTracker._instance = None
