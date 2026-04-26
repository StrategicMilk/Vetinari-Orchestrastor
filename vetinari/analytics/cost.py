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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, cast

from vetinari.utils.serialization import dataclass_to_dict

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
        """Calculate the total USD cost for a given number of input and output tokens.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.

        Returns:
            Total cost in USD combining input, output, and per-request fees.
        """
        return input_tokens / 1000 * self.input_per_1k + output_tokens / 1000 * self.output_per_1k + self.per_request


# Built-in defaults (approximate public pricing as of early 2026)
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI family
    "openai:gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "openai:gpt-4o-mini": ModelPricing(input_per_1k=0.0006, output_per_1k=0.0024),
    "openai:o3-mini": ModelPricing(input_per_1k=0.0011, output_per_1k=0.0044),
    # Claude family (current generation)
    "anthropic:claude-opus-4": ModelPricing(input_per_1k=0.015, output_per_1k=0.075),
    "anthropic:claude-sonnet-4": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    "anthropic:claude-haiku-4": ModelPricing(input_per_1k=0.0008, output_per_1k=0.004),
    # Gemini family
    "google:gemini-2.5-pro": ModelPricing(input_per_1k=0.00625, output_per_1k=0.025),
    "google:gemini-2.5-flash": ModelPricing(input_per_1k=0.0015, output_per_1k=0.006),
    # Local models — zero cost by default
    "local:*": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
}


# ---------------------------------------------------------------------------
# Cost entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEntry:
    """A single billable inference call."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    agent: str | None = None
    task_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    cost_usd: float | None = None  # populated automatically by record() when None
    latency_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CostEntry(provider={self.provider!r}, model={self.model!r}, "
            f"agent={self.agent!r}, task_id={self.task_id!r}, cost_usd={self.cost_usd!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this cost entry to a plain dictionary for JSON export.

        Returns:
            Dictionary containing all cost entry fields.
        """
        return cast(dict[str, Any], dataclass_to_dict(self))


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

    def __repr__(self) -> str:
        return (
            f"CostReport(total_cost_usd={self.total_cost_usd!r}, "
            f"total_tokens={self.total_tokens!r}, total_requests={self.total_requests!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this cost report to a plain dictionary for JSON export.

        Returns:
            Dictionary containing aggregated cost breakdowns by agent, provider, model, and task.
        """
        return cast(dict[str, Any], dataclass_to_dict(self))


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
        self._entries: deque[CostEntry] = deque(maxlen=1000)
        self._pricing: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def set_pricing(self, provider: str, model: str, pricing: ModelPricing) -> None:
        """Set pricing.

        Args:
            provider: The provider.
            model: The model.
            pricing: The pricing.
        """
        with self._lock:
            self._pricing[f"{provider}:{model}"] = pricing

    def get_pricing(self, provider: str, model: str) -> ModelPricing:
        """Get pricing.

        Args:
            provider: The provider.
            model: The model.

        Returns:
            The ModelPricing result.
        """
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

        Returns:
            The CostEntry result.
        """
        with self._lock:
            if entry.cost_usd is None:
                # Only recalculate when caller did not provide an explicit cost.
                # A caller-supplied cost_usd=0.0 is preserved as-is.
                pricing = self.get_pricing(entry.provider, entry.model)
                entry = CostEntry(
                    provider=entry.provider,
                    model=entry.model,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    agent=entry.agent,
                    task_id=entry.task_id,
                    timestamp=entry.timestamp,
                    cost_usd=pricing.compute(entry.input_tokens, entry.output_tokens),
                    latency_ms=entry.latency_ms,
                )
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

        Returns:
            CostReport with total spend, token counts, and breakdowns by
            agent, provider, model (``provider:model`` key), and task.
            All monetary values are in USD.
        """
        with self._lock:
            entries = list(self._entries)

        if agent:
            entries = [e for e in entries if e.agent == agent]
        if task_id:
            entries = [e for e in entries if e.task_id == task_id]
        if since is not None:
            entries = [e for e in entries if e.timestamp >= since]

        total_cost = sum(e.cost_usd or 0.0 for e in entries)
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
            cost = e.cost_usd or 0.0

            by_agent[key_a] = by_agent.get(key_a, 0.0) + cost
            by_provider[key_p] = by_provider.get(key_p, 0.0) + cost
            by_model[key_m] = by_model.get(key_m, 0.0) + cost
            by_task[key_t] = by_task.get(key_t, 0.0) + cost

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
        """Return the N most expensive agents.

        Returns:
            List of dicts sorted by descending cost, each with ``agent``
            (agent name) and ``cost_usd`` (total spend) keys.
        """
        report = self.get_report()
        ranked = sorted(report.by_agent.items(), key=lambda x: x[1], reverse=True)
        return [{"agent": k, "cost_usd": v} for k, v in ranked[:n]]

    def get_top_models(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the N most expensive provider/model combos.

        Returns:
            List of dicts sorted by descending cost, each with ``model``
            (``provider:model`` key) and ``cost_usd`` (total spend) keys.
        """
        report = self.get_report()
        ranked = sorted(report.by_model.items(), key=lambda x: x[1], reverse=True)
        return [{"model": k, "cost_usd": v} for k, v in ranked[:n]]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Summarise the current tracker state without computing a full report.

        Returns:
            Dictionary with ``total_entries`` (number of recorded cost entries)
            and ``configured_models`` (number of pricing rules loaded).
        """
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "configured_models": len(self._pricing),
            }

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated cost summary for the token-stats endpoint.

        Returns:
            Dictionary with ``total_cost_usd`` and ``by_model`` keys
            summarising all recorded cost entries.
        """
        with self._lock:
            total_cost = sum(e.cost_usd or 0.0 for e in self._entries)
            by_model: dict[str, dict[str, float]] = {}
            for entry in self._entries:
                key = f"{entry.provider}:{entry.model}" if entry.provider else entry.model
                if key not in by_model:
                    by_model[key] = {"cost_usd": 0.0, "tokens": 0}
                by_model[key]["cost_usd"] += entry.cost_usd or 0.0
                by_model[key]["tokens"] += entry.input_tokens + entry.output_tokens
            return {
                "total_cost_usd": round(total_cost, 6),
                "by_model": by_model,
            }

    def clear(self) -> None:
        """Clear for the current context."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def get_cost_tracker() -> CostTracker:
    """Return the singleton CostTracker instance, creating it if necessary.

    Returns:
        The shared CostTracker singleton used for all cost attribution.
    """
    return CostTracker()


def reset_cost_tracker() -> None:
    """Reset cost tracker."""
    with CostTracker._class_lock:
        CostTracker._instance = None
