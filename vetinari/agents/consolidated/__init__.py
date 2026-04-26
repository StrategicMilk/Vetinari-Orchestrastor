"""Consolidated Agents Package (v0.5.0).

======================================
Two agents live in this sub-package:

- WorkerAgent:    Unified execution engine (24 modes across 4 groups).
- InspectorAgent: Independent quality gate (4 modes).

The Worker internally delegates to ConsolidatedResearcherAgent,
ConsolidatedOracleAgent, BuilderAgent (parent package), and
OperationsAgent — these are internal implementation details,
not standalone agents.

Imports are lazy via __getattr__ to prevent circular import issues caused by
sub-modules (quality_agent, quality_patterns, etc.) importing from each other
through this package namespace during their own loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vetinari.agents.consolidated.quality_agent import InspectorAgent, get_inspector_agent
    from vetinari.agents.consolidated.worker_agent import WorkerAgent, get_worker_agent

__all__ = [
    "InspectorAgent",
    "WorkerAgent",
    "get_inspector_agent",
    "get_worker_agent",
]

_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy-load package symbols on first access to avoid circular imports."""
    if name in _cache:
        return _cache[name]
    if name in ("InspectorAgent", "get_inspector_agent"):
        from vetinari.agents.consolidated.quality_agent import (
            InspectorAgent,
            get_inspector_agent,
        )

        _cache["InspectorAgent"] = InspectorAgent
        _cache["get_inspector_agent"] = get_inspector_agent
        return _cache[name]
    if name in ("WorkerAgent", "get_worker_agent"):
        from vetinari.agents.consolidated.worker_agent import (
            WorkerAgent,
            get_worker_agent,
        )

        _cache["WorkerAgent"] = WorkerAgent
        _cache["get_worker_agent"] = get_worker_agent
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
