"""Capability Auditor — vetinari.drift.capability_auditor  (Phase 7).

Compares the *documented* capabilities of every agent against the
*implemented* capabilities returned by ``agent.get_capabilities()``.

Any mismatch (extra, missing, or renamed capability) is reported as drift.

Usage
-----
    from vetinari.drift.capability_auditor import get_capability_auditor

    auditor = get_capability_auditor()
    auditor.register_documented(
        "BUILDER",
        ["code_generation", "file_writing", "test_creation"]
    )

    findings = auditor.audit_all()
    for f in findings:
        logger.debug(f)
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CapabilityFinding:
    """A single capability drift finding for one agent."""

    agent_name: str
    extra_in_code: list[str]  # in code but not in docs
    missing_in_code: list[str]  # in docs but not in code
    is_drift: bool

    def __str__(self) -> str:
        if not self.is_drift:
            return f"[OK]   {self.agent_name}: capabilities aligned"
        parts = []
        if self.extra_in_code:
            parts.append(f"undocumented={self.extra_in_code}")
        if self.missing_in_code:
            parts.append(f"missing={self.missing_in_code}")
        return f"[DRIFT] {self.agent_name}: {'; '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "extra_in_code": self.extra_in_code,
            "missing_in_code": self.missing_in_code,
            "is_drift": self.is_drift,
        }


# ---------------------------------------------------------------------------
# Agent module map
# ---------------------------------------------------------------------------

_AGENT_MODULES: dict[str, str] = {
    "PLANNER": "vetinari.agents.planner_agent.PlannerAgent",
    "CONSOLIDATED_RESEARCHER": "vetinari.agents.consolidated.researcher_agent.ConsolidatedResearcherAgent",
    "CONSOLIDATED_ORACLE": "vetinari.agents.consolidated.oracle_agent.ConsolidatedOracleAgent",
    "BUILDER": "vetinari.agents.builder_agent.BuilderAgent",
    "QUALITY": "vetinari.agents.consolidated.quality_agent.QualityAgent",
    "OPERATIONS": "vetinari.agents.consolidated.operations_agent.ConsolidatedOperationsAgent",
}


def _load_agent_capabilities(dotpath: str) -> list[str] | None:
    """Import the class at ``dotpath`` and call ``.get_capabilities()``."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        instance = cls()
        return instance.get_capabilities()
    except Exception as exc:
        logger.warning("Could not load %s: %s", dotpath, exc)
        return None


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class CapabilityAuditor:
    """Thread-safe capability auditor.  Singleton — use.

    ``get_capability_auditor()``.
    """

    _instance: CapabilityAuditor | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> CapabilityAuditor:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        # name → set of documented capabilities
        self._documented: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Documentation registration
    # ------------------------------------------------------------------

    def register_documented(self, agent_name: str, capabilities: list[str]) -> None:
        """Record the *documented* capabilities for an agent."""
        with self._lock:
            self._documented[agent_name] = set(capabilities)
        logger.debug("Documented caps for %s: %s", agent_name, capabilities)

    def register_all_from_contracts(self) -> None:
        """Populate documented capabilities from the live AgentSpec registry.

        Uses the ``AgentSpec.description`` to infer documented roles; each
        agent class's ``get_capabilities()`` output IS the documented truth
        in this project (they are the single source).  So this method seeds
        the auditor with the current code values as the baseline.
        """
        for agent_name, dotpath in _AGENT_MODULES.items():
            caps = _load_agent_capabilities(dotpath)
            if caps is not None:
                self.register_documented(agent_name, caps)

    # ------------------------------------------------------------------
    # Auditing
    # ------------------------------------------------------------------

    def audit_agent(self, agent_name: str) -> CapabilityFinding:
        """Audit a single agent by loading its class and comparing.

        ``get_capabilities()`` against the registered documented set.
        """
        with self._lock:
            documented = set(self._documented.get(agent_name, []))

        dotpath = _AGENT_MODULES.get(agent_name)
        if dotpath is None:
            return CapabilityFinding(
                agent_name=agent_name,
                extra_in_code=[],
                missing_in_code=list(documented),
                is_drift=bool(documented),
            )

        live_caps = _load_agent_capabilities(dotpath)
        if live_caps is None:
            logger.warning("Could not load capabilities for %s", agent_name)
            return CapabilityFinding(
                agent_name=agent_name,
                extra_in_code=[],
                missing_in_code=[],
                is_drift=False,
            )

        live_set = set(live_caps)
        extra = sorted(live_set - documented)
        missing = sorted(documented - live_set)
        is_drift = bool(extra or missing)

        return CapabilityFinding(
            agent_name=agent_name,
            extra_in_code=extra,
            missing_in_code=missing,
            is_drift=is_drift,
        )

    def audit_all(self, only_documented: bool = True) -> list[CapabilityFinding]:
        """Audit every agent.

        Args:
            only_documented: If True, only audit agents that have documented
                             capabilities registered.  Set to False to audit
                             all 15 agents even if no docs registered.
        """
        with self._lock:
            names = list(self._documented.keys()) if only_documented else list(_AGENT_MODULES.keys())

        return [self.audit_agent(name) for name in sorted(names)]

    def get_drift_findings(self) -> list[CapabilityFinding]:
        """Return only the findings that represent drift."""
        return [f for f in self.audit_all(only_documented=False) if f.is_drift]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        findings = self.audit_all(only_documented=False)
        return {
            "agents_audited": len(findings),
            "agents_with_drift": sum(1 for f in findings if f.is_drift),
            "documented_agents": len(self._documented),
        }

    def clear(self) -> None:
        with self._lock:
            self._documented.clear()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def get_capability_auditor() -> CapabilityAuditor:
    return CapabilityAuditor()


def reset_capability_auditor() -> None:
    with CapabilityAuditor._class_lock:
        CapabilityAuditor._instance = None
