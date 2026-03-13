"""Vetinari Agent Compatibility Shim (v0.4.0).

============================================
Re-exports legacy agent class names with deprecation warnings, mapping
them to the consolidated 6-agent architecture.

Legacy code that imports e.g. ``from vetinari.agents.explorer_agent import ExplorerAgent``
should update to ``from vetinari.agents.compat import ExplorerAgent`` (or preferably
use the consolidated agent directly).

All legacy singletons (get_*_agent) are wired to the appropriate consolidated agent.

Uses lazy imports so that importing a single legacy alias does NOT trigger
loading every consolidated agent module.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any


def _deprecation(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in v0.5.0. Use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# ── Lazy loaders ────────────────────────────────────────────────────


def _get_planner():
    m = importlib.import_module("vetinari.agents.planner_agent")
    return m.PlannerAgent, m.get_planner_agent


def _get_builder():
    m = importlib.import_module("vetinari.agents.builder_agent")
    return m.BuilderAgent, m.get_builder_agent


def _get_researcher():
    m = importlib.import_module("vetinari.agents.consolidated.researcher_agent")
    return m.ConsolidatedResearcherAgent, m.get_consolidated_researcher_agent


def _get_oracle():
    m = importlib.import_module("vetinari.agents.consolidated.oracle_agent")
    return m.ConsolidatedOracleAgent, m.get_consolidated_oracle_agent


def _get_quality():
    m = importlib.import_module("vetinari.agents.consolidated.quality_agent")
    return m.QualityAgent, m.get_quality_agent


def _get_operations():
    m = importlib.import_module("vetinari.agents.consolidated.operations_agent")
    return m.OperationsAgent, m.get_operations_agent


# ── Mapping from legacy names to lazy loaders ────────────────────────
# Key: (attribute_name) -> (loader_fn, index)  where index 0=class, 1=getter
_LEGACY_CLASS_MAP = {
    # -> ConsolidatedResearcherAgent
    "ExplorerAgent": _get_researcher,
    "LibrarianAgent": _get_researcher,
    "ResearcherAgent": _get_researcher,
    "DataEngineerAgent": _get_researcher,
    "UIPlannerAgent": _get_researcher,
    "DevOpsAgent": _get_researcher,
    "VersionControlAgent": _get_researcher,
    "ArchitectAgent": _get_researcher,
    # -> ConsolidatedOracleAgent
    "OracleAgent": _get_oracle,
    # -> QualityAgent
    "EvaluatorAgent": _get_quality,
    "SecurityAuditorAgent": _get_quality,
    "TestAutomationAgent": _get_quality,
    # -> OperationsAgent
    "SynthesizerAgent": _get_operations,
    "DocumentationAgent": _get_operations,
    "CostPlannerAgent": _get_operations,
    "ExperimentationManagerAgent": _get_operations,
    "ImprovementAgent": _get_operations,
    "ErrorRecoveryAgent": _get_operations,
    # -> PlannerAgent
    "UserInteractionAgent": _get_planner,
    "ContextManagerAgent": _get_planner,
    "OrchestratorAgent": _get_planner,
    # -> BuilderAgent
    "ImageGeneratorAgent": _get_builder,
    # Consolidated classes themselves (for direct access)
    "PlannerAgent": _get_planner,
    "BuilderAgent": _get_builder,
    "ConsolidatedResearcherAgent": _get_researcher,
    "ConsolidatedOracleAgent": _get_oracle,
    "QualityAgent": _get_quality,
    "OperationsAgent": _get_operations,
}

_LEGACY_GETTER_MAP = {
    "get_explorer_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_librarian_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_researcher_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_oracle_agent": ("get_consolidated_oracle_agent", _get_oracle),
    "get_evaluator_agent": ("get_quality_agent", _get_quality),
    "get_synthesizer_agent": ("get_operations_agent", _get_operations),
    "get_ui_planner_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_security_auditor_agent": ("get_quality_agent", _get_quality),
    "get_data_engineer_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_documentation_agent": ("get_operations_agent", _get_operations),
    "get_cost_planner_agent": ("get_operations_agent", _get_operations),
    "get_test_automation_agent": ("get_quality_agent", _get_quality),
    "get_experimentation_manager_agent": ("get_operations_agent", _get_operations),
    "get_improvement_agent": ("get_operations_agent", _get_operations),
    "get_user_interaction_agent": ("get_planner_agent", _get_planner),
    "get_devops_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_version_control_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_error_recovery_agent": ("get_operations_agent", _get_operations),
    "get_context_manager_agent": ("get_planner_agent", _get_planner),
    "get_image_generator_agent": ("get_builder_agent", _get_builder),
    "get_orchestrator_agent": ("get_planner_agent", _get_planner),
    "get_architect_agent": ("get_consolidated_researcher_agent", _get_researcher),
    # Consolidated getters (direct access)
    "get_planner_agent": ("get_planner_agent", _get_planner),
    "get_builder_agent": ("get_builder_agent", _get_builder),
    "get_consolidated_researcher_agent": ("get_consolidated_researcher_agent", _get_researcher),
    "get_consolidated_oracle_agent": ("get_consolidated_oracle_agent", _get_oracle),
    "get_quality_agent": ("get_quality_agent", _get_quality),
    "get_operations_agent": ("get_operations_agent", _get_operations),
}


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of legacy agent names."""
    # Class aliases
    if name in _LEGACY_CLASS_MAP:
        loader = _LEGACY_CLASS_MAP[name]
        cls, _ = loader()
        # Cache on module for subsequent access
        globals()[name] = cls
        return cls

    # Getter functions — return a wrapper that issues deprecation + delegates
    if name in _LEGACY_GETTER_MAP:
        new_name, loader = _LEGACY_GETTER_MAP[name]

        def _getter(config: dict[str, Any] | None = None, _old=name, _new=new_name, _ldr=loader):
            _deprecation(_old, _new)
            _, real_getter = _ldr()
            return real_getter(config)

        # Cache on module
        globals()[name] = _getter
        return _getter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
