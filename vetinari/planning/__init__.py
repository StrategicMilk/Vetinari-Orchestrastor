"""Planning, decomposition, and task management subsystem.

Canonical module ownership:
  - ``vetinari.planning.plan_mode``   — plan generation (PlanModeEngine); use this for
    creating new plans from a goal string.
  - ``vetinari.planning.plan_types``  — all planning-domain types: Plan, Subtask,
    PlanRiskLevel, DefinitionOfDone, etc.
  - ``vetinari.planning.plan_api``    — REST layer; Flask Blueprint for all
    /api/v1/plans/* endpoints.
  - ``vetinari.planning.decomposition`` — task decomposition from a high-level plan into
    typed Subtask objects.
  - ``vetinari.planning.plan_validator`` — structural and semantic plan validation;
    runs after goal decomposition and before execution plan creation.

The ``planning.planning`` submodule is DEPRECATED legacy code (wave-based plan
management).  Do not import from it directly; use the canonical modules above.
"""

from __future__ import annotations

import warnings

from vetinari.planning.decision_tree import (
    DecisionNode,
    DecisionTreeResult,
    Option,
    extract_decisions,
)
from vetinari.planning.decomposition import DecompositionEngine, SubtaskSpec
from vetinari.planning.plan_types import (  # noqa: VET123 - barrel export preserves public import compatibility
    DefinitionOfDone,
    DefinitionOfReady,
    PlanApprovalRequest,
    PlanGenerationRequest,
    PlanRiskLevel,
    TaskDomain,
    TaskRationale,
)
from vetinari.planning.plan_validator import (
    ValidationResult,
    validate_plan,
)

# Suppress the DeprecationWarning from our own submodule re-export — callers
# importing from `vetinari.planning` should not see the internal deprecation.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from vetinari.planning.planning import (
        Plan,
        PlanManager,
        PlanningExecutionPlan,
        Wave,
        WaveStatus,
        get_plan_manager,
    )

from vetinari.planning.subtask_tree import SubtaskTree

__all__ = [
    "DecisionNode",
    "DecisionTreeResult",
    "DecompositionEngine",
    "DefinitionOfDone",
    "DefinitionOfReady",
    "Option",
    "Plan",
    "PlanApprovalRequest",
    "PlanGenerationRequest",
    "PlanManager",
    "PlanRiskLevel",
    "PlanningExecutionPlan",
    "SubtaskSpec",
    "SubtaskTree",
    "TaskDomain",
    "TaskRationale",
    "ValidationResult",
    "Wave",
    "WaveStatus",
    "extract_decisions",
    "get_plan_manager",
    "validate_plan",
]
