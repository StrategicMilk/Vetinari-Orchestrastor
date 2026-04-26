"""Validation, verification, and goal checking subsystem."""

from __future__ import annotations

from vetinari.validation.document_judge import DocumentJudge, JudgeConfig
from vetinari.validation.document_quality import (  # noqa: VET123 - barrel export preserves public import compatibility
    DimensionScore,
    QualityReport,
    evaluate_document,
)
from vetinari.validation.document_types import (
    DocumentProfile,
    DocumentType,
    get_profile_for_type,
    load_document_profiles,
)
from vetinari.validation.entailment_checker import EntailmentChecker, EntailmentResult
from vetinari.validation.goal_verifier import (
    GoalVerificationReport,
    GoalVerifier,
    get_goal_verifier,
)
from vetinari.validation.prevention import CheckResult, PreventionGate, PreventionResult
from vetinari.validation.root_cause import (
    CausalEdge,
    CausalGraph,
    DefectCategory,
    RootCauseAnalysis,
    RootCauseAnalyzer,
    build_causal_graph,
    walk_graph_for_root_cause,
)
from vetinari.validation.static_verifier import StaticCheckResult, StaticVerifier
from vetinari.validation.verification import (  # noqa: VET123 - barrel export preserves public import compatibility
    CascadeOrchestrator,
    CascadeVerdict,
    CodeSyntaxVerifier,
    ImportVerifier,
    JSONStructureVerifier,
    SecurityVerifier,
    ValidationVerificationResult,
    Validator,
    VerificationIssue,
    VerificationLevel,
    VerificationPipeline,
    VerificationStatus,
    Verifier,
    get_cascade_orchestrator,
    get_verifier_pipeline,
)
from vetinari.validation.wiring import (
    StageGateResult,
    VerificationSummary,
    run_stage_gate,
    verify_worker_output,
    wire_validation_subsystem,
)

__all__ = [
    "CascadeOrchestrator",
    "CascadeVerdict",
    "CausalEdge",
    "CausalGraph",
    "CheckResult",
    "CodeSyntaxVerifier",
    "DefectCategory",
    "DimensionScore",
    "DocumentJudge",
    "DocumentProfile",
    "DocumentType",
    "EntailmentChecker",
    "EntailmentResult",
    "GoalVerificationReport",
    "GoalVerifier",
    "ImportVerifier",
    "JSONStructureVerifier",
    "JudgeConfig",
    "PreventionGate",
    "PreventionResult",
    "QualityReport",
    "RootCauseAnalysis",
    "RootCauseAnalyzer",
    "SecurityVerifier",
    "StageGateResult",
    "StaticCheckResult",
    "StaticVerifier",
    "ValidationVerificationResult",
    "Validator",
    "VerificationIssue",
    "VerificationLevel",
    "VerificationPipeline",
    "VerificationStatus",
    "VerificationSummary",
    "Verifier",
    "build_causal_graph",
    "evaluate_document",
    "get_cascade_orchestrator",
    "get_goal_verifier",
    "get_profile_for_type",
    "get_verifier_pipeline",
    "load_document_profiles",
    "run_stage_gate",
    "verify_worker_output",
    "walk_graph_for_root_cause",
    "wire_validation_subsystem",
]
