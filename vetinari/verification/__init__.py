"""Verification subsystem — claim-level verification, consistency checking, and AST analysis.

The active implementation is :class:`vetinari.validation.verification.CascadeOrchestrator`
(3-tier cascade: static → entailment → LLM). This package re-exports it for
backwards compatibility.

Also exports:
- Sandbox-based code verification (:mod:`vetinari.verification.sandbox_verifier`)
- Claim extraction and entailment checking (:mod:`vetinari.verification.claim_extractor`,
  :mod:`vetinari.verification.entailment_checker`) for the Inspector pipeline (US-013)
- Implementation consistency checking (:mod:`vetinari.verification.consistency_checker`)
  for detecting inconsistent patterns across source files (US-014)
- AST-based code analysis (:mod:`vetinari.verification.ast_analyzer`) for structure
  extraction, dead code detection, and complexity hotspots (US-015)
- Fail-closed claim verification hooks (:mod:`vetinari.verification.claim_verifier`)
  for the six failure modes required by SESSION-05 SHARD-03
- Verification cascade report (:mod:`vetinari.verification.cascade_report`)
  with an iterable ``unsupported_claims`` top-level field (Task 3.2)
"""

from __future__ import annotations

from vetinari.validation.verification import (
    CascadeOrchestrator,
    CascadeVerdict,
    get_cascade_orchestrator,
)
from vetinari.verification.ast_analyzer import (
    AstAnalysisResult,
    SymbolKind,
    analyze_source,
    get_function_defs,
    get_import_graph,
)
from vetinari.verification.cascade_report import (
    CascadeClaimRecord,
    VerificationCascadeReport,
    build_cascade_report,
)
from vetinari.verification.claim_extractor import (
    Claim,
    ClaimType,
    extract_claims,
    extract_claims_by_type,
)
from vetinari.verification.claim_verifier import (
    DEFAULT_FRESHNESS_WINDOW_SECONDS,
    verify_citation_present,
    verify_claim_fail_closed,
    verify_evidence_freshness,
    verify_file_claim,
    verify_human_attestation,
    verify_no_entailment_contradiction,
    verify_not_llm_only,
)
from vetinari.verification.consistency_checker import (
    ConsistencyIssue,
    PatternCategory,
    check_consistency,
    check_consistency_across_files,
    extract_patterns,
)
from vetinari.verification.entailment_checker import (
    ClaimVerdict,
    VerificationReport,
    verify_claim,
    verify_claims,
)
from vetinari.verification.sandbox_verifier import (
    SandboxFailure,
    SandboxVerification,
    cleanup_sandbox_artifacts,
    verify_code,
    verify_code_safe,
)

__all__ = [
    "DEFAULT_FRESHNESS_WINDOW_SECONDS",
    "AstAnalysisResult",
    "CascadeClaimRecord",
    "CascadeOrchestrator",
    "CascadeVerdict",
    "Claim",
    "ClaimType",
    "ClaimVerdict",
    "ConsistencyIssue",
    "PatternCategory",
    "SandboxFailure",
    "SandboxVerification",
    "SymbolKind",
    "VerificationCascadeReport",
    "VerificationReport",
    "analyze_source",
    "build_cascade_report",
    "check_consistency",
    "check_consistency_across_files",
    "cleanup_sandbox_artifacts",
    "extract_claims",
    "extract_claims_by_type",
    "extract_patterns",
    "get_cascade_orchestrator",
    "get_function_defs",
    "get_import_graph",
    "verify_citation_present",
    "verify_claim",
    "verify_claim_fail_closed",
    "verify_claims",
    "verify_code",
    "verify_code_safe",
    "verify_evidence_freshness",
    "verify_file_claim",
    "verify_human_attestation",
    "verify_no_entailment_contradiction",
    "verify_not_llm_only",
]
