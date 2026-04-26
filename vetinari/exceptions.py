"""Vetinari Exception Hierarchy.

=============================

Centralized exception definitions. Import from here rather than defining
ad-hoc exceptions throughout the codebase.

Usage::

    from vetinari.exceptions import InferenceError, ConfigurationError

    raise InferenceError("Model not responding", model_id="qwen-7b")
"""

from __future__ import annotations


class VetinariError(Exception):
    """Base exception for all Vetinari errors."""

    def __init__(self, message: str = "", **context):
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------


class ConfigurationError(VetinariError):
    """Invalid or missing configuration."""


class StorageError(VetinariError):
    """Filesystem or persistence layer failure."""


# ---------------------------------------------------------------------------
# Inference / adapter errors
# ---------------------------------------------------------------------------


class InferenceError(VetinariError):
    """LLM inference failed after all retries."""


class AdapterError(VetinariError):
    """Adapter-level communication failure."""


class ModelNotFoundError(InferenceError):
    """Requested model is not loaded or available."""


class ModelUnavailableError(InferenceError):
    """No model is available to serve inference requests."""


class VetinariTimeoutError(InferenceError):
    """Inference timed out.

    Named to avoid shadowing the built-in ``TimeoutError``.
    """


# Backward-compatible alias — prefer VetinariTimeoutError in new code.
InferenceTimeoutError = VetinariTimeoutError


# ---------------------------------------------------------------------------
# Agent / orchestration errors
# ---------------------------------------------------------------------------


class AgentError(VetinariError):
    """An agent failed to execute its task."""


class PlanningError(AgentError):
    """Plan generation or decomposition failed."""


class ExecutionError(AgentError):
    """Task execution failed."""


class VerificationError(AgentError):
    """Output verification failed quality checks."""


class CircularDependencyError(PlanningError):
    """Execution graph contains circular dependencies."""


# ---------------------------------------------------------------------------
# Security / safety errors
# ---------------------------------------------------------------------------


class SecurityError(VetinariError):
    """Security policy violation."""


class SandboxError(SecurityError):
    """Sandbox constraint violation."""


class SandboxPolicyViolation(SandboxError):
    """A sandboxed operation attempted an action the policy forbids.

    Distinct from SandboxError so callers can match specifically on policy
    violations (e.g., blocked path writes) without catching every sandbox
    constraint failure.
    """


class GuardrailError(SecurityError):
    """Input/output guardrail triggered."""


# ---------------------------------------------------------------------------
# Enforcement errors
# ---------------------------------------------------------------------------


class CapabilityNotAvailable(AgentError):
    """Agent attempted an action outside its declared capabilities."""


class DelegationDepthExceeded(AgentError):
    """Agent delegation chain exceeded maximum allowed depth."""


class JurisdictionViolation(AgentError):
    """Agent attempted to modify files outside its jurisdiction."""


class QualityGateFailed(AgentError):
    """Output did not meet required quality gate thresholds."""


class CompositeEnforcementError(AgentError):
    """Multiple enforcement violations detected in a single check.

    Collects all violations rather than failing on the first one, so callers
    can see the full list of issues.

    Args:
        violations: List of individual exception instances.
    """

    def __init__(self, violations: list[Exception], **context) -> None:
        self.violations = violations
        messages = [str(v) for v in violations]
        combined = f"{len(violations)} enforcement violation(s): " + "; ".join(messages)
        super().__init__(combined, **context)


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) errors
# ---------------------------------------------------------------------------


class MCPError(VetinariError):
    """Error in MCP client/server communication."""


# ---------------------------------------------------------------------------
# Evidence / claim-gate errors
# ---------------------------------------------------------------------------


class RecycleFailedAbort(VetinariError):
    """Raised when RecycleStore.retire fails and the destructive op is aborted.

    The decorated function is never called.  The original target is untouched
    because the recycle store rolls back on failure.  The caller should surface
    this to the user with guidance to investigate disk/permission issues before
    retrying the destructive operation.

    Args:
        message: Human-readable description of why the recycle failed.
        target: String path of the file/directory that could not be recycled.
    """


class InsufficientEvidenceError(VetinariError):
    """Raised when an OutcomeSignal cannot be constructed due to inadequate evidence.

    The most common trigger is constructing a HUMAN_ATTESTED OutcomeSignal
    without any AttestedArtifact on a path that requires tool-backed evidence
    (e.g., promotion audit, release proof, high-accuracy factual claim closure).

    Bare human attestation ("a user said yes") is valid only for intent
    confirmation (destructive-op consent, override appeal) and must be
    accompanied by ``use_case="INTENT_CONFIRMATION"`` on the signal.

    Args:
        message: Human-readable description of what evidence is missing.
        basis: The EvidenceBasis value that triggered the error.
        use_case: The use-case label on the signal being constructed.
    """
