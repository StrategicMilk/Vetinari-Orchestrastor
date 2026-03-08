"""
Vetinari Exception Hierarchy
=============================

Centralized exception definitions. Import from here rather than defining
ad-hoc exceptions throughout the codebase.

Usage::

    from vetinari.exceptions import InferenceError, ConfigurationError

    raise InferenceError("Model not responding", model_id="qwen-7b")
"""


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


class TimeoutError(InferenceError):
    """Inference timed out."""


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


class GuardrailError(SecurityError):
    """Input/output guardrail triggered."""


# ---------------------------------------------------------------------------
# Analytics / drift errors
# ---------------------------------------------------------------------------


class DriftError(VetinariError):
    """Schema or contract drift detected."""


class SLABreachError(VetinariError):
    """SLA target breached."""
