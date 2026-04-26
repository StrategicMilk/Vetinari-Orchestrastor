"""Vetinari safety module — guardrails, content filtering, and lifecycle guards."""

from __future__ import annotations

from vetinari.safety.agent_monitor import AgentMonitor, AgentTimeoutError, StepLimitExceeded, get_agent_monitor
from vetinari.safety.guardrails import GuardrailsManager, get_guardrails, reset_guardrails
from vetinari.safety.nemo_provider import NeMoGuardrailsProvider, get_nemo_provider, reset_nemo_provider
from vetinari.safety.policy_enforcer import PolicyDecision, PolicyEnforcer, get_policy_enforcer
from vetinari.safety.prompt_sanitizer import (
    is_content_delimited,
    sanitize_task_description,
    sanitize_worker_output,
)
from vetinari.safety.protected_mutation import (
    ConfirmedIntent,
    DestructiveAction,
    UnconfirmedDestructiveAction,
    protected_mutation,
)
from vetinari.safety.recycle import RecycleStore

__all__ = [
    "AgentMonitor",
    "AgentTimeoutError",
    "ConfirmedIntent",
    "DestructiveAction",
    "GuardrailsManager",
    "NeMoGuardrailsProvider",
    "PolicyDecision",
    "PolicyEnforcer",
    "RecycleStore",
    "StepLimitExceeded",
    "UnconfirmedDestructiveAction",
    "get_agent_monitor",
    "get_guardrails",
    "get_nemo_provider",
    "get_policy_enforcer",
    "is_content_delimited",
    "protected_mutation",
    "reset_guardrails",
    "reset_nemo_provider",
    "sanitize_task_description",
    "sanitize_worker_output",
]
